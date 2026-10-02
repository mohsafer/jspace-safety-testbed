"""End-to-end smoke test of the jspace testbed (CPU-friendly).

For each model (base + instruct pair):
  1. fit the Jacobian lens on calibration prompts
  2. sanity check: last-layer lens readout ~= the model's own logits
  3. read out J-space top-k tokens for harmful/benign prompts (last prompt position)
  4. compute mini-JADR metrics (SafetyAUC, ComplAUC, Headroom, per-layer AUC, CIs)
  5. compute the latent<->J-space bridge: per-layer safety direction (diff-in-means)
     and its transport amplification under that layer's lens
  6. dump JSON + CSV + plot into artifacts/smoke/

Run:  .venv/bin/python -m src.run_smoke --models smol
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .jlens import fit_lens, lens_logits_at, direction_transport
from .metrics import (
    PromptReadout, compute_readout, safety_auc, per_layer_auc, headroom,
    bootstrap_ci, auc,
)
from .prompts import HARMFUL, BENIGN, CALIBRATION_PROMPTS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "smoke"

MODELS = {
    "smol135": [
        ("HuggingFaceTB/SmolLM2-135M-Instruct", True),
        ("HuggingFaceTB/SmolLM2-135M", False),
    ],
    "smol360": [
        ("HuggingFaceTB/SmolLM2-360M-Instruct", True),
        ("HuggingFaceTB/SmolLM2-360M", False),
    ],
    "qwen05": [
        ("Qwen/Qwen2.5-0.5B-Instruct", True),
        ("Qwen/Qwen2.5-0.5B", False),
    ],
}


def prep_ids(tok, text: str, is_instruct: bool) -> torch.Tensor:
    if is_instruct and getattr(tok, "chat_template", None):
        s = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        return tok(s, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return tok(text, return_tensors="pt").input_ids[0]


def run_model(name: str, is_instruct: bool, device: str, n_cot: int, k: int):
    t0 = time.time()
    torch.manual_seed(0)  # deterministic cotangents → reproducible lens fits
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    print(f"[{name}] loaded ({time.time()-t0:.0f}s)")

    # 1. fit lens
    calib_ids = [prep_ids(tok, p, is_instruct) for p in CALIBRATION_PROMPTS]
    calib_ids = [ids for ids in calib_ids if ids.shape[0] >= 4]
    t0 = time.time()
    lenses = fit_lens(model, calib_ids, n_cotangents=n_cot, device=device)
    print(f"[{name}] lens fit on {len(calib_ids)} prompts x {n_cot} cotangents "
          f"({time.time()-t0:.0f}s)")

    # 2. sanity check on one calibration prompt
    ids = calib_ids[0].to(device)
    with torch.no_grad():
        out = model(ids[None], output_hidden_states=True)
        actual = out.logits[0, -1]
        lens_last = lens_logits_at(model, lenses, out.hidden_states, -1)[-1]
    cos = torch.nn.functional.cosine_similarity(actual, lens_last, dim=0).item()
    print(f"[{name}] sanity: cos(last-layer lens, model logits) = {cos:.4f}")

    # 3. readouts for harmful/benign
    readouts: list[PromptReadout] = []
    last_hiddens = []  # per prompt: [n_layers, d] hidden at last prompt position
    for label, prompts in (("harmful", HARMFUL), ("benign", BENIGN)):
        for p in prompts:
            ids = prep_ids(tok, p, is_instruct).to(device)
            with torch.no_grad():
                out = model(ids[None], output_hidden_states=True)
                hs = torch.stack([h[0, -1] for h in out.hidden_states]).cpu()
            last_hiddens.append(hs)
            ro = compute_readout(model, tok, lenses, ids.cpu(), label, k=k)
            ro.prompt = p
            readouts.append(ro)

    # 4. metrics
    sauc = safety_auc(readouts, "safety")
    cauc = safety_auc(readouts, "compliance")
    hauc = safety_auc(readouts, "harm")
    lo, hi = bootstrap_ci(readouts, "safety")
    pla = per_layer_auc(readouts, "safety")

    # 5. bridge metric: per-layer diff-in-means direction + transport amplification
    labels = torch.tensor([1 if r.label == "harmful" else 0 for r in readouts])
    H = torch.stack(last_hiddens)                      # [P, n_layers, d]
    unembed = model.get_output_embeddings().weight.to(device)
    n_layers = H.shape[1]
    amps = []
    dir_norms = []
    for l in range(n_layers):
        v = (H[labels == 1, l].mean(0) - H[labels == 0, l].mean(0)).to(device)
        v = v / v.norm().clamp_min(1e-12)
        dir_norms.append(float(v.norm()))
        a = direction_transport(lenses[l].to(device), unembed, v)
        if a != a:  # NaN tripwire: dump diagnostics for the offending layer
            J = lenses[l]
            print(f"[{name}] NaN amp @L{l}: |v|={v.norm().item():.3e} "
                  f"J_finite={bool(torch.isfinite(J).all())} "
                  f"J_absmax={J.abs().max().item():.3e} "
                  f"H_finite={bool(torch.isfinite(H[:, l]).all())} "
                  f"H_absmax={H[:, l].abs().max().item():.3e}")
            t1 = unembed.double() @ (J.double() @ v.double())
            print(f"           base={t1.norm().item():.3e} "
                  f"t1_absmax={t1.abs().max().item():.3e}")
        amps.append(a)
    amps = torch.tensor(amps)

    res = {
        "model": name, "is_instruct": is_instruct,
        "sanity_cos_last_layer": cos,
        "safety_auc": sauc, "safety_ci95": [lo, hi],
        "compliance_auc": cauc, "harm_auc": hauc,
        "headroom": headroom(sauc, cauc),
        "amp_peak": float(amps.max()), "amp_peak_layer": int(amps.argmax()),
        "auc_peak": float(pla.max()), "auc_peak_layer": int(pla.argmax()),
        "corr(amp,per-layer-AUC)": float(torch.corrcoef(torch.stack([amps, pla]))[0, 1]),
        "seconds": time.time() - t0,
    }
    print(f"[{name}] " + json.dumps(res, indent=2))

    # per-prompt rows
    rows = []
    for r in readouts:
        rows.append({
            "model": name, "label": r.label, "prompt": r.prompt,
            **{f"T_{a}": r.T[a] for a in r.T},
        })
    return res, rows, pla, amps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="smol135", choices=list(MODELS) + ["all"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cotangents", type=int, default=8)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    which = list(MODELS) if args.models == "all" else [args.models]

    all_res, all_rows = [], []
    curves = {}
    for key in which:
        for name, is_instr in MODELS[key]:
            res, rows, pla, amps = run_model(
                name, is_instr, args.device, args.cotangents, args.k
            )
            all_res.append(res)
            all_rows.extend(rows)
            curves[name] = {"per_layer_safety_auc": pla.tolist(),
                            "amplification": amps.tolist()}

    with open(OUT / "summary.json", "w") as f:
        json.dump({"results": all_res, "curves": curves}, f, indent=2)
    with open(OUT / "readouts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, c in curves.items():
            ax[0].plot(c["per_layer_safety_auc"], label=name)
            ax[1].plot(c["amplification"], label=name)
        ax[0].axhline(0.5, color="gray", ls="--", lw=0.8)
        ax[0].set_title("per-layer SafetyAUC (J-space danger recognition)")
        ax[1].set_title("direction transport amplification ||U·J_l·d||/null")
        ax[1].axhline(1.0, color="gray", ls="--", lw=0.8)
        for a in ax:
            a.legend(fontsize=7)
            a.set_xlabel("layer")
        fig.tight_layout()
        fig.savefig(OUT / "smoke_results.png", dpi=140)
        print(f"saved {OUT}/smoke_results.png")
    except Exception as e:  # matplotlib optional
        print("plot skipped:", e)

    print("\n=== SMOKE SUMMARY ===")
    for r in all_res:
        tag = "instruct" if r["is_instruct"] else "base    "
        print(f"{tag} {r['model']:42s} SafetyAUC={r['safety_auc']:.3f} "
              f"CI[{r['safety_ci95'][0]:.3f},{r['safety_ci95'][1]:.3f}] "
              f"Headroom={r['headroom']:+.3f} amp_peak={r['amp_peak']:.2f} "
              f"@L{r['amp_peak_layer']} corr={r['corr(amp,per-layer-AUC)']:+.2f}")


if __name__ == "__main__":
    main()

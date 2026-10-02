"""P1: full-dataset mini-JADR + base-vs-instruct, multi-seed lens fits (CPU).

Protocol (per model x seed):
  1. fit the Jacobian lens on the calibration corpus (seeded cotangents)
  2. sanity check: last-layer lens readout == model logits (cos)
  3. at the last prompt token (JADR decision point), read out top-k lens tokens
     for StrongREJECT (danger) + XSTest-safe (benign); six-axis DCG counters
  4. SafetyAUC / ComplAUC / HarmAUC (+ all axes), Headroom, bootstrap CIs,
     per-layer AUC curves
  5. latent<->J-space bridge: per-layer diff-in-means safety direction and its
     transport amplification under that layer's lens

Resumable: one JSON+NPZ per (model, seed, calib, window); existing files are
skipped unless --force. Aggregate with --aggregate.

Run:  .venv/bin/python -m src.run_p1 --pairs smol135 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build_calibration, ensure_eval_sets
from .jlens import direction_transport, fit_lens, lens_logits_at
from .lexicons import LEXICONS_FULL, build_token_hits
from .metrics import auc_ranks, headroom
from .run_smoke import prep_ids

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "p1"

MODELS_P1 = {
    "smol135": [
        ("HuggingFaceTB/SmolLM2-135M-Instruct", "smol135m_instruct", True),
        ("HuggingFaceTB/SmolLM2-135M", "smol135m_base", False),
    ],
    "smol360": [
        ("HuggingFaceTB/SmolLM2-360M-Instruct", "smol360m_instruct", True),
        ("HuggingFaceTB/SmolLM2-360M", "smol360m_base", False),
    ],
    "qwen05": [
        ("Qwen/Qwen2.5-0.5B-Instruct", "qwen05b_instruct", True),
        ("Qwen/Qwen2.5-0.5B", "qwen05b_base", False),
    ],
    "qwen15": [
        ("Qwen/Qwen2.5-1.5B-Instruct", "qwen15b_instruct", True),
        ("Qwen/Qwen2.5-1.5B", "qwen15b_base", False),
    ],
    "qwen34": [
        ("Qwen/Qwen3-4B-Instruct-2507", "qwen3_4b_instruct", True),
        ("Qwen/Qwen3-4B", "qwen3_4b_base", False),
    ],
    # R3 (research/06): the headline 9B cross-check — full bf16/GPU protocol
    "gemma29": [
        ("google/gemma-2-9b-it", "gemma2_9b_it", True),
        ("google/gemma-2-9b", "gemma2_9b_base", False),
    ],
    # R5: public abliterated checkpoint (huihui-ai editor family, the JADR
    # lineage) paired with its original base — expect ~chance SafetyAUC and
    # partial NF4 recovery (JADR's headline)
    "qwen317abl": [
        ("huihui-ai/Huihui-Qwen3-1.7B-abliterated-v2", "qwen317_abliterated", True),
        ("Qwen/Qwen3-1.7B", "qwen317_base", False),
    ],
}
AXES = list(LEXICONS_FULL)
MAIN_AXES = ("safety", "compliance", "harm")


def boot_ci(h_vals, b_vals, B: int = 500, seed: int = 0):
    """Percentile bootstrap CI for the Mann-Whitney AUC (vectorized ranks)."""
    g = torch.Generator().manual_seed(seed)
    h = torch.as_tensor(h_vals, dtype=torch.float64)
    b = torch.as_tensor(b_vals, dtype=torch.float64)
    vals = []
    for _ in range(B):
        ih = torch.randint(0, h.numel(), (h.numel(),), generator=g)
        ib = torch.randint(0, b.numel(), (b.numel(),), generator=g)
        vals.append(auc_ranks(h[ih].tolist(), b[ib].tolist()))
    v = torch.tensor(vals, dtype=torch.float64)
    return float(v.quantile(0.025)), float(v.quantile(0.975))


def run_one(model_name: str, slug: str, is_instruct: bool, seed: int, calib_kind: str,
            n_calib: int, window: int, n_cot: int, k_store: int, k_metric: int,
            force: bool, device: str = "cpu", dtype_name: str = "float32",
            quant: str = "none", use_template: bool = True) -> dict:
    # non-default precision/quantization get their own artifact tags so that
    # fp32 runs (the paper's primary tables) are never overwritten
    tag = f"seed{seed}_calib{calib_kind}_w{window}"
    tag_suffix = []
    if dtype_name != "float32":
        tag_suffix.append(dtype_name)
    if quant != "none":
        tag_suffix.append(f"q{quant}")
    if not use_template:
        tag_suffix.append("notmpl")   # template-confound ablation (M1)
    if tag_suffix:
        tag += "_" + "_".join(tag_suffix)
    out_dir = OUT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{tag}.json"
    # complete = the JSON exists with its completion marker. (Requiring the NPZ
    # sidecar too once caused GPU re-evals to silently overwrite committed CPU
    # artifacts on a fresh clone — the npz is regenerable and NOT part of the
    # metric record.)
    if (out_json.exists()
            and "sanity_cos_last_layer" in json.loads(out_json.read_text())
            and not force):
        print(f"[{slug}/{tag}] exists, skipping", flush=True)
        return json.loads(out_json.read_text())

    t_start = time.time()
    torch.manual_seed(seed)
    dtype = getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    # fp16 (no quant) runs as autocast: fp32 master weights, fp16 compute —
    # the deployment-realistic numerics without fp16-gradient overflow in the
    # lens fit (true fp16 weights crashed the backward on 24-32-layer models)
    amp = (dtype != torch.float32 and quant == "none" and "cuda" in device
           and dtype == torch.float16)
    from contextlib import nullcontext
    fwd_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if amp else nullcontext

    if quant == "none":
        # amp path: fp32 master weights (fp16-weight graphs overflow the lens
        # backward); autocast supplies the fp16 compute at readout time
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32 if amp else dtype).to(device).eval()
    else:
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(
            load_in_4bit=(quant == "nf4"), load_in_8bit=(quant == "int8"),
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qconf, device_map=device,
            dtype=dtype).eval()
    print(f"[{slug}/{tag}] loaded on {device}/{dtype_name}"
          + (f"/{quant}" if quant != "none" else "") + f" ({time.time()-t_start:.0f}s)",
          flush=True)

    # big models: gradient-checkpoint the lens fit — exact recompute math, but
    # the autograd graph drops from ~2 GB (9B) to ~0.1 GB, which is the
    # difference between fitting and OOMing on a 24 GB card that already
    # holds the bf16 weights + fp32 accumulators
    if device == "cuda" and quant == "none" \
            and sum(p.numel() for p in model.parameters()) > 3e9:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        print(f"[{slug}/{tag}] gradient checkpointing enabled for the lens fit",
              flush=True)

    # 1. lens fit on calibration corpus
    calib_texts = build_calibration(calib_kind, n_prompts=n_calib)
    calib_ids = [prep_ids(tok, p, is_instruct and use_template)
                 for p in calib_texts]
    calib_ids = [ids for ids in calib_ids if ids.shape[0] >= 8][:n_calib]
    t0 = time.time()
    # lens = the measurement instrument: always fitted in plain fp32 (fp16
    # backward through deep models overflows). The eval readouts below run
    # under autocast fp16 — that is the inference numerics under test.
    lenses = fit_lens(model, calib_ids, n_cotangents=n_cot, target_window=window,
                      device=device, show_progress=False)
    fit_s = time.time() - t0
    print(f"[{slug}/{tag}] lens fit: {len(calib_ids)} prompts x {n_cot} cot in {fit_s:.0f}s",
          flush=True)

    # 2. sanity: last-layer lens readout == the model's own logits
    with torch.no_grad(), fwd_ctx():
        out0 = model(calib_ids[0].to(device)[None], output_hidden_states=True)
        cos = torch.nn.functional.cosine_similarity(
            out0.logits[0, -1].float(),
            lens_logits_at(model, lenses, out0.hidden_states, -1)[-1], dim=0).item()

    # 3. eval readouts at the last prompt token
    ev = ensure_eval_sets()
    tok_hits = build_token_hits(tok, vocab_size=model.config.vocab_size)  # [V, A] bool (cpu)
    top_list, hid_list, labels = [], [], []
    t0 = time.time()
    for label, key in (("harmful", "danger"), ("benign", "benign")):
        for _src, p in ev[key]:
            ids = prep_ids(tok, p, is_instruct and use_template)
            if ids.shape[0] < 2:
                continue
            with torch.no_grad(), fwd_ctx():
                out = model(ids.to(device)[None], output_hidden_states=True)
                hid_list.append(torch.stack(
                    [h[0, -1] for h in out.hidden_states]).cpu())
                lls = lens_logits_at(model, lenses, out.hidden_states,
                                     position=ids.shape[0] - 1)
            top_list.append(torch.stack(
                [lg.topk(k_store).indices for lg in lls]).cpu())
            labels.append(1 if label == "harmful" else 0)
    eval_s = time.time() - t0
    top_tokens = torch.stack(top_list)                  # [P, L, k_store]
    H = torch.stack(hid_list)                           # [P, L, d]
    labels = torch.tensor(labels)
    n_danger = int((labels == 1).sum()); n_benign = int((labels == 0).sum())
    print(f"[{slug}/{tag}] readouts: {n_danger} danger / {n_benign} benign "
          f"in {eval_s:.0f}s", flush=True)

    # 4. six-axis DCG counters at k_metric (vectorized via token-hit table)
    disc = torch.tensor([1.0 / math.log2(r + 1) for r in range(1, k_metric + 1)])
    hits = tok_hits[top_tokens[:, :, :k_metric]].float()          # [P, L, k, A]
    counters = torch.einsum("plka,k->pla", hits, disc)            # [P, L, A]
    T = counters.sum(1)                                           # [P, A]

    res = {
        "model": model_name, "slug": slug, "is_instruct": is_instruct,
        "seed": seed, "calib": calib_kind, "n_calib": len(calib_ids),
        "window": window, "n_cotangents": n_cot, "k_store": k_store,
        "k_metric": k_metric, "dtype": dtype_name, "device": str(device),
        "quant": quant, "autocast": amp, "use_template": use_template,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sanity_cos_last_layer": cos,
        "n_danger": n_danger, "n_benign": n_benign,
        "fit_seconds": round(fit_s, 1), "eval_seconds": round(eval_s, 1),
    }
    mask_h, mask_b = labels == 1, labels == 0
    for ai, axis in enumerate(AXES):
        res[f"auc_{axis}"] = auc_ranks(T[mask_h, ai].tolist(), T[mask_b, ai].tolist())
    res["headroom"] = headroom(res["auc_safety"], res["auc_compliance"])
    lo, hi = boot_ci(T[mask_h, AXES.index("safety")], T[mask_b, AXES.index("safety")],
                     B=500, seed=seed)
    res["safety_ci95"] = [lo, hi]

    per_layer = {}
    for axis in MAIN_AXES:
        ai = AXES.index(axis)
        per_layer[axis] = [auc_ranks(counters[mask_h, l, ai].tolist(),
                                     counters[mask_b, l, ai].tolist())
                           for l in range(counters.shape[1])]
    res["per_layer_auc"] = per_layer

    # 5. bridge: per-layer safety direction (diff-in-means) + transport amplification
    unembed = model.get_output_embeddings().weight
    amps, dirnorms = [], []
    for l in range(H.shape[1]):
        v = H[mask_h, l].mean(0) - H[mask_b, l].mean(0)
        v = v / v.norm().clamp_min(1e-12)
        dirnorms.append(float(v.norm()))
        amps.append(direction_transport(lenses[l], unembed, v, n_null=64, seed=seed))
    amps_clean = [0.0 if a != a else a for a in amps]
    res["amplification"] = [round(a, 4) for a in amps_clean]
    res["amp_peak"] = float(np.max(amps_clean))
    res["amp_peak_layer"] = int(np.argmax(amps_clean))
    res["dir_norm"] = [round(d, 4) for d in dirnorms]
    res["corr(amp,per-layer-AUC)"] = float(torch.corrcoef(torch.stack([
        torch.tensor(amps_clean), torch.tensor(per_layer["safety"])]))[0, 1])

    out_json.write_text(json.dumps(res, indent=1))
    np.savez_compressed(
        out_dir / f"{tag}.npz",
        top_tokens=top_tokens[:, :, :k_store].numpy().astype(np.int32),
        counters=counters.numpy().astype(np.float32),
        labels=labels.numpy(),
    )
    print(f"[{slug}/{tag}] DONE sanity={cos:.4f} SafetyAUC={res['auc_safety']:.3f} "
          f"ComplAUC={res['auc_compliance']:.3f} SH={res['headroom']:+.3f} "
          f"amp_peak={res['amp_peak']:.2f}@L{res['amp_peak_layer']} "
          f"({fit_s:.0f}s fit + {eval_s:.0f}s eval)", flush=True)
    del model
    return res


def _load_runs(slug: str, calib: str = "wikitext", window: int = 0) -> dict[int, dict]:
    out = {}
    for f in sorted(OUT.glob(f"{slug}/*_calib{calib}_w{window}.json")):
        r = json.loads(f.read_text())
        # keep the primary fp32/CPU protocol authoritative in aggregates; GPU,
        # precision, and quantization variants live in their own tags (the
        # parity tables read those directly)
        if (r.get("device", "cpu") != "cpu"
                or r.get("dtype", "float32") != "float32"
                or r.get("quant", "none") != "none"):
            continue
        out[r["seed"]] = r
    return out


def aggregate():
    """Mean +- seed spread per model; paired base-vs-instruct deltas per pair."""
    if not OUT.exists():
        print("no artifacts yet")
        return
    slugs = sorted({f.parent.name for f in OUT.glob("*/*.json")})
    agg = {}
    for slug in slugs:
        runs = _load_runs(slug).values()
        if not runs:
            continue
        runs = sorted(runs, key=lambda r: r["seed"])
        seeds = [r["seed"] for r in runs]
        entry = {"model": runs[0]["model"], "is_instruct": runs[0]["is_instruct"],
                 "seeds": seeds, "n_seeds": len(runs)}
        for key in ("sanity_cos_last_layer", "auc_safety", "auc_compliance", "auc_harm",
                    "auc_evasion", "headroom", "amp_peak", "amp_peak_layer"):
            vals = [r[key] for r in runs]
            entry[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "values": vals}
        entry["safety_ci95_values"] = [r["safety_ci95"] for r in runs]
        entry["per_layer_auc_safety_mean"] = np.mean(
            [r["per_layer_auc"]["safety"] for r in runs], axis=0).tolist()
        entry["amplification_mean"] = np.mean([r["amplification"] for r in runs],
                                              axis=0).tolist()
        agg[slug] = entry

    pairs = {}
    for pair_key, members in MODELS_P1.items():
        ins, base = _load_runs(members[0][1]), _load_runs(members[1][1])
        common = sorted(set(ins) & set(base))
        if not common:
            continue
        deltas: dict[str, list] = {k: [] for k in
                                   ("auc_safety", "auc_compliance", "headroom", "amp_peak")}
        for s in common:
            for k in deltas:
                deltas[k].append(ins[s][k] - base[s][k])
        pairs[pair_key] = {"seeds": common, **{
            k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
            for k, v in deltas.items()}}
        for slug, runs in ((members[0][1], ins), (members[1][1], base)):
            pairs[pair_key][slug] = {k: agg[slug][k] for k in
                                     ("auc_safety", "auc_compliance", "headroom", "amp_peak")}

    out = {"aggregates": agg, "pair_deltas": pairs}
    (OUT / "aggregate.json").write_text(json.dumps(out, indent=1))
    # compact console table
    print(f"{'slug':22s} {'seeds':10s} {'SafetyAUC':>16s} {'ComplAUC':>9s} "
          f"{'Headroom':>9s} {'amp_pk':>7s} {'sanity':>7s}")
    for slug, e in agg.items():
        print(f"{slug:22s} {str(e['seeds']):10s} "
              f"{e['auc_safety']['mean']:6.3f}±{e['auc_safety']['std']:5.3f} "
              f"{e['auc_compliance']['mean']:9.3f} {e['headroom']['mean']:+9.3f} "
              f"{e['amp_peak']['mean']:7.2f} {e['sanity_cos_last_layer']['mean']:7.4f}")
    print("\npair deltas (instruct - base, paired by seed):")
    for pk, d in pairs.items():
        print(f"  {pk}: dSafetyAUC={d['auc_safety']['mean']:+.3f}±{d['auc_safety']['std']:.3f} "
              f"dCompl={d['auc_compliance']['mean']:+.3f} "
              f"dHeadroom={d['headroom']['mean']:+.3f} (n={len(d['seeds'])} seeds)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="smol135",
                    help="comma list from: " + ",".join(MODELS_P1) + " | all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--calib", default="wikitext",
                    choices=["wikitext", "generic", "safety"])
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--cotangents", type=int, default=8)
    ap.add_argument("--k-store", type=int, default=200)
    ap.add_argument("--k-metric", type=int, default=100)
    ap.add_argument("--members", default="both", choices=["both", "instruct", "base"])
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--quant", default="none", choices=["none", "nf4", "int8"],
                    help="bitsandbytes quantization of the weights (device load)")
    ap.add_argument("--no-chat-template", action="store_true",
                    help="evaluate instruct checkpoints WITHOUT the chat "
                         "template (template-confound ablation, M1)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        return

    which = list(MODELS_P1) if args.pairs == "all" else args.pairs.split(",")
    for key in which:
        for name, slug, is_instr in MODELS_P1[key]:
            if args.members == "instruct" and not is_instr:
                continue
            if args.members == "base" and is_instr:
                continue
            for seed in args.seeds:
                run_one(name, slug, is_instr, seed, args.calib, args.n_calib,
                        args.window, args.cotangents, args.k_store, args.k_metric,
                        args.force, device=args.device, dtype_name=args.dtype,
                        quant=args.quant,
                        use_template=not args.no_chat_template)


if __name__ == "__main__":
    main()

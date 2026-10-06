"""R7: causal steering interventions along the per-layer safety directions.

JADR's #1 future-work item (research/06 R7): the amplification profile (Fig. 4)
is correlational — this script tests it causally. The per-layer safety
direction v_l (diff-in-means harm-vs-benign, the SAME directions the bridge
metric transports) is ADDED to the residual stream at chosen layers,

    h_l <- h_l + alpha * v_l        (forward hook, unit-norm v_l)

and two responses are measured over the eval sets:
  - SafetyAUC (the J-space lens readout) vs alpha — does steering the latent
    direction move the J-space reading where the profile says it should?
  - behavioral refusal rate on a fixed danger subset vs alpha — does it move
    BEHAVIOR (Arditi: adding the refusal direction induces refusal)?

Layer selection comes from the P1 artifacts themselves: the peak-amplification
layer (J-space-accessible) and the layer with WORST per-layer safety AUC
(latent-only candidate), so accessible-vs-latent steering is compared directly.

Artifacts: artifacts/steering/<slug>__<tag>.json ->
  alpha-sweep curves per steered layer (SafetyAUC + refusal rate),
  plus a single-layer profile at fixed alpha.

Run:  .venv/bin/python -m src.run_steering --pairs qwen15 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ensure_eval_sets
from .jlens import fit_lens, lens_logits_at
from .lexicons import LEXICONS_FULL, build_token_hits
from .metrics import auc_ranks
from .run_dpo import REFUSAL_PAT
from .run_p1 import MODELS_P1
from .run_smoke import prep_ids

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "steering"
AXES = list(LEXICONS_FULL)
SAFETY_AI = AXES.index("safety")
K = 100


def directions_and_profile(slug: str):
    """Load the P1 seed-0 artifact: diff-in-means directions are recomputed
    here (they depend on the forward pass), but the amplification profile and
    per-layer AUC used for LAYER SELECTION come from the P1 record."""
    p = ROOT / "artifacts" / "p1" / slug / "seed0_calibwikitext_w0.json"
    r = json.loads(p.read_text())
    return r["amplification"], r["per_layer_auc"]["safety"]


def compute_directions(model, tok, is_instruct, device):
    """Per-layer harm-vs-benign diff-in-means directions (unit norm)."""
    ev = ensure_eval_sets()
    hid, labels = [], []
    with torch.no_grad():
        for label, key in (("harmful", "danger"), ("benign", "benign")):
            for _src, p in ev[key]:
                ids = prep_ids(tok, p, is_instruct)
                if ids.shape[0] < 2:
                    continue
                out = model(ids.to(device)[None], output_hidden_states=True)
                hid.append(torch.stack([h[0, -1] for h in out.hidden_states]).cpu())
                labels.append(1 if label == "harmful" else 0)
    H = torch.stack(hid).to(device)
    labels = torch.tensor(labels, device=device)
    dirs = []
    for l in range(H.shape[1]):
        v = H[labels == 1, l].mean(0) - H[labels == 0, l].mean(0)
        dirs.append((v / v.norm().clamp_min(1e-12)).cpu())
    return dirs


class Steerer:
    """Adds alpha * v_l at layer l (decoder-layer outputs only; embedding layer
    0 stays untouched — v_0 is the static embedding geometry)."""

    def __init__(self, model, layers_vecs: dict[int, torch.Tensor]):
        self.model = model
        self.layers_vecs = layers_vecs
        self.handles = []
        inner = model.model
        for l, v in layers_vecs.items():
            def hook(module, inp, out, v=v):
                if isinstance(out, tuple):
                    h = out[0]
                    return (h + v.to(h.dtype).to(h.device), *out[1:])
                return out + v.to(out.dtype).to(out.device)
            self.handles.append(inner.layers[l].register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def safety_auc(model, lenses, tok_hits, device, danger_prompts, benign_prompts,
               k=K):
    """SafetyAUC of the lens readout over the given danger/benign prompts."""
    unembed = model.get_output_embeddings().weight
    Ts = []
    labels = []
    for label, prompts in (("harmful", danger_prompts), ("benign", benign_prompts)):
        for p in prompts:
            ids = prep_ids(tok_of_current[0], p, tok_of_current[1])
            if ids.shape[0] < 2:
                continue
            out = model(ids.to(device)[None], output_hidden_states=True)
            pos = ids.shape[0] - 1
            T = 0.0
            for l, J in enumerate(lenses):
                h = out.hidden_states[l][0, pos]
                logits = unembed @ (J @ h)
                hits = tok_hits[logits.topk(k).indices, SAFETY_AI].float()
                T += (hits * device_disc).sum().item()
            Ts.append(T)
            labels.append(1 if label == "harmful" else 0)
    Ts = torch.tensor(Ts)
    labels = torch.tensor(labels)
    return auc_ranks(Ts[labels == 1].tolist(), Ts[labels == 0].tolist())


tok_of_current = [None, False]     # (tokenizer, is_instruct) set by run_one
device_disc = None


@torch.no_grad()
def refusal_rate(model, tok, prompts, device, max_new=48):
    hits = 0
    for p in prompts:
        ids = prep_ids(tok, p, False).to(device)
        out = model.generate(ids[None], max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[0]:], skip_special_tokens=True)
        hits += bool(REFUSAL_PAT.search(text))
    return hits / len(prompts)


def run_one(pair_key: str, member_idx: int, device: str, n_calib: int,
            alphas: list[float], profile_alpha: float, n_behav: int,
            n_danger_auc: int):
    """Steering experiment for one model."""
    global device_disc
    name, slug, is_instruct = MODELS_P1[pair_key][member_idx]
    tag = f"seed0_calibwikitext_w0"
    out_json = OUT / f"{slug}__{tag}.json"
    if out_json.exists() and "alpha_sweep" in json.loads(out_json.read_text()):
        print(f"[{slug}] exists, skipping", flush=True)
        return json.loads(out_json.read_text())

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32).to(device).eval()
    tok_of_current[0] = tok
    tok_of_current[1] = is_instruct
    device_disc = torch.tensor(
        [1.0 / np.log2(r + 1) for r in range(1, K + 1)], device=device)

    ampl, per_layer_auc = directions_and_profile(slug)
    dirs = compute_directions(model, tok, is_instruct, device)

    # lens for the SafetyAUC readout (P1 recipe)
    from .data import build_calibration
    calib = [prep_ids(tok, p, is_instruct) for p in
             build_calibration("wikitext", n_prompts=n_calib)]
    calib = [c for c in calib if c.shape[0] >= 8][:n_calib]
    lenses = fit_lens(model, calib, n_cotangents=8, device=device,
                      show_progress=False)
    tok_hits = build_token_hits(
        tok, vocab_size=model.config.vocab_size).to(device)

    ev = ensure_eval_sets()
    rng = random.Random(0)
    beh_prompts = rng.sample([p for _, p in ev["danger"]], n_behav)
    # AUC on a fixed danger subsample at steering time (full 313 x ~20 configs
    # is affordable on GPU, but the subsample keeps every config comparable)
    danger_auc = rng.sample([p for _, p in ev["danger"]], n_danger_auc)
    benign_auc = [p for _, p in ev["benign"]]

    res = {"model": name, "slug": slug, "is_instruct": is_instruct,
           "alphas": alphas, "profile_alpha": profile_alpha,
           "n_behav": n_behav, "n_danger_auc": n_danger_auc,
           "timestamp": datetime.now(timezone.utc).isoformat()}

    # --- alpha sweeps at the two selected layers --------------------------
    # clamp to a valid decoder layer: Steerer hooks inner.layers[l], so the
    # hidden-state index must stay <= num_hidden_layers - 1
    amp_peak_layer = min(int(np.argmax(ampl[1:])) + 1,
                         model.config.num_hidden_layers - 1)
    latent_layer = min(int(np.argmin(per_layer_auc[1:])) + 1,
                       model.config.num_hidden_layers - 1)
    res["layers"] = {"amp_peak": amp_peak_layer, "latent_only": latent_layer}
    sweeps = {}
    for label, layer in (("amp_peak", amp_peak_layer), ("latent_only", latent_layer)):
        curves = {"auc": [], "refusal": []}
        for a in [0.0] + alphas:
            st = Steerer(model, {layer: dirs[layer] * a})
            auc = safety_auc(model, lenses, tok_hits, device,
                             danger_auc, benign_auc)
            rr = refusal_rate(model, tok, beh_prompts, device)
            st.remove()
            curves["auc"].append(auc)
            curves["refusal"].append(rr)
            print(f"  [{slug}/{label}] layer {layer} alpha {a:+.2f}: "
                  f"AUC={auc:.3f} refusal={rr:.2f}", flush=True)
        sweeps[label] = curves
    res["alpha_sweep"] = sweeps

    # --- fixed-alpha layer profile (causal version of Fig. 4) -------------
    profile = []
    for l in range(1, model.config.num_hidden_layers):
        st = Steerer(model, {l: dirs[l] * profile_alpha})
        auc = safety_auc(model, lenses, tok_hits, device,
                         danger_auc, benign_auc)
        st.remove()
        profile.append(auc)
    res["layer_profile_auc"] = profile
    res["layer_profile_alpha"] = profile_alpha
    # pairing: profile[i] steers decoder layer i+1, whose OUTPUT is
    # hidden-state i+2 — so the matched amplification slice is ampl[2:],
    # not ampl[1:] (ampl is hidden-state-indexed; decoder l -> hidden l+1).
    # This mismatch (27 vs 28) crashed the first R7 run at np.corrcoef.
    res["corr_profile_vs_amplification"] = float(np.corrcoef(
        profile, ampl[2:2 + len(profile)])[0, 1])

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1))
    print(f"[{slug}] DONE", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="qwen15", help="comma list | all")
    ap.add_argument("--members", default="both", choices=["both", "instruct", "base"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0])
    ap.add_argument("--profile-alpha", type=float, default=2.0)
    ap.add_argument("--n-behav", type=int, default=30)
    ap.add_argument("--n-danger-auc", type=int, default=100)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    which = list(MODELS_P1) if args.pairs == "all" else args.pairs.split(",")
    for key in which:
        for mi in range(2):
            if args.members == "instruct" and mi == 1:
                continue
            if args.members == "base" and mi == 0:
                continue
            run_one(key, mi, args.device, args.n_calib, args.alphas,
                    args.profile_alpha, args.n_behav, args.n_danger_auc)


if __name__ == "__main__":
    main()

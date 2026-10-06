"""Lexicon-perturbation stability of SafetyAUC (W2 instrument dependence).

Re-scores the STORED top-200 lens tokens (no model re-runs) under random
half-subsets of the safety lexicon: for each of K resamples, keep a random
half of the 170 safety stems, rebuild the boolean token-hit column, recompute
T_saf exactly as run_p1 does (boolean hits x 1/log2(r+1) discount, summed
over layers, top-100), and re-derive the Mann-Whitney SafetyAUC. The spread
of AUCs across resamples bounds how much the paper's headline metric depends
on particular stem choices.

Run: .venv/bin/python -m src.run_lexicon_stability
Out: artifacts/p1/lexicon_stability.json
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .data import ensure_eval_sets
from .lexicons import SAFETY, build_token_hits

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "artifacts" / "p1"
SEED_TAG = "seed0_calibwikitext_w0"
K_METRIC = 100
K_RESAMPLE = 200

MODELS = {
    "smol135m_instruct": "HuggingFaceTB/SmolLM2-135M",
    "smol135m_base": "HuggingFaceTB/SmolLM2-135M",
    "smol360m_instruct": "HuggingFaceTB/SmolLM2-360M",
    "smol360m_base": "HuggingFaceTB/SmolLM2-360M",
    "qwen05b_instruct": "Qwen/Qwen2.5-0.5B",
    "qwen05b_base": "Qwen/Qwen2.5-0.5B",
    "qwen15b_instruct": "Qwen/Qwen2.5-1.5B",
    "qwen15b_base": "Qwen/Qwen2.5-1.5B",
}


def auc_vec(harmful: np.ndarray, benign: np.ndarray) -> float:
    """Mann-Whitney AUC, ties count half (matches src.metrics.auc)."""
    b = np.sort(benign)
    less = np.searchsorted(b, harmful, side="left")
    eq = np.searchsorted(b, harmful, side="right") - less
    wins = less.sum() + 0.5 * eq.sum()
    return float(wins / (len(harmful) * len(benign)))


def main() -> None:
    ev = ensure_eval_sets()
    n_danger = len(ev["danger"])
    disc = torch.tensor([1.0 / math.log2(r + 1) for r in range(1, K_METRIC + 1)])
    rng = np.random.RandomState(0)

    out = {"meta": {
        "k_resamples": K_RESAMPLE, "half_size": len(SAFETY) // 2,
        "n_safety_stems": len(SAFETY), "seed": 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }, "models": {}, "pairs": {}}

    cache_tok: dict[str, tuple] = {}
    per_slug_auc: dict[str, np.ndarray] = {}
    slug_data: dict[str, dict] = {}

    for slug, tok_name in MODELS.items():
        npz_path = P1 / slug / f"{SEED_TAG}.npz"
        if not npz_path.exists():
            print(f"[{slug}] npz missing, skipped")
            continue
        z = np.load(npz_path)
        top = torch.from_numpy(z["top_tokens"])[:, :, :K_METRIC]
        if slug not in cache_tok:
            tok = AutoTokenizer.from_pretrained(tok_name)
            vocab = AutoConfig.from_pretrained(tok_name).vocab_size
            hits_full = build_token_hits(tok, vocab_size=vocab)
            ids, row = np.unique(top.numpy(), return_inverse=True)
            row = row.reshape(top.shape)
            norm = [(t or "").replace("\u0120", " ").replace("\u2581", " ").lower().strip()
                    for t in tok.convert_ids_to_tokens(ids.tolist())]
            match = np.zeros((len(ids), len(SAFETY)), dtype=bool)
            for wi, w in enumerate(SAFETY):
                match[:, wi] = [w in t for t in norm]
            cache_tok[slug] = (hits_full[:, 0], row, match)
        saf_col, row, match = cache_tok[slug]

        P, L, k = top.shape
        full = saf_col[top.numpy()][:, :, :k].float()
        T_full = torch.einsum("plk,k->p", full, disc).numpy()
        labels = np.zeros(P, dtype=int)
        labels[:n_danger] = 1
        ref = auc_vec(T_full[labels == 1], T_full[labels == 0])

        rep_json = P1 / slug / f"{SEED_TAG}.json"
        reported = json.loads(rep_json.read_text()).get("auc_safety") if rep_json.exists() else None
        slug_data[slug] = dict(ref=ref, reported=reported, labels=labels,
                               row=row, full=full)
        print(f"[{slug}] ref={ref:.4f} reported={reported}")

    # paired resampling: the SAME half-lexicon on both members of a pair
    pair_defs = [("smol135m_instruct", "smol135m_base"),
                 ("smol360m_instruct", "smol360m_base"),
                 ("qwen05b_instruct", "qwen05b_base"),
                 ("qwen15b_instruct", "qwen15b_base")]
    # compute per-model resample AUCs (same subset sequence for all models)
    tok_of = {sl: MODELS[sl] for sl in slug_data}
    for slug in slug_data:
        d = slug_data[slug]
        saf_col, row, match = cache_tok[slug]
        aucs = np.empty(K_RESAMPLE)
        for ri in range(K_RESAMPLE):
            rng2 = np.random.RandomState(1000 + ri)   # same subsets across models
            sel = rng2.choice(len(SAFETY), len(SAFETY) // 2, replace=False)
            sub = match[:, sel].any(1)[row]
            T = torch.einsum("plk,k->p", torch.from_numpy(sub.astype(np.float32)), disc).numpy()
            lab = d["labels"]
            aucs[ri] = auc_vec(T[lab == 1], T[lab == 0])
        per_slug_auc[slug] = aucs
        m = slug_data[slug]
        out["models"][slug] = {
            "ref_auc": round(m["ref"], 4), "reported_auc": m["reported"],
            "resample_mean": round(float(aucs.mean()), 4),
            "resample_sd": round(float(aucs.std(ddof=1)), 4),
            "resample_min": round(float(aucs.min()), 4),
            "resample_max": round(float(aucs.max()), 4),
            "max_abs_shift": round(float(np.abs(aucs - m["ref"]).max()), 4),
        }
        print(f"[{slug}] mean={aucs.mean():.3f} sd={aucs.std(ddof=1):.4f} "
              f"max_shift={np.abs(aucs - m['ref']).max():.4f}")

    for a, b in pair_defs:
        if a not in per_slug_auc or b not in per_slug_auc:
            continue
        da, db = slug_data[a], slug_data[b]
        gap = da["ref"] - db["ref"]
        tuned_wins = int((per_slug_auc[a] > per_slug_auc[b]).sum())
        out["pairs"][f"{a}|{b}"] = {
            "ref_gap": round(gap, 4),
            "tuned_gt_base_resamples": tuned_wins,
            "k": K_RESAMPLE,
        }
        print(f"[pair {a} vs {b}] ref_gap={gap:+.3f} ordering preserved "
              f"{tuned_wins}/{K_RESAMPLE}")

    (P1 / "lexicon_stability.json").write_text(json.dumps(out, indent=1))
    print("wrote", P1 / "lexicon_stability.json")


if __name__ == "__main__":
    main()

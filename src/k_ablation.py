"""Recompute SafetyAUC at multiple readout depths k from stored top-k tokens (H5c).

The P1 runs store the top-200 lens tokens per prompt/layer, so the k ablation is
a pure re-analysis: rebuild DCG counters at each k via the token-hit table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .lexicons import LEXICONS_FULL, build_token_hits
from .metrics import auc_ranks

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "artifacts" / "p1"


def k_curve(slug: str, seed: int = 0, calib: str = "wikitext", window: int = 0,
            ks=(25, 50, 100, 150, 200), discount: bool = True):
    f = P1 / slug / f"seed{seed}_calib{calib}_w{window}.npz"
    j = P1 / slug / f"seed{seed}_calib{calib}_w{window}.json"
    if not f.exists() or not j.exists():
        return [], []
    z = np.load(f)
    top = torch.from_numpy(z["top_tokens"])            # [P, L, k_store]
    labels = torch.from_numpy(z["labels"]).bool()
    r = json.loads(j.read_text())
    tok = AutoTokenizer.from_pretrained(r["model"])
    from transformers import AutoConfig
    vconf = AutoConfig.from_pretrained(r["model"])
    # pad to the model's logit width (Qwen: len(tok) < config.vocab_size)
    tok_hits = build_token_hits(tok, vocab_size=vconf.vocab_size)   # [V, A]
    ai = list(LEXICONS_FULL).index("safety")
    kmax = top.shape[-1]
    ks = [k for k in ks if k <= kmax]
    aucs = []
    for k in ks:
        if discount:
            w = torch.tensor([1.0 / math.log2(rr + 1) for rr in range(1, k + 1)])
        else:
            w = torch.ones(k)                          # flat counting (R18)
        hits = tok_hits[top[:, :, :k], ai].float()     # [P, L, k]
        T = torch.einsum("plk,k->pl", hits, w).sum(1)
        aucs.append(auc_ranks(T[labels].tolist(), T[~labels].tolist()))
    return ks, aucs


def dcg_vs_flat(slugs=None):
    """R18: SafetyAUC at k=100 under DCG vs flat counting, all models.
    Rank correlation across models quantifies the rank-weighting sensitivity."""
    slugs = slugs or sorted({f.parent.name for f in P1.glob("*/seed0_calibwikitext_w0.json")})
    d, fl = [], []
    for slug in slugs:
        a_d = k_curve(slug, ks=(100,), discount=True)[1]
        a_f = k_curve(slug, ks=(100,), discount=False)[1]
        if a_d and a_f:
            d.append(a_d[0]); fl.append(a_f[0])
            print(f"{slug:32s} DCG={a_d[0]:.3f} flat={a_f[0]:.3f}")
    from .metrics import auc_ranks
    r = auc_ranks(d, fl)
    print(f"rank-AUC(DCG vs flat as two rankings... use Pearson): "
          f"Pearson r = {float(np.corrcoef(d, fl)[0, 1]):.3f} over {len(d)} models")
    return d, fl


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--flat-sensitivity":
        dcg_vs_flat()
    else:
        slug = sys.argv[1] if len(sys.argv) > 1 else "smol135m_instruct"
        ks, aucs = k_curve(slug)
        print(slug, dict(zip(ks, [round(a, 3) for a in aucs])))

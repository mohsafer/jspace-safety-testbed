"""R14 / W6 toy validation: the amplification metric A_l on a 2-layer linear
transformer with a planted safety direction, where everything is computable
in closed form (research/09 W6 items 1 + 3).

Model (linear, no attention — the lens math is position-independent here):
    h1 = W1 h0,  h2 = W2 h1,  logits = U h2
    W2 = I + s * v v^T   (planted amplifier along the unit direction v)

Closed forms:
    J_1 = W2,  J_2 = I                      (layer-to-output Jacobians)
    A_1(v) = ||U J_1 v|| / med_m ||U J_1 u_m||   (u_m random unit)
    E||U J v||^2 = ||U J||_F^2 / d          (v uniform on the sphere; item 1)
    ||J_1 v|| = ||(I + s v v^T) v|| = 1 + s — the planted direction's gain
    through the remaining map is EXACTLY 1+s, so A_1(v) > 1 iff the late-layer
    map carries v toward output space: "A > 1 => carried to output" holds by
    construction (item 3).

The estimator under test is the PRODUCTION Hutchinson estimator
(jlens.fit_lens) applied verbatim to the toy model — validating the estimator,
not a re-implementation.

Artifacts: artifacts/r14_toy.json
Run:  .venv/bin/python -m src.r14_toy
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .jlens import fit_lens

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "r14_toy.json"


class FixedShim(torch.nn.Module):
    """HF-like wrapper around the linear toy: an embedding-style calibration
    corpus (ids -> hidden states, exactly like a real tokenizer embedding) so
    fit_lens's position accounting holds; hidden_states contract of
    transformers 5.x."""

    def __init__(self, W1, W2, n_vocab=512, d=64):
        super().__init__()
        self.W1 = torch.nn.Parameter(W1, requires_grad=False)
        self.W2 = torch.nn.Parameter(W2, requires_grad=False)
        self.embed = torch.nn.Parameter(torch.randn(n_vocab, d) / math.sqrt(d),
                                        requires_grad=True)
        self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=d)

    def forward(self, ids, output_hidden_states=False):
        # every hidden_states entry must be the exact autograd node the next
        # layer consumes (mirrors how HF chains hidden_states tensors)
        h_in = self.embed[ids[0]][None]
        h1_in = (self.W1 @ h_in[0].T).T[None]
        h2_in = (self.W2 @ h1_in[0].T).T[None]
        return SimpleNamespace(hidden_states=(h_in, h1_in, h2_in))

    def get_output_embeddings(self):
        return None


def main():
    torch.manual_seed(0)
    d, V = 64, 1000
    n_pos = 256
    n_cot = 8

    W1 = torch.randn(d, d) / math.sqrt(d)
    U = torch.randn(V, d) / math.sqrt(d)
    v = torch.randn(d)
    v = v / v.norm()

    results = {"d": d, "V": V, "n_cot": n_cot, "n_pos": n_pos, "sweep": []}

    # ---- W6 item 1: null formula  E||U J v||^2 = ||U J||_F^2 / d ----------
    s_check = 4.0
    W2c = torch.eye(d) + s_check * torch.outer(v, v)
    UJ = U @ W2c
    closed = float((UJ ** 2).sum() / d)
    g = torch.Generator().manual_seed(1)
    mc = []
    for _ in range(4000):
        u = torch.randn(d, generator=g)
        u = u / u.norm()
        mc.append(float((U @ (W2c @ u)).norm()) ** 2)
    results["null_formula"] = {
        "closed_form_E2": closed,
        "montecarlo_E2": float(np.mean(mc)),
        "rel_error": abs(float(np.mean(mc)) - closed) / closed,
    }

    # ---- null median + A_1: estimator vs closed form across planted gains --
    gg = torch.Generator().manual_seed(2)
    nulls_W2c = []
    for _ in range(2000):
        u = torch.randn(d, generator=gg)
        u = u / u.norm()
        nulls_W2c.append(float((U @ (W2c @ u)).norm()))
    med = float(np.median(nulls_W2c))

    for s in (0.0, 1.0, 4.0):
        W2 = torch.eye(d) + s * torch.outer(v, v)
        shim = FixedShim(W1, W2, n_vocab=512, d=d)
        g_cal = torch.Generator().manual_seed(3)
        calib = [torch.randint(0, 512, (8,), generator=g_cal)
                 for _ in range(n_pos // 8)]          # 32 prompts x 8 tokens
        lenses = fit_lens(shim, calib, n_cotangents=n_cot, target_window=0,
                          device="cpu", show_progress=False)
        J1_hat = lenses[1]

        A_closed = float((U @ (W2 @ v)).norm()) / med
        A_hat = float((U @ (J1_hat @ v)).norm()) / med
        results["sweep"].append({
            "s": s,
            "A_closed": A_closed,
            "A_estimated": A_hat,
            "rel_error": abs(A_hat - A_closed) / A_closed,
            "direction_gain_through_late_map": 1.0 + s,   # ||J1 v||, exact
        })

    # ---- "A > 1 => carried to output": output-norm identity check ---------
    s = 4.0
    W2 = torch.eye(d) + s * torch.outer(v, v)
    results["carried_to_output"] = {
        "Uv_norm": float((U @ v).norm()),
        "UJv_norm": float((U @ (W2 @ v)).norm()),
        "ratio": float((U @ (W2 @ v)).norm() / (U @ v).norm()),
        "expected_ratio": 1.0 + s,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()

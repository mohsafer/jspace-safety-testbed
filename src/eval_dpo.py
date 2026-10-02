"""Evaluate DPO-tuned models: P1 readout protocol (seed 0) + behavioral refusal.

Usage: .venv/bin/python -m src.eval_dpo --model smol135 --variant vanilla
Then:  .venv/bin/python -m src.eval_dpo --model smol135 --variant jpen --lam 1.0
Also evaluates the untouched base once (slug <slug>_dpo_base) as the pre-tuning point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import ensure_eval_sets
from .run_dpo import refusal_rate
from .run_p1 import MODELS_P1, OUT as P1_OUT, run_one

ROOT = Path(__file__).resolve().parent.parent


def eval_model(model_path: str, slug: str, seed: int = 0, force: bool = False,
               device: str = "cpu"):
    run_one(model_path, slug, False, seed, "wikitext", 100, 0, 8, 200, 100, force,
            device=device)


def behavioral(model_path: str, tok_name: str, n: int = 100, seed: int = 0,
               device: str = "cpu"):
    import random
    ev = ensure_eval_sets()
    rng = random.Random(seed)
    subset = rng.sample([p for _, p in ev["danger"]], n)
    tok = AutoTokenizer.from_pretrained(tok_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32).to(device).eval()
    return refusal_rate(model, tok, subset)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smol135",
                    choices=["smol135", "smol360", "qwen05", "qwen15"])
    ap.add_argument("--variant", default="base",
                    choices=["base", "vanilla", "jpen"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; 0 = original directory names")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    base_slug = next(m[1] for m in MODELS_P1[args.model] if not m[2])
    base_name = next(m[0] for m in MODELS_P1[args.model] if not m[2])
    tail = "" if args.seed == 0 else f"_seed{args.seed}"
    if args.variant == "base":
        path, slug = base_name, f"{base_slug}_dpo_base{tail}"
    else:
        tag = f"{args.variant}_lam{args.lam}{tail}"
        path = str(ROOT / "artifacts" / "dpo" / f"{base_slug}_{tag}" / "merged")
        slug = f"{base_slug}_dpo_{tag}"

    eval_model(path, slug, seed=args.seed, force=args.force, device=args.device)
    rr = behavioral(path, path, seed=args.seed, device=args.device)
    print(f"{slug}: behavioral refusal rate = {rr:.3f}")

    # attach refusal rate to the P1 json
    f = P1_OUT / slug / f"seed{args.seed}_calibwikitext_w0.json"
    r = json.loads(f.read_text())
    r["refusal_rate"] = rr
    f.write_text(json.dumps(r, indent=1))
    print("updated", f)

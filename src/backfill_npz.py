"""Backfill ONLY the npz sidecars (top_tokens/counters/labels) for models whose
seed0 JSON record exists but whose gitignored npz was lost at migration.

NEVER writes the JSON record — the JSON with its completion marker is the
metric record (AGENTS.md); this script only regenerates the regenerable
sidecar so CPU re-analyses (lexicon stability, qualitative figures) can run.
Faithfulness check: the recomputed counters' SafetyAUC is printed next to the
committed JSON's reported value; they must agree to ~1e-3.

Run: .venv/bin/python -m src.backfill_npz [--slugs a,b,c]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build_calibration, ensure_eval_sets
from .jlens import fit_lens, lens_logits_at
from .lexicons import LEXICONS_FULL, build_token_hits
from .run_p1 import OUT
from .run_smoke import prep_ids

MISSING = [
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "smol360m_instruct", True),
    ("HuggingFaceTB/SmolLM2-360M", "smol360m_base", False),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen05b_instruct", True),
    ("Qwen/Qwen2.5-0.5B", "qwen05b_base", False),
]
SEED, TAG, N_CALIB, N_COT, K_STORE, K_METRIC = 0, "seed0_calibwikitext_w0", 100, 8, 200, 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slugs", default="")
    args = ap.parse_args()
    want = set(args.slugs.split(",")) if args.slugs else None

    ev = ensure_eval_sets()
    disc = torch.tensor([1.0 / math.log2(r + 1) for r in range(1, K_METRIC + 1)])

    for model_name, slug, is_instruct in MISSING:
        if want and slug not in want:
            continue
        out_dir = OUT / slug
        npz_path = out_dir / f"{TAG}.npz"
        json_path = out_dir / f"{TAG}.json"
        if npz_path.exists():
            print(f"[{slug}] npz exists, skipping")
            continue
        assert json_path.exists(), f"{slug}: no JSON record to validate against"
        reported = json.loads(json_path.read_text())["auc_safety"]

        t0 = time.time()
        torch.manual_seed(SEED)
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32).to("cpu").eval()
        print(f"[{slug}] loaded ({time.time()-t0:.0f}s)", flush=True)

        calib_ids = [prep_ids(tok, p, is_instruct) for p in
                     build_calibration("wikitext", n_prompts=N_CALIB)]
        calib_ids = [x for x in calib_ids if x.shape[0] >= 8][:N_CALIB]
        lenses = fit_lens(model, calib_ids, n_cotangents=N_COT,
                          target_window=0, device="cpu", show_progress=False)
        print(f"[{slug}] lens fit {len(calib_ids)}x{N_COT} "
              f"({time.time()-t0:.0f}s)", flush=True)

        tok_hits = build_token_hits(tok, vocab_size=model.config.vocab_size)
        top_list, labels = [], []
        for label, key in (("harmful", "danger"), ("benign", "benign")):
            for _src, p in ev[key]:
                ids = prep_ids(tok, p, is_instruct)
                if ids.shape[0] < 2:
                    continue
                with torch.no_grad():
                    out = model(ids[None], output_hidden_states=True)
                    lls = lens_logits_at(model, lenses, out.hidden_states,
                                         position=ids.shape[0] - 1)
                top_list.append(torch.stack(
                    [lg.topk(K_STORE).indices for lg in lls]))
                labels.append(1 if label == "harmful" else 0)
        top_tokens = torch.stack(top_list)
        labels_t = torch.tensor(labels)
        hits = tok_hits[top_tokens[:, :, :K_METRIC]].float()
        counters = torch.einsum("plka,k->pla", hits, disc)

        ai = list(LEXICONS_FULL).index("safety")
        T = counters[:, :, ai].sum(1).numpy()
        h, b = T[labels_t == 1], T[labels_t == 0]
        bs = np.sort(b)
        less = np.searchsorted(bs, h, side="left")
        eq = np.searchsorted(bs, h, side="right") - less
        auc = float((less.sum() + 0.5 * eq.sum()) / (len(h) * len(b)))

        np.savez_compressed(
            npz_path,
            top_tokens=top_tokens.numpy().astype(np.int32),
            counters=counters.numpy().astype(np.float32),
            labels=labels_t.numpy(),
        )
        print(f"[{slug}] npz written; SafetyAUC recomputed {auc:.4f} vs "
              f"committed {reported:.4f} (drift {abs(auc-reported):.4f})",
              flush=True)
        del model


if __name__ == "__main__":
    main()

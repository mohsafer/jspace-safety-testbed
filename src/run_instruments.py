"""R11 / W2: instrument triangulation for the J-space safety metrics.

The review's W2: all metrics rest on self-built lexicons + substring matching;
no independent instrument correlates. Four triangulation instruments, one
pass per model (runbook Phase 3):

  1. latent linear probe (the supervised anchor): logistic regression
     (sklearn, C=1.0) on h_l per layer, 50/50 cross-fit over the eval sets
     (train on half, test on the other half, swapped) -> probe-AUC per layer
     + layer-max. The probe-vs-lens gap is a SUPERVISORED version of the
     accessibility gap.
  2. split-half lexicon reliability: SafetyAUC with lexicon halves A/B
     (seeded random split of each axis list); split-half correlation across
     models (psychometric-style reliability of the instrument).
  3. random-lexicon control: size-matched random word lists (seeded);
     SafetyAUC should sit at ~0.5 — rules out generic token-frequency
     artifacts driving the counter.
  4. logit-lens baseline: U·h_l readout instead of U·J_l·h_l, same DCG
     counters and eval sets -> logit-lens SafetyAUC. Tests whether the
     causal J-lens adds anything over the free logit-lens (the exact
     features-vs-tokens critique, arXiv:2609.01936).

Readout position/protocol identical to run_p1: last prompt token, top-100
DCG counters, StrongREJECT vs XSTest-safe, lens fitted on wikitext-100.

Artifacts: artifacts/instruments/<slug>__<tag>.json (per model+seed) and
artifacts/instruments/instruments_aggregate.json -> triangulation table.

Run:  .venv/bin/python -m src.run_instruments --pairs all --seeds 0 --device cuda
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import build_calibration, ensure_eval_sets
from .jlens import fit_lens
from .lexicons import LEXICONS_FULL, build_token_hits
from .metrics import auc_ranks, headroom
from .run_p1 import MODELS_P1
from .run_smoke import prep_ids

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "instruments"
AXES = list(LEXICONS_FULL)
K = 100


# ------------------------------------------------------------------ helpers
def _load_model(name: str, device: str):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32).to(device).eval()
    return model, tok


# ------------------------------------------------------- instrument pieces
def split_half_tables(tok, model_vocab: int, seed: int = 0):
    """Lexicon halves A/B per axis (seeded) + a size-matched random-lexicon
    control. Returns token-hit tables [V, A] for full/halfA/halfB/random."""
    vocab = tok.get_vocab()
    rng = random.Random(seed)
    tables = {"full": torch.zeros(model_vocab, len(AXES), dtype=torch.bool),
              "halfA": torch.zeros(model_vocab, len(AXES), dtype=torch.bool),
              "halfB": torch.zeros(model_vocab, len(AXES), dtype=torch.bool),
              "random": torch.zeros(model_vocab, len(AXES), dtype=torch.bool)}
    for ai, axis in enumerate(AXES):
        words = LEXICONS_FULL[axis]
        hits = [w for w in words if w in vocab]
        missing = len(words) - len(hits)
        shuffled = hits[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        for w in shuffled[:half]:
            tables["halfA"][vocab[w], ai] = True
        for w in shuffled[half:]:
            tables["halfB"][vocab[w], ai] = True
        for w in hits:
            tables["full"][vocab[w], ai] = True
        # random control: same number of tokens, drawn from the model vocab
        ids = rng.sample(range(model_vocab), len(hits))
        for i in ids:
            tables["random"][i, ai] = True
    return tables, missing


@torch.no_grad()
def probe_auc_per_layer(H: torch.Tensor, labels: torch.Tensor):
    """Cross-fit logistic probe on h_l per layer (50/50 halves, swapped)."""
    from sklearn.linear_model import LogisticRegression
    idx = np.arange(H.shape[0])
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    per_layer = []
    for l in range(H.shape[1]):
        X = H[:, l].numpy()
        y = labels.numpy()
        preds = np.zeros_like(y, dtype=float)
        for tr, te in ((a, b), (b, a)):
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(X[tr], y[tr])
            preds[te] = clf.predict_proba(X[te])[:, 1]
        per_layer.append(auc_ranks(preds[y == 1].tolist(), preds[y == 0].tolist()))
    return per_layer


@torch.no_grad()
def logit_lens_counters(model, tok, is_instruct, eval_sets, device):
    """Instrument 4: same DCG counters from the FREE readout U·h_l (no lens)."""
    tok_hits, _ = split_half_tables(tok, model.config.vocab_size)
    th = tok_hits["full"].to(device)
    top_list, labels = [], []
    for label, key in (("harmful", "danger"), ("benign", "benign")):
        for _src, p in eval_sets[key]:
            ids = prep_ids(tok, p, is_instruct)
            if ids.shape[0] < 2:
                continue
            out = model(ids.to(device)[None], output_hidden_states=True)
            pos = ids.shape[0] - 1
            unembed = model.get_output_embeddings().weight
            lls = [unembed @ out.hidden_states[l][0, pos].float()
                   for l in range(len(out.hidden_states))]
            top_list.append(torch.stack([lg.topk(K).indices for lg in lls]).cpu())
            labels.append(1 if label == "harmful" else 0)
    top = torch.stack(top_list)
    disc = torch.tensor([1.0 / math.log2(r + 1) for r in range(1, K + 1)],
                        device=th.device)
    return torch.einsum("plka,k->pla", th[top[:, :, :K]].float(), disc), torch.tensor(labels)


# ------------------------------------------------------------------ driver
def run_one(pair_key: str, member_idx: int, seed: int, n_calib: int,
            device: str, force: bool):
    name, slug, is_instruct = MODELS_P1[pair_key][member_idx]
    tag = f"seed{seed}_calibwikitext_w0"
    out_json = OUT / f"{slug}__{tag}.json"
    if out_json.exists() and "probe_auc_per_layer" in json.loads(out_json.read_text()) and not force:
        print(f"[{slug}] exists, skipping", flush=True)
        return json.loads(out_json.read_text())

    model, tok = _load_model(name, device)

    # lens: same fitting recipe as run_p1 (wikitext, 100 prompts, 8 cotangents)
    calib = [prep_ids(tok, p, is_instruct) for p in
             build_calibration("wikitext", n_prompts=n_calib)]
    calib = [c for c in calib if c.shape[0] >= 8][:n_calib]
    t0 = time.time()
    lenses = fit_lens(model, calib, n_cotangents=8, device=device,
                      show_progress=False)
    print(f"[{slug}] lens fit {time.time()-t0:.0f}s", flush=True)

    tables, missing_words = split_half_tables(tok, model.config.vocab_size, seed)
    th_full = tables["full"].to(device)
    eval_sets = ensure_eval_sets()

    # instruments 1-3 share one forward sweep over the eval sets; the four
    # counter tables differ only in the token-hit table applied to the same
    # stored top-K lists
    top_list, hid_list, labels = [], [], []
    for label, key in (("harmful", "danger"), ("benign", "benign")):
        for _src, p in eval_sets[key]:
            ids = prep_ids(tok, p, is_instruct)
            if ids.shape[0] < 2:
                continue
            with torch.no_grad():
                out = model(ids.to(device)[None], output_hidden_states=True)
                hid_list.append(torch.stack(
                    [h[0, -1] for h in out.hidden_states]).cpu())
                lls = []
                for l, J in enumerate(lenses):
                    h = out.hidden_states[l][0, ids.shape[0] - 1]
                    lls.append(model.get_output_embeddings().weight @ (J @ h))
                top_list.append(torch.stack(
                    [lg.topk(K).indices for lg in lls]).cpu())
            labels.append(1 if label == "harmful" else 0)
    top = torch.stack(top_list)[:, :, :K]
    H = torch.stack(hid_list)
    labels_t = torch.tensor(labels)
    disc = torch.tensor([1.0 / math.log2(r + 1) for r in range(1, K + 1)])

    def counters_of(table):
        th = table.to(device)
        return torch.einsum("plka,k->pla", th[top].float(),
                            disc.to(th.device)).cpu()

    c_full = counters_of(tables["full"])
    c_halfA = counters_of(tables["halfA"])
    c_halfB = counters_of(tables["halfB"])
    c_rand = counters_of(tables["random"])

    ai = AXES.index("safety")
    T_full = c_full.sum(1)

    res = {
        "model": name, "slug": slug, "is_instruct": is_instruct, "seed": seed,
        "n_calib": len(calib), "k": K, "device": device,
        "lexicon_tokens_missing": missing_words,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "safety_auc_lens": auc_ranks(T_full[labels_t == 1, ai].tolist(),
                                     T_full[labels_t == 0, ai].tolist()),
        "probe_auc_per_layer": probe_auc_per_layer(H, labels_t),
        "safety_auc_random_lexicon": auc_ranks(
            c_rand.sum(1)[labels_t == 1, ai].tolist(),
            c_rand.sum(1)[labels_t == 0, ai].tolist()),
    }
    res["probe_auc_max"] = float(np.max(res["probe_auc_per_layer"]))
    # split-half reliability: SafetyAUC with half A vs half B (same prompts),
    # and their per-model agreement via the correlation across layers
    ta, tb = c_halfA.sum(1), c_halfB.sum(1)
    res["safety_auc_halfA"] = auc_ranks(ta[labels_t == 1, ai].tolist(),
                                        ta[labels_t == 0, ai].tolist())
    res["safety_auc_halfB"] = auc_ranks(tb[labels_t == 1, ai].tolist(),
                                        tb[labels_t == 0, ai].tolist())
    res["split_half_layer_r"] = float(np.corrcoef(
        [auc_ranks(c_halfA[labels_t == 1, l, ai].tolist(),
                   c_halfA[labels_t == 0, l, ai].tolist())
         for l in range(c_halfA.shape[1])],
        [auc_ranks(c_halfB[labels_t == 1, l, ai].tolist(),
                   c_halfB[labels_t == 0, l, ai].tolist())
         for l in range(c_halfB.shape[1])])[0, 1])

    # instrument 4: free logit-lens readout (separate sweep; no lens involved)
    c_logit, labels_logit = logit_lens_counters(model, tok, is_instruct,
                                                eval_sets, device)
    tl = c_logit.sum(1)
    res["safety_auc_logitlens"] = auc_ranks(tl[labels_logit == 1, ai].tolist(),
                                            tl[labels_logit == 0, ai].tolist())

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1))
    print(f"[{slug}] DONE lensAUC={res['safety_auc_lens']:.3f} "
          f"probeMax={res['probe_auc_max']:.3f} logitLens={res['safety_auc_logitlens']:.3f} "
          f"random={res['safety_auc_random_lexicon']:.3f} "
          f"splitHalfR={res['split_half_layer_r']:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def aggregate():
    rows = {}
    for f in sorted(OUT.glob("*__seed*.json")):
        r = json.loads(f.read_text())
        rows[r["slug"]] = r
    out = {}
    for slug, r in rows.items():
        out[slug] = {
            "safety_auc_lens": r["safety_auc_lens"],
            "probe_auc_max": r["probe_auc_max"],
            "probe_auc_argmax_layer": int(np.argmax(r["probe_auc_per_layer"])),
            "safety_auc_logitlens": r["safety_auc_logitlens"],
            "safety_auc_random_lexicon": r["safety_auc_random_lexicon"],
            "safety_auc_halfA": r["safety_auc_halfA"],
            "safety_auc_halfB": r["safety_auc_halfB"],
            "split_half_layer_r": r["split_half_layer_r"],
        }
    # cross-model split-half reliability (psychometric r across models)
    if len(rows) >= 3:
        a = [rows[s]["safety_auc_halfA"] for s in rows]
        b = [rows[s]["safety_auc_halfB"] for s in rows]
        out["_cross_model_split_half_r"] = float(np.corrcoef(a, b)[0, 1])
    (OUT / "instruments_aggregate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="all", help="comma list | all")
    ap.add_argument("--members", default="both",
                    choices=["both", "instruct", "base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-calib", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
        return
    which = list(MODELS_P1) if args.pairs == "all" else args.pairs.split(",")
    for key in which:
        for mi in range(2):
            if args.members == "instruct" and mi == 1:
                continue
            if args.members == "base" and mi == 0:
                continue
            for seed in args.seeds:
                run_one(key, mi, seed, args.n_calib, args.device, args.force)
    aggregate()


if __name__ == "__main__":
    main()

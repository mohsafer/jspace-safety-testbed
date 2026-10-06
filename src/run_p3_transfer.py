"""R17: attack-transferability matrix for the P3 suffixes (review response).

Cross-applies every stored attacked suffix to every other target (and to the
never-tuned base models), measuring behavioral success and both monitor
readings. Answers the standard GCG-reviewer question "do these suffixes
transfer?" nearly for free from stored artifacts.

For each (suffix source target, arm) x (eval target):
  - load the eval target + its frozen lens (cached by run_p3_gcg)
  - for each stored attacked prompt of that (source, arm):
      rebuild ids = eval-target prefix + stored suffix
        (the suffix tokens are target-agnostic; the PROMPT text is shared)
      -> T_saf (prompt side), behavioral check (greedy gen), refusal
  - aggregate: success rate, refusal rate, alarm rate, T_saf mean per cell

Run:  .venv/bin/python -m src.run_p3_transfer --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lexicons import build_token_hits
from .run_dpo import REFUSAL_PAT
from .run_p3_gcg import TARGETS, _disc, behavioral_check, monitor_and_latent, calibrate_harm_directions
from .run_smoke import prep_ids

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "p3"


def slug_to_target(slug: str):
    for key, (name, is_instr) in TARGETS.items():
        got = name[4:] if name.startswith("dpo:") else name.split("/")[-1]
        if got == slug:
            return name, is_instr
    raise KeyError(slug)


def load_target(slug: str, device: str, k_metric: int):
    """Model + lens + tok_hits + harm ref for one eval target."""
    target, is_instruct = slug_to_target(slug)
    if target.startswith("dpo:"):
        name = str(ROOT / "artifacts" / "dpo" / target[4:] / "merged")
    else:
        name = target
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32).to(device).eval()
    lenses = [J.to(device) for J in torch.load(
        OUT / f"lens_{slug}.pt", weights_only=True)]
    tok_hits = build_token_hits(
        tok, vocab_size=model.config.vocab_size).to(device)
    disc = _disc(k_metric, torch.device(device))
    harm_dirs, harm_stats = calibrate_harm_directions(
        model, tok, is_instruct, device)
    cal = json.loads((OUT / slug / "monitor_calibration.json").read_text())
    return dict(model=model, tok=tok, is_instruct=is_instruct, lenses=lenses,
                tok_hits=tok_hits, disc=disc, harm_dirs=harm_dirs,
                harm_stats=harm_stats, tau95=cal["tau95"])


def eval_suffix(ctx, prompt: str, suffix_ids: list[int], affirm: str,
                k_metric: int, device: str) -> dict:
    """Apply one stored suffix to this eval target."""
    tok, is_instruct = ctx["tok"], ctx["is_instruct"]
    if is_instruct and getattr(tok, "chat_template", None):
        pre = tok.apply_chat_template(
            [{"role": "user", "content": prompt + " "}],
            tokenize=False, add_generation_prompt=True)
        ids_p = tok(pre, return_tensors="pt",
                    add_special_tokens=False).input_ids[0][:-1]
    else:
        ids_p = tok(prompt + " ", return_tensors="pt").input_ids[0]
    ids = torch.cat([ids_p, torch.tensor(suffix_ids)]).to(device)
    T, _ = monitor_and_latent(ctx["model"], ctx["lenses"], ctx["tok_hits"],
                              ctx["disc"], ctx["harm_dirs"], ctx["harm_stats"],
                              ids, k_metric)
    affirm_ids = tok(affirm, add_special_tokens=False,
                     return_tensors="pt").input_ids[0]
    beh = behavioral_check(ctx["model"], tok, ids_p.to(device),
                           torch.tensor(suffix_ids).to(device),
                           affirm_ids.to(device), device)
    return {"T_saf": T, "alarm": bool(T > ctx["tau95"]), **beh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k-metric", type=int, default=100)
    ap.add_argument("--eval-targets", default="all",
                    help="comma list of eval-target slugs; default = every "
                    "source slug. 'fp32' = the three 24GB-fittable targets "
                    "(gemma-2-9b-it would need ~37 GB fp32 and is excluded).")
    args = ap.parse_args()
    disc = _disc(args.k_metric, torch.device(args.device))

    sources = {}   # (slug, arm) -> list of {prompt, suffix_ids, affirm}
    for tdir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        for arm in ("vanilla", "silent"):
            rows = []
            for pj in sorted((tdir / arm).glob("prompt*.json")):
                r = json.loads(pj.read_text())
                if "suffix_ids" in r:
                    rows.append({"prompt": r["prompt"],
                                 "suffix_ids": r["suffix_ids"],
                                 "affirm": r.get("affirm", "Sure, here is")})
            if rows:
                sources[(tdir.name, arm)] = rows
    slugs = sorted({s for s, _ in sources})
    if args.eval_targets == "fp32":
        slugs = [s for s in slugs if "gemma" not in s]
    elif args.eval_targets != "all":
        want = set(args.eval_targets.split(","))
        slugs = [s for s in slugs if s in want]
    print(f"sources: {sorted(sources)}; eval targets: {slugs}")

    # source tokenizers: cross-family suffix ids must be decoded to text and
    # re-encoded with the eval tokenizer (raw ids exceed other vocabularies)
    src_toks = {}
    results = {}
    for eval_slug in slugs:
        ctx = load_target(eval_slug, args.device, args.k_metric)
        for (src_slug, arm), rows in sorted(sources.items()):
            if src_slug not in src_toks:
                tname, _ = slug_to_target(src_slug)
                tname = (str(ROOT / "artifacts" / "dpo" / tname[4:] / "merged")
                         if tname.startswith("dpo:") else tname)
                src_toks[src_slug] = AutoTokenizer.from_pretrained(tname)
            src_tok = src_toks[src_slug]
            eval_vocab = ctx["model"].config.vocab_size
            # same tokenizer family -> raw ids are exact (decode/re-encode is
            # NOT behavior-preserving: it dropped self-transfer 0.47 -> 0.20)
            same_tok = (src_tok.name_or_path == ctx["tok"].name_or_path
                        and src_tok.vocab_size == ctx["tok"].vocab_size)
            cell = {"n": 0, "success": 0, "refusal": 0, "alarm": 0, "T_saf": []}
            for r in rows:
                sfx = r["suffix_ids"]
                if max(sfx) >= eval_vocab or not same_tok:
                    text = src_tok.decode(sfx)
                    sfx = ctx["tok"](text, add_special_tokens=False
                                     ).input_ids
                out = eval_suffix(ctx, r["prompt"], sfx,
                                  r["affirm"], args.k_metric, args.device)
                cell["n"] += 1
                cell["success"] += int(out["affirm_success"])
                cell["refusal"] += int(out["refusal"])
                cell["alarm"] += int(out["alarm"])
                cell["T_saf"].append(out["T_saf"])
            n = cell.pop("n")
            rec = {k: (round(v / n, 3) if k != "T_saf"
                       else round(sum(v) / len(v), 2)) for k, v in cell.items()}
            results[f"{src_slug}|{arm}|->|{eval_slug}"] = rec
            print(f"{src_slug[:26]:26s}/{arm:7s} -> {eval_slug[:26]:26s} "
                  f"succ={rec['success']:.2f} ref={rec['refusal']:.2f} "
                  f"alarm={rec['alarm']:.2f} T={rec['T_saf']}", flush=True)
        del ctx["model"]
        torch.cuda.empty_cache()

    out = OUT / "transfer_matrix.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"wrote {out} ({len(results)} cells)")


if __name__ == "__main__":
    main()

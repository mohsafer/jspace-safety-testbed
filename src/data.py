"""P1/P2 datasets + calibration corpora. Cached as JSON under data/.

Eval sets (JADR roles, arXiv:2607.12792):
  danger: StrongREJECT (313 prompts) — the danger set
  benign: XSTest-safe (250 prompts, label=="safe" in Paul/XSTest v2 mirror)
Deviation from JADR: their 70 OKTest lexically-alarming-but-safe prompts are not
publicly retrievable; the safe control is XSTest-only. Documented in the paper.

Calibration corpora for lens fitting (H5 corpus ablation):
  wikitext: 120 x ~110-word chunks of wikitext-103 (pretrain-like; default)
  generic:  the 32 hand-written smoke sentences
  safety:   a mixed sample of the eval prompts themselves (worst-case in-domain)
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STRONGREJECT = "AlignmentResearch/StrongREJECT"   # split="validation", col content
XSTEST = "Paul/XSTest"                            # split="train", cols prompt/label
WIKITEXT = "Salesforce/wikitext"                  # config wikitext-103-raw-v1
HH_HF = "Anthropic/hh-rlhf"                       # harmless-base, P2 tuning


def _cache(name: str):
    DATA.mkdir(exist_ok=True)
    return DATA / f"{name}.json"


def _dump(name: str, obj):
    _cache(name).write_text(json.dumps(obj, ensure_ascii=False, indent=1))


def _load_json(name: str):
    return json.loads(_cache(name).read_text())


def _parse_sr_content(raw) -> str:
    """StrongREJECT 'content' rows arrive as stringified one-element lists
    (['...']) or actual lists depending on the loader — normalize to str."""
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    t = str(raw).strip()
    if t.startswith("[") and t.endswith("]"):
        t = t[1:-1].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "'\"":
        t = t[1:-1]
    return t.strip()


def ensure_eval_sets(force: bool = False) -> dict:
    """Returns {"danger": [[source, prompt], ...], "benign": [...]}."""
    f = _cache("eval_sets")
    if f.exists() and not force:
        return _load_json("eval_sets")
    from datasets import load_dataset

    sr = load_dataset(STRONGREJECT, split="validation")
    danger = [[
        "strongreject",
        _parse_sr_content(r["content"]),
    ] for r in sr if _parse_sr_content(r["content"])]

    xs = load_dataset(XSTEST, split="train")
    benign = [["xstest", r["prompt"]] for r in xs if r["label"] == "safe" and r["prompt"]]

    out = {"danger": danger, "benign": benign}
    _dump("eval_sets", out)
    return out


def build_calibration(kind: str, n_prompts: int = 120, seed: int = 0) -> list[str]:
    """Calibration texts for lens fitting (~110 words each for wikitext)."""
    f = _cache(f"calib_{kind}")
    if f.exists():
        texts = _load_json(f"calib_{kind}")
    elif kind == "wikitext":
        from datasets import load_dataset
        ds = load_dataset(WIKITEXT, "wikitext-103-raw-v1", split="train", streaming=True)
        words, texts = [], []
        for row in ds:
            line = row["text"].strip()
            if not line or line.startswith("="):        # skip headers/empty
                continue
            words.extend(line.split())
            while len(words) >= 110:
                texts.append(" ".join(words[:110]))
                words = words[110:]
            if len(texts) >= n_prompts + 40:
                break
        _dump(f"calib_{kind}", texts[: n_prompts + 40])
    elif kind == "generic":
        from .prompts import CALIBRATION_PROMPTS
        texts = list(CALIBRATION_PROMPTS)
    elif kind == "safety":
        ev = ensure_eval_sets()
        pool = [p for _, p in ev["danger"]] + [p for _, p in ev["benign"]]
        rng = random.Random(seed)
        rng.shuffle(pool)
        texts = pool
    else:
        raise ValueError(kind)

    rng = random.Random(seed)
    if len(texts) > n_prompts:
        texts = texts[:n_prompts]
    return texts


def ensure_tuning_pairs(n_pairs: int = 3000, seed: int = 0, force: bool = False) -> list[dict]:
    """P2 tuning data: Anthropic HH pairs (prompt, chosen, rejected).

    NOTE: hh-rlhf on the Hub now exposes only the 'default' config (combined
    160,800-row train split); the per-subset configs are gone. We filter to
    single-turn rows whose chosen response looks refusal-like, which recovers
    the harmless-subset signal. Deviation noted in the paper.
    """
    f = _cache("tuning_hh")
    if f.exists() and not force:
        return _load_json("tuning_hh")
    from datasets import load_dataset

    ds = load_dataset(HH_HF, split="train")
    refusal_pat = ("sorry", "cannot", "can't", "can not", "won't", "will not",
                   "not able", "unable", "refuse", "inappropriate", "illegal",
                   "unethical", "dangerous", "against the law", "harmful")
    rows = []
    for r in ds:
        ch, rj = r["chosen"], r["rejected"]
        if "Human:" not in ch or "Assistant:" not in ch:
            continue
        human = ch.split("Assistant:")[0].replace("Human:", "", 1).strip()
        a_ch = ch.split("Assistant:", 1)[1].strip()
        a_rj = rj.split("Assistant:", 1)[1].strip() if "Assistant:" in rj else None
        if not human or not a_ch or not a_rj:
            continue
        if "\nHuman:" in a_ch:            # multi-turn — keep single-turn only
            continue
        low = a_ch.lower()[:120]          # refusal marker near the response opening
        if not any(p in low for p in refusal_pat):
            continue
        rows.append({"prompt": human, "chosen": a_ch, "rejected": a_rj})
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_pairs]
    _dump("tuning_hh", rows)
    return rows


if __name__ == "__main__":
    ev = ensure_eval_sets()
    print(f"danger={len(ev['danger'])} benign={len(ev['benign'])}")
    for kind in ("wikitext", "generic", "safety"):
        c = build_calibration(kind)
        print(f"calib_{kind}: n={len(c)} | ex: {c[0][:70]!r}")

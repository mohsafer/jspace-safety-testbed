"""Generate paper tables + number macros from artifacts (single source of truth).

Emits:
  paper/tables/main_results.tex     per-model metrics (mean±std over seeds, CI)
  paper/tables/pair_deltas.tex      paired instruct-base deltas per family
  paper/tables/ablations.tex        corpus / window ablation numbers
  paper/tables/datasets.tex         eval data summary
  paper/tables/dpo_multiseed.tex    P2 pre/post per variant, seeds aggregated
  paper/tables/p3_attacks.tex       P3/H4 GCG evasion per target x arm
  paper/tables/precision_quant.tex  fp32-CPU vs fp32/fp16/NF4-GPU parity + Qwen3-4B
  paper/numbers.tex                 \newcommand macros used in the prose

Run: .venv/bin/python -m src.make_tables
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "artifacts" / "p1"
TAB = ROOT / "paper" / "tables"
NUM = ROOT / "paper" / "numbers.tex"

FAMILIES = [("smol135", "SmolLM2-135M"), ("smol360", "SmolLM2-360M"),
            ("qwen05", "Qwen2.5-0.5B"), ("qwen15", "Qwen2.5-1.5B")]
SLUGS = {
    "smol135": ("smol135m_instruct", "smol135m_base"),
    "smol360": ("smol360m_instruct", "smol360m_base"),
    "qwen05": ("qwen05b_instruct", "qwen05b_base"),
    "qwen15": ("qwen15b_instruct", "qwen15b_base"),
}


def f3(x):
    return f"{x:.3f}"


def wilson(p: float, n: int, z: float = 1.96):
    """Wilson 95% CI for a binomial proportion — used for the R13 judge
    columns (n = 15/50 cells; guide review M1: rates must ship with CIs)."""
    if n == 0:
        return (float('nan'), float('nan'))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_runs(slug, calib="wikitext", window=0):
    """Primary-protocol runs only: fp32 on CPU (the paper's declared protocol).
    GPU / fp16 / quantized runs live under their own tags and belong to the
    parity & quantization tables, never the main ones."""
    out = {}
    for f in sorted(P1.glob(f"{slug}/*_calib{calib}_w{window}.json")):
        r = json.loads(f.read_text())
        if r.get("device", "cpu") != "cpu":
            continue
        if r.get("dtype", "float32") != "float32" or r.get("quant", "none") != "none":
            continue
        out[r["seed"]] = r
    return out


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    macros = {}

    # ---- main results table ----
    rows = []
    agg_all = {}
    for pk, pretty in FAMILIES:
        for member, tag in (("instruct", SLUGS[pk][0]), ("base", SLUGS[pk][1])):
            runs = load_runs(tag)
            if not runs:
                continue
            rs = list(runs.values())
            seeds = sorted(runs)
            agg_all[tag] = rs
            ci = np.array([r["safety_ci95"] for r in rs])
            row = (f"{pretty} {'\\;tuned' if member == 'instruct' else '\\;base'} & "
                   f"{len(seeds)} & "
                   f"{f3(np.mean([r['auc_safety'] for r in rs]))}"
                   f"\\,{{\\tiny$\\pm${np.std([r['auc_safety'] for r in rs]):.3f}}} & "
                   f"[{f3(ci[:, 0].mean())}, {f3(ci[:, 1].mean())}] & "
                   f"{f3(np.mean([r['auc_compliance'] for r in rs]))} & "
                   f"{f3(np.mean([r['auc_harm'] for r in rs]))} & "
                   f"{np.mean([r['headroom'] for r in rs]):+.3f} & "
                   f"{np.mean([r['amp_peak'] for r in rs]):.2f}"
                   f"\\,(L{int(np.mean([r['amp_peak_layer'] for r in rs]))}) \\\\")
            rows.append(row)
    main = (
        "\\begin{tabular}{lcccccccc}\n\\toprule\n"
        "Model & seeds & SafetyAUC & 95\\% CI & ComplAUC & HarmAUC & Headroom & "
        "$A_{\\max}$ \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "main_results.tex").write_text(main)

    # ---- pair deltas ----
    drows = []
    for pk, pretty in FAMILIES:
        si, sb = SLUGS[pk]
        ri, rb = load_runs(si), load_runs(sb)
        common = sorted(set(ri) & set(rb))
        if not common:
            continue
        d_s = [ri[s]["auc_safety"] - rb[s]["auc_safety"] for s in common]
        d_c = [ri[s]["auc_compliance"] - rb[s]["auc_compliance"] for s in common]
        d_h = [ri[s]["headroom"] - rb[s]["headroom"] for s in common]
        d_a = [ri[s]["amp_peak"] - rb[s]["amp_peak"] for s in common]
        drows.append(f"{pretty} & {len(common)} & "
                     f"{np.mean(d_s):+.3f}$\\pm${np.std(d_s):.3f} & "
                     f"{np.mean(d_c):+.3f} & {np.mean(d_h):+.3f} & "
                     f"{np.mean(d_a):+.2f} \\\\")
    deltas = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
              "Family & $n$ & $\\Delta$SafetyAUC & $\\Delta$ComplAUC & "
              "$\\Delta$Headroom & $\\Delta A_{\\max}$ \\\\\n\\midrule\n"
              + "\n".join(drows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "pair_deltas.tex").write_text(deltas)

    # ---- ablations (corpus, window; 135M pair) ----
    arows = []
    for tag, label in ((SLUGS["smol135"][0], "135M instr."), (SLUGS["smol135"][1], "135M base")):
        for calib in ("wikitext", "generic", "safety"):
            runs = load_runs(tag, calib=calib)
            if not runs:
                continue
            rs = list(runs.values())
            arows.append(f"{label} & {calib} & $W{{=}}0$ & {len(rs)} & "
                         f"{f3(np.mean([r['auc_safety'] for r in rs]))}"
                         f"$\\pm${np.std([r['auc_safety'] for r in rs]):.3f} \\\\")
        for w in (0, 1):
            runs = load_runs(tag, calib="wikitext", window=w)
            if not runs or w == 0:
                continue
            rs = list(runs.values())
            arows.append(f"{label} & wikitext & $W{{=}}1$ & {len(rs)} & "
                         f"{f3(np.mean([r['auc_safety'] for r in rs]))}"
                         f"$\\pm${np.std([r['auc_safety'] for r in rs]):.3f} \\\\")
    abl = ("\\begin{tabular}{llccc}\n\\toprule\n"
           "Model & corpus & window & seeds & SafetyAUC \\\\\n\\midrule\n"
           + "\n".join(arows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "ablations.tex").write_text(abl)

    # ---- datasets ----
    ev = json.loads((ROOT / "data" / "eval_sets.json").read_text())
    n_tune = len(json.loads((ROOT / "data" / "tuning_hh.json").read_text())) \
        if (ROOT / "data" / "tuning_hh.json").exists() else 0
    macros["nTuningPairs"] = str(n_tune)
    ds = ("\\begin{tabular}{llll}\n\\toprule\n"
          "Role & Dataset & $n$ & Use \\\\\n\\midrule\n"
          f"Danger & StrongREJECT & {len(ev['danger'])} & danger set \\\\\n"
          f"Safe & XSTest (safe half) & {len(ev['benign'])} & benign control \\\\\n"
          "Calibration & wikitext-103 chunks & 100 & lens fitting \\\\\n"
          f"Tuning & HH (refusal-filtered) & {n_tune} & DPO pairs \\\\\n"
          "\\bottomrule\n\\end{tabular}\n")
    (TAB / "datasets.tex").write_text(ds)

    # ---- lexicon sizes (generated, never hand-typed) ----
    import sys
    sys.path.insert(0, str(ROOT))
    from src.lexicons import LEXICONS_FULL
    jadr_spec = {"safety": 170, "compliance": 47, "evasion": 49, "softening": 16,
                 "hedging": 22, "harm": 84}
    ls = ["\\begin{tabular}{lrr}\n\\toprule\nAxis & Ours (words) & JADR spec \\\\\n\\midrule\n"]
    for axis in ("safety", "compliance", "evasion", "softening", "hedging", "harm"):
        ls.append(f"{axis} & {len(LEXICONS_FULL[axis])} & {jadr_spec[axis]} \\\\\n")
    ls.append("\\bottomrule\n\\end{tabular}\n")
    (TAB / "lexicon_sizes.tex").write_text("".join(ls))

    # ---- prose macros ----
    def m(name, val):
        macros[name] = val

    # letter-coded families (LaTeX macros cannot contain digits)
    # A=SmolLM2-135M, B=SmolLM2-360M, C=Qwen2.5-0.5B, D=Qwen2.5-1.5B
    fam_of = {"smol135": "A", "smol360": "B", "qwen05": "C", "qwen15": "D"}
    # pre-seed every planned macro so LaTeX compiles while runs are in flight
    for pk, _ in FAMILIES:
        for member in ("instruct", "base"):
            key = f"{'instr' if member == 'instruct' else 'base'}{fam_of[pk]}"
            for suf, v in (("safety", "--"), ("safetystd", "--"), ("compl", "--"),
                           ("headroom", "--"), ("amp", "--"), ("ampL", "--"),
                           ("sanity", "--"), ("corr", "--")):
                m(f"{key}{suf}", v)

    for tag, rs in agg_all.items():
        pk = next(p for p, sl in (("smol135", SLUGS["smol135"]), ("smol360", SLUGS["smol360"]),
                                  ("qwen05", SLUGS["qwen05"]), ("qwen15", SLUGS["qwen15"]))
                  if tag in sl)
        fam = fam_of[pk]
        key = f"{'instr' if 'instruct' in tag else 'base'}{fam}"
        m(key + "safety", f3(np.mean([r["auc_safety"] for r in rs])))
        m(key + "safetystd", f"{np.std([r['auc_safety'] for r in rs]):.3f}")
        m(key + "compl", f3(np.mean([r["auc_compliance"] for r in rs])))
        m(key + "headroom", f"{np.mean([r['headroom'] for r in rs]):+.3f}")
        m(key + "amp", f"{np.mean([r['amp_peak'] for r in rs]):.2f}")
        m(key + "ampL", str(int(np.mean([r["amp_peak_layer"] for r in rs]))))
        m(key + "sanity", f"{min(r['sanity_cos_last_layer'] for r in rs):.4f}")

    # ---- v2.14: sanity-check floor across ALL runs, honest split by protocol ----
    _sp, _sd = [], []
    for _f in sorted(glob.glob(str(P1 / "*" / "*.json"))):
        if "dpo" in _f or "judge" in _f or "stability" in _f:
            continue
        try:
            _d = json.loads(open(_f).read())
        except Exception:
            continue
        if not isinstance(_d, dict) or "sanity_cos_last_layer" not in _d:
            continue
        _tag = os.path.basename(_f)
        (_sd if any(t in _tag for t in ("bfloat16", "qnf4", "qint8", "float16"))
         else _sp).append(_d["sanity_cos_last_layer"])
    if _sp:
        macros["sanityPrimaryMin"] = f"{min(_sp):.5f}"
        macros["sanityPrimaryN"] = str(len(_sp))
    if _sd:
        macros["sanityDepMin"] = f"{min(_sd):.4f}"
        macros["sanityDepN"] = str(len(_sd))
        m(key + "corr", f"{np.mean([r['corr(amp,per-layer-AUC)'] for r in rs]):+.2f}")
    # ---- DPO macros (pre/post, per variant per model; missing runs stay "--") ----
    for mslug, mm in (("smol135m", "a"), ("smol360m", "b")):
        for variant, vm in (("base", "Pre"), ("vanilla_lam1.0", "Van"),
                            ("jpen_lam1.0", "Jpen")):
            f = P1 / f"{mslug}_base_dpo_{variant}" / "seed0_calibwikitext_w0.json"
            if f.exists():
                r = json.loads(f.read_text())
                macros[f"dpo{mm}{vm}Safety"] = f3(r["auc_safety"])
                macros[f"dpo{mm}{vm}Compl"] = f3(r["auc_compliance"])
                macros[f"dpo{mm}{vm}Headroom"] = f"{r['headroom']:+.3f}"
                macros[f"dpo{mm}{vm}Amp"] = f"{r['amp_peak']:.2f}"
                macros[f"dpo{mm}{vm}Refusal"] = f"{r.get('refusal_rate', float('nan')):.2f}"
            else:
                for suf in ("Safety", "Compl", "Headroom", "Amp", "Refusal"):
                    macros[f"dpo{mm}{vm}{suf}"] = "--"
    # deltas of interest
    for mslug, mm in (("smol135m", "a"), ("smol360m", "b")):
        try:
            pre = json.loads((P1 / f"{mslug}_base_dpo_base" /
                              "seed0_calibwikitext_w0.json").read_text())
            van = json.loads((P1 / f"{mslug}_base_dpo_vanilla_lam1.0" /
                              "seed0_calibwikitext_w0.json").read_text())
            jpen = json.loads((P1 / f"{mslug}_base_dpo_jpen_lam1.0" /
                               "seed0_calibwikitext_w0.json").read_text())
            macros[f"dpo{mm}VanGain"] = f"{van['auc_safety'] - pre['auc_safety']:+.3f}"
            macros[f"dpo{mm}JpenGain"] = f"{jpen['auc_safety'] - pre['auc_safety']:+.3f}"
            macros[f"dpo{mm}VanJpenGap"] = f"{van['auc_safety'] - jpen['auc_safety']:+.3f}"
        except FileNotFoundError:
            for suf in ("VanGain", "JpenGain", "VanJpenGap"):
                macros[f"dpo{mm}{suf}"] = "--"

    # ---- P2 multi-seed (incl. qwen at scale): SafetyAUC mean+-std over seeds ----
    dpo_models = [("smol135m", "SmolLM2-135M"), ("smol360m", "SmolLM2-360M"),
                  ("qwen05b", "Qwen2.5-0.5B"), ("qwen15b", "Qwen2.5-1.5B")]
    msrows = []
    for mslug, pretty in dpo_models:
        pre_dir = P1 / f"{mslug}_base_dpo_base"
        cells = [pretty]
        pre_f = pre_dir / "seed0_calibwikitext_w0.json"
        cells.append(f3(json.loads(pre_f.read_text())["auc_safety"])
                     if pre_f.exists() else "--")
        for variant in ("vanilla", "jpen"):
            vals, refs = [], []
            dirs = [(P1 / f"{mslug}_base_dpo_{variant}_lam1.0", 0)]
            dirs += [(P1 / f"{mslug}_base_dpo_{variant}_lam1.0_seed{s}", s)
                     for s in (1, 2)]
            for d, eseed in dirs:
                # dir encodes the training seed; the P1 json tag encodes the
                # EVAL seed, which eval_dpo wires to the training seed
                f = d / f"seed{eseed}_calibwikitext_w0.json"
                if f.exists():
                    r = json.loads(f.read_text())
                    vals.append(r["auc_safety"])
                    if "refusal_rate" in r:
                        refs.append(r["refusal_rate"])
            if vals:
                cells.append(f"{np.mean(vals):.3f}$\\pm${np.std(vals):.3f}"
                             f" ({len(vals)})")
                cells.append(f"{np.mean(refs):.2f}" if refs else "--")
            elif mslug == "qwen15b":
                cells += ["\\pend{R1}", "\\pend{R1}"]
            else:
                cells += ["--", "--"]
        msrows.append(" & ".join(cells) + " \\\\")
    mst = ("\\begin{tabular}{lcccccc}\n\\toprule\n"
           "Model & pre & vanilla SafetyAUC & ref. & J-pen SafetyAUC & ref. \\\\\n"
           "\\midrule\n" + "\n".join(msrows) +
           "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "dpo_multiseed.tex").write_text(mst)

    # ---- P3 / H4 attack table (from p3 summaries; post-hoc adds response side) ----
    P3 = ROOT / "artifacts" / "p3"
    p3_targets = [("smol135m_base_vanilla_lam1.0", "SmolLM2-135M (DPO)"),
                  ("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct"),
                  ("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct"),
                  ("gemma-2-9b-it", "gemma-2-9b-it (bf16)")]
    prow = {}
    for slug, _ in p3_targets:
        for arm in ("vanilla", "silent"):
            # v2.15: the widened n=50 cells are the primary data where they
            # exist (the n=15 cells are a nested subset); 9B stays n=15
            f = P3 / slug / f"{arm}_n50" / "summary.json"
            if not f.exists():
                f = P3 / slug / arm / "summary.json"
            prow[(slug, arm)] = json.loads(f.read_text()) if f.exists() else None
    # R13 judge grades, with Wilson CIs (guide review M1: rates ship with CIs)
    JG = {}
    jg_f = P3 / "judge" / "judge_grades.json"
    if jg_f.exists():
        for k, v in json.loads(jg_f.read_text()).get("p3", {}).items():
            JG[k] = v
    prows = []
    for slug, pretty in p3_targets:
        for arm, alab in (("vanilla", "GCG (compliance)"), ("silent", "silent (compliance $-$ monitor)")):
            s = prow[(slug, arm)]
            jg = JG.get(f"{slug}__{arm}")
            if jg and jg.get("judge_harmful_rate") is not None and jg.get("n"):
                lo, hi = wilson(jg["judge_harmful_rate"], jg["n"])
                jcell = f"{jg['judge_harmful_rate']:.2f} [{lo:.2f},{hi:.2f}]"
            else:
                jcell = "--"
            if s is None:
                prows.append(f"{pretty} & {alab} & -- & -- & -- & -- & -- & -- & {jcell} \\\\")
                continue
            prows.append(
                f"{pretty} & {alab} & {s.get('n_prompts', '--')} & "
                f"{s.get('affirm_success_rate', float('nan')):.2f} & "
                f"{s.get('refusal_rate', float('nan')):.2f} & "
                f"{s.get('alarm_rate', float('nan')):.2f}"
                f" / {s.get('alarm_response_rate', float('nan')):.2f} & "
                f"{s.get('T_saf_mean', float('nan')):.2f}"
                f" / {s.get('T_saf_response_mean', float('nan')):.2f} & "
                f"{s.get('gap_G_mean', float('nan')):+.2f} & {jcell} \\\\")
    pt = ("\\begin{tabular}{llccccccc}\n\\toprule\n"
          "Target & Attack & $n$ & succ. & ref. & alarm (p / r) & "
          "$T_{\\mathrm{saf}}$ (p / r) & $G$ & judge-harm [CI] \\\\\n\\midrule\n"
          + "\n".join(prows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "p3_attacks.tex").write_text(pt)

    # ---- v2.7: R6 defended-model rows appended to the attack table ----
    def_dir = P3 / "merged"
    if def_dir.exists():
        for arm, alab in (("vanilla_defended", "+ R6 defense (GCG compliance)"),
                          ("silent_defended", "+ R6 defense (silent)")):
            f = def_dir / arm / "summary.json"
            if not f.exists():
                continue
            g = json.loads(f.read_text())
            prows.append(
                f"SmolLM2-135M (DPO, R6-defended) & {alab} & "
                f"{g.get('n_prompts', '--')} & "
                f"{g.get('affirm_success_rate', float('nan')):.2f} & "
                f"{g.get('refusal_rate', float('nan')):.2f} & "
                f"{g.get('alarm_rate', float('nan')):.2f}"
                f" / {g.get('alarm_response_rate', float('nan')):.2f} & "
                f"{g.get('T_saf_mean', float('nan')):.2f}"
                f" / {g.get('T_saf_response_mean', float('nan')):.2f} & "
                f"{g.get('gap_G_mean', float('nan')):+.2f} & -- \\\\")
        pt = ("\\begin{tabular}{llccccccc}\n\\toprule\n"
              "Target & Attack & $n$ & succ. & ref. & alarm (p / r) & "
              "$T_{\\mathrm{saf}}$ (p / r) & $G$ & judge-harm [CI] \\\\\n\\midrule\n"
              + "\n".join(prows) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "p3_attacks.tex").write_text(pt)

    # ---- v2.7 prose macros: R6 defense / R7 steering / M2 judge / PEZ ----
    dl = ROOT / "artifacts" / "defense" / ("smol135m_base_vanilla_lam1.0"
                                           "_defended_lam1.0") / "defense_log.json"
    if dl.exists():
        recs = json.loads(dl.read_text())
        r0 = [r["saf_mass"] for r in recs if r["round"] == 0][-10:]
        rl = [r["saf_mass"] for r in recs
              if r["round"] == recs[-1]["round"]][-10:]
        macros["rSixMassFirst"] = f"{sum(r0) / len(r0):.2f}"
        macros["rSixMassLast"] = f"{sum(rl) / len(rl):.2f}"
    cal = def_dir / "monitor_calibration.json"
    if cal.exists():
        macros["rSixDefTau"] = f"{json.loads(cal.read_text())['tau95']:.2f}"
    for arm, am in (("vanilla_defended", "Van"), ("silent_defended", "Sil")):
        f = def_dir / arm / "summary.json"
        if f.exists():
            g = json.loads(f.read_text())
            macros[f"rSixDef{am}Succ"] = f"{g['affirm_success_rate']:.2f}"
            macros[f"rSixDef{am}Alarm"] = f"{g['alarm_rate']:.2f}"
            macros[f"rSixDef{am}T"] = f"{g['T_saf_mean']:.2f}"
            macros[f"rSixDef{am}G"] = f"{g['gap_G_mean']:+.2f}"
    # ---- v2.9: R6 lambda sweep (program 15) -> table + prose macros ----
    sweep = {}
    tag2lam = {"dlam05": 0.5, "dlam20": 2.0, "dlam40": 4.0}
    for f in P3.glob("merged/*_dlam*/summary.json"):
        tag = f.parent.name.split("_", 1)[1]
        if tag not in tag2lam:
            continue
        sweep[(tag2lam[tag], f.parent.name.split("_", 1)[0])] = json.loads(f.read_text())
    for arm, am in (("vanilla_defended", "Van"), ("silent_defended", "Sil")):
        f = def_dir / arm / "summary.json"
        if f.exists():
            sweep[(1.0, arm.split("_")[0])] = json.loads(f.read_text())
    if sweep:
        srows = []
        for lam in sorted({k[0] for k in sweep}):
            for arm, alab in (("vanilla", "GCG (compliance)"),
                              ("silent", "silent (compliance $-$ monitor)")):
                g = sweep.get((lam, arm))
                if g is None:
                    continue
                srows.append(
                    f"$\lambda_{{\mathrm{{def}}}}={lam:g}$ & {alab} & "
                    f"{g.get('affirm_success_rate', float('nan')):.2f} & "
                    f"{g.get('alarm_rate', float('nan')):.2f} & "
                    f"{g.get('T_saf_mean', float('nan')):.2f} & "
                    f"{g.get('gap_G_mean', float('nan')):+.2f} \\\\")
        (TAB / "r6_sweep.tex").write_text(
            "\\begin{tabular}{llcccc}\n\\toprule\n"
            "Setting & Attack & succ. & alarm & $T_{\\mathrm{saf}}$ & $G$ \\\\\n\\midrule\n"
            + "\n".join(srows) + "\n\\bottomrule\n\\end{tabular}\n")
        sil = [g["affirm_success_rate"] for (l, a), g in sweep.items() if a == "silent"]
        van = [g["affirm_success_rate"] for (l, a), g in sweep.items() if a == "vanilla"]
        macros["rSixSweepSilMin"] = f"{min(sil):.2f}"
        macros["rSixSweepSilMax"] = f"{max(sil):.2f}"
        macros["rSixSweepVanMin"] = f"{min(van):.2f}"
        macros["rSixSweepVanMax"] = f"{max(van):.2f}"
        macros["rSixSweepN"] = str(len({k[0] for k in sweep}))
    for slug, suf in (("qwen15b_instruct", "I"), ("qwen15b_base", "B")):
        f = ROOT / "artifacts" / "steering" / f"{slug}__seed0_calibwikitext_w0.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        macros[f"rSevCorr{suf}"] = f"{d['corr_profile_vs_amplification']:+.2f}"
        macros[f"rSevLayer{suf}"] = str(d["layers"]["amp_peak"])
        ap = d["alpha_sweep"]["amp_peak"]
        i_neg = d["alphas"].index(min(d["alphas"]))
        i_pos = d["alphas"].index(max(d["alphas"]))
        macros[f"rSevAucNeg{suf}"] = f"{ap['auc'][i_neg]:.3f}"
        macros[f"rSevAucPos{suf}"] = f"{ap['auc'][i_pos]:.3f}"
    # M2: second (cross-family) judge — same cells, agreement on the claim
    pj = P3 / "judge" / "judge_grades.json"
    gj = P3 / "judge" / "grades_gemma9" / "judge_grades.json"
    if pj.exists() and gj.exists():
        a = json.loads(pj.read_text()).get("p3", {})
        b = json.loads(gj.read_text()).get("p3", {})
        # sign agreement per target: does each judge rank silent above vanilla?
        agree = total = 0
        for tgt in sorted({k.rsplit("__", 1)[0] for k in a if "__" in k}):
            va = {k.rsplit("__", 1)[1]: a[k].get("judge_harmful_rate")
                  for k in a if k.rsplit("__", 1)[0] == tgt}
            vb = {k.rsplit("__", 1)[1]: b[k].get("judge_harmful_rate")
                  for k in b if k.rsplit("__", 1)[0] == tgt}
            if None in (va.get("silent"), va.get("vanilla"),
                        vb.get("silent"), vb.get("vanilla")):
                continue
            total += 1
            if (va["silent"] - va["vanilla"] > 0) == (vb["silent"] - vb["vanilla"] > 0):
                agree += 1
        macros["rJudgeAgree"] = f"{agree}/{total}"
        for k, suf in (("Qwen2.5-0.5B-Instruct__silent", "C"),
                       ("Qwen2.5-1.5B-Instruct__silent", "D")):
            v = b.get(k)
            if v and v.get("judge_harmful_rate") is not None:
                macros[f"judgeG{suf}Harm"] = f"{v['judge_harmful_rate']:.2f}"
                n = v.get("n")
                if n:
                    lo, hi = wilson(v["judge_harmful_rate"], n)
                    macros[f"judgeG{suf}CI"] = f"[{lo:.2f},{hi:.2f}]"
    # PEZ continuous-prefix baseline
    pez_rows = []
    for slug in ("smol135m_base_vanilla_lam1.0", "Qwen2.5-0.5B-Instruct"):
        for arm in ("vanilla", "silent"):
            f = P3 / slug / f"pez_{arm}" / "summary.json"
            if f.exists():
                g = json.loads(f.read_text())
                pez_rows.append((slug, arm, g))
    if pez_rows:
        macros["rPezSucc"] = f"{max(g['affirm_success_rate'] for _, _, g in pez_rows):.2f}"
        for arm, am in (("vanilla", "Van"), ("silent", "Sil")):
            ts = [g["T_saf_mean"] for _, a, g in pez_rows if a == arm]
            if ts:
                macros[f"rPez{am}T"] = f"{max(ts):.2f}"

    # P3 prose macros (per target letter e=135M-dpo, c=0.5B, d=1.5B)
    # letter codes (LaTeX): F = 135M-DPO target, C = 0.5B, D = 1.5B
    for slug, mm in (("smol135m_base_vanilla_lam1.0", "F"),
                     ("Qwen2.5-0.5B-Instruct", "C"), ("Qwen2.5-1.5B-Instruct", "D")):
        for arm, am in (("vanilla", "Van"), ("silent", "Sil")):
            s = prow[(slug, arm)]
            if s:
                macros[f"p{mm}{am}Succ"] = f"{s.get('affirm_success_rate', float('nan')):.2f}"
                macros[f"p{mm}{am}Alarm"] = f"{s.get('alarm_rate', float('nan')):.2f}"
                macros[f"p{mm}{am}AlarmResp"] = f"{s.get('alarm_response_rate', float('nan')):.2f}"
                macros[f"p{mm}{am}T"] = f"{s.get('T_saf_mean', float('nan')):.2f}"
                macros[f"p{mm}{am}TResp"] = f"{s.get('T_saf_response_mean', float('nan')):.2f}"
                macros[f"p{mm}{am}G"] = f"{s.get('gap_G_mean', float('nan')):+.2f}"
            else:
                for suf in ("Succ", "Alarm", "AlarmResp", "T", "TResp", "G"):
                    macros[f"p{mm}{am}{suf}"] = "--"

    # ---- precision / quantization parity table (P1 protocol, matched seed) ----
    def variant_val(tag, dtype, quant, device, seeds):
        vals = []
        for s in seeds:
            extra = []
            if dtype != "float32":
                extra.append(dtype)
            if quant != "none":
                extra.append(f"q{quant}")
            tagf = f"seed{s}_calibwikitext_w0" + ("_" + "_".join(extra) if extra else "")
            f = P1 / tag / f"{tagf}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["auc_safety"])
        return vals

    PAR_SEEDS = (10, 11)
    q34 = [("qwen3_4b_instruct", "Qwen3-4B-Instruct"),
           ("qwen3_4b_base", "Qwen3-4B base")]
    qrows = []
    for pk, pretty in FAMILIES:
        for member, tag in (("tuned", SLUGS[pk][0]), ("base", SLUGS[pk][1])):
            cpu = variant_val(tag, "float32", "none", "cpu",
                              range(0, 8))  # primary protocol seeds
            gpu32 = variant_val(tag, "float32", "none", "cuda", PAR_SEEDS)
            fp16 = variant_val(tag, "float16", "none", "cuda", PAR_SEEDS)
            nf4 = variant_val(tag, "float32", "nf4", "cuda", PAR_SEEDS)
            bf16 = variant_val(tag, "bfloat16", "none", "cuda", (0, 1, 2))

            def cell(v, show_std=False):
                if not v:
                    return "--"
                if show_std and len(v) > 1:
                    return f"{np.mean(v):.3f}$\\pm${np.std(v):.3f}"
                return f"{np.mean(v):.3f}"
            qrows.append(
                f"{pretty} \\;{member} & {cell(cpu, True)} & {cell(gpu32)} & "
                f"{cell(fp16)} & {cell(bf16, True)} & {cell(nf4)} \\\\")
    # R2: the 4B pair now has full bf16 rows (n=3, A30)
    for tag, pretty in q34:
        cpu = variant_val(tag, "float32", "none", "cpu", (0, 1))
        gpu32 = variant_val(tag, "float32", "none", "cuda", PAR_SEEDS)
        bf16 = variant_val(tag, "bfloat16", "none", "cuda", (0, 1, 2))
        nf4 = variant_val(tag, "float32", "nf4", "cuda", (0, 1))
        qrows.append(
            f"{pretty} (fp32 anchor) & {cell(cpu, True)} & {cell(gpu32)} & -- & "
            f"{cell(bf16, True)} & {cell(nf4)} \\\\")
    # R3: gemma-2-9b pair — the GPU IS the primary at 9B (bf16)
    for tag, pretty in (("gemma2_9b_it", "gemma-2-9b-it"),
                        ("gemma2_9b_base", "gemma-2-9b base")):
        bf16 = variant_val(tag, "bfloat16", "none", "cuda", (0, 1, 2))
        nf4 = variant_val(tag, "float32", "nf4", "cuda", (0, 1))
        qrows.append(f"{pretty} (bf16 primary) & -- & -- & -- & "
                     f"{cell(bf16, True)} & {cell(nf4)} \\\\")
    # R5: abliterated pair
    for tag, pretty in (("qwen317_abliterated", "Qwen3-1.7B-abliterated"),
                        ("qwen317_base", "Qwen3-1.7B base")):
        bf16 = variant_val(tag, "bfloat16", "none", "cuda", (0, 1))
        nf4 = variant_val(tag, "float32", "nf4", "cuda", (0, 1))
        qrows.append(f"{pretty} & -- & -- & -- & "
                     f"{cell(bf16, True)} & {cell(nf4)} \\\\")
    qt = ("\\begin{tabular}{lccccc}\n\\toprule\n"
          "Model & fp32/CPU (seeds) & fp32/GPU & fp16/GPU & bf16/GPU & NF4/GPU \\\\\n"
          "\\midrule\n" + "\n".join(qrows) + "\n\\bottomrule\n\\end{tabular}\n")
    (TAB / "precision_quant.tex").write_text(qt)

    # parity prose macros (per family letter; tuned member)
    for pk, _ in FAMILIES:
        tag = SLUGS[pk][0]
        for lab, dt, q, dev, seeds in (("Gpu", "float32", "none", "cuda", PAR_SEEDS),
                                       ("Fp", "float16", "none", "cuda", PAR_SEEDS),
                                       ("Nf", "float32", "nf4", "cuda", PAR_SEEDS)):
            v = variant_val(tag, dt, q, dev, seeds)
            macros[f"par{fam_of[pk]}{lab}"] = f"{np.mean(v):.3f}" if v else "--"

    # ---- v2.2 new tables: monitor ladder, instrument triangulation, transfer ----
    lad_f = P3 / "ladder" / "ladder_table.json"
    if lad_f.exists():
        lad = json.loads(lad_f.read_text())
        per = lad.get("per_target", lad)
        lrows = []
        pretty_map = {"smol135m_base_vanilla_lam1.0": "SmolLM2-135M (DPO)",
                      "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Inst",
                      "Qwen2.5-1.5B-Instruct": "Qwen2.5-1.5B-Inst"}
        for slug in sorted(per):
            tbl = per[slug]
            p = pretty_map.get(slug, slug)
            sets = [("attacked_vanilla", "vanilla att."),
                    ("attacked_silent", "silent att."),
                    ("attacked_silent_m8", "silent$_{m8}$ re-att."),
                    ("attacked_silent_lrn", "silent$_{lrn}$ re-att."),
                    ("clean_danger", "clean danger"),
                    ("benign", "benign (FPR)")]
            for setname, label in sets:
                if setname not in tbl.get("last", {}):
                    continue
                cells = []
                succ = None
                for rung in ("last", "multi8", "genmean", "learned"):
                    ce = tbl.get(rung, {}).get(setname)
                    cells.append("--" if not ce else f"{ce['alarm_rate']:.2f}")
                    if ce and succ is None:
                        succ = ce.get("success_rate")
                sc = f" ({succ:.2f})" if succ is not None else ""
                lrows.append(f"{p} & {label}{sc} & " + " & ".join(cells) + " \\\\")
        lad_tex = ("\\begin{tabular}{llcccc}\n\\toprule\n"
                   "Target & prompt set & last & multi8 & genmean & learned \\\\\n\\midrule\n"
                   + "\n".join(lrows) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "ladder.tex").write_text(lad_tex)

    ins_f = ROOT / "artifacts" / "instruments" / "instruments_aggregate.json"
    if ins_f.exists():
        ins = json.loads(ins_f.read_text())
        irows = []
        for slug, v in sorted(ins.items()):
            if slug.startswith("_"):
                continue
            irows.append(
                f"{slug.replace('_', '\\_')} & {v['safety_auc_lens']:.3f} & {v['probe_auc_max']:.3f} & "
                f"{v['safety_auc_logitlens']:.3f} & {v['safety_auc_random_lexicon']:.3f} & "
                f"{v['split_half_layer_r']:.3f} \\\\")
        cm = ins.get("_cross_model_split_half_r")
        if cm is not None:
            irows.append(f"\\multicolumn{{6}}{{l}}{{cross-model split-half $r = {cm:.3f}$}} \\\\")
        ins_tex = ("\\begin{tabular}{lccccc}\n\\toprule\n"
                   "Model & lens & probe (max) & logit-lens & random-lex & split-half $r$ \\\\\n\\midrule\n"
                   + "\n".join(irows) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "instruments.tex").write_text(ins_tex)

    tr_f = P3 / "transfer_matrix.json"
    if tr_f.exists():
        tr = json.loads(tr_f.read_text())
        # keys are "src|arm|->|eval" (run_p3_transfer); escape underscores for LaTeX
        esc = lambda t: t.replace("_", "\\_")
        evals = sorted({k.split("|->|")[1] for k in tr})
        srcs = sorted({k.split("|->|")[0] for k in tr})
        trows = []
        for src in srcs:
            cells = []
            for e in evals:
                c = tr.get(f"{src}|->|{e}")
                cells.append(f"{c['success']:.2f}/{c['alarm']:.2f}" if c else "--")
            trows.append(f"{esc(src.replace('|', ' / '))} & " + " & ".join(cells) + " \\\\")
        tr_tex = ("\\begin{tabular}{l" + "c" * len(evals) + "}\n\\toprule\n"
                  "source $\\to$ eval & " + " & ".join(esc(e) for e in evals) + " \\\\\n\\midrule\n"
                  + "\n".join(trows) + "\n\\bottomrule\n\\end{tabular}\n")
        (TAB / "transfer.tex").write_text(tr_tex)

    # ---- v2.2 new prose macros ----
    m3 = P3 / "gemma-2-9b-it"
    for arm, am in (("vanilla", "Van"), ("silent", "Sil")):
        f = m3 / arm / "summary.json"
        if f.exists():
            g = json.loads(f.read_text())
            macros[f"pG{am}Succ"] = f"{g['affirm_success_rate']:.2f}"
            macros[f"pG{am}Alarm"] = f"{g['alarm_rate']:.2f}"
            macros[f"pG{am}T"] = f"{g['T_saf_mean']:.2f}"
            macros[f"pG{am}G"] = f"{g['gap_G_mean']:+.2f}"
    pq = P3 / "judge" / "judge_grades.json"
    if pq.exists():
        jg = json.loads(pq.read_text()).get("p3", {})
        for k, suf in (("Qwen2.5-0.5B-Instruct__silent", "C"),
                       ("Qwen2.5-1.5B-Instruct__silent", "D")):
            v = jg.get(k)
            if v and v.get("judge_harmful_rate") is not None and v.get("n"):
                lo, hi = wilson(v["judge_harmful_rate"], v["n"])
                macros[f"judge{suf}Harm"] = f"{v['judge_harmful_rate']:.2f}"
                macros[f"judge{suf}CI"] = f"[{lo:.2f},{hi:.2f}]"
    for slug, mm in (("Qwen3-1.7B-abliterated", "Abl"), ("Qwen3-1.7B base", "AblBase")):
        pass  # filled below from variant_val-like reads
    ins_f2 = ROOT / "artifacts" / "instruments" / "instruments_aggregate.json"
    if ins_f2.exists():
        ins = json.loads(ins_f2.read_text())
        it, ib = ins.get("qwen3_4b_instruct"), ins.get("qwen3_4b_base")
        if it:
            macros["rFourInLens"] = f"{it['safety_auc_lens']:.3f}"
            macros["rFourInLogit"] = f"{it['safety_auc_logitlens']:.3f}"
            macros["rFourInProbe"] = f"{it['probe_auc_max']:.3f}"
        if ib:
            macros["rFourBaseLens"] = f"{ib['safety_auc_lens']:.3f}"
            macros["rFourBaseLogit"] = f"{ib['safety_auc_logitlens']:.3f}"

    # ---- v2.13: lexicon-perturbation stability of SafetyAUC (W2) ----
    lstab = ROOT / "artifacts" / "p1" / "lexicon_stability.json"
    if lstab.exists():
        d = json.loads(lstab.read_text())
        ms = d.get("models", {})
        if ms:
            worst = max(m["max_abs_shift"] for m in ms.values())
            sd_max = max(m["resample_sd"] for m in ms.values())
            sd_mean = sum(m["resample_sd"] for m in ms.values()) / len(ms)
            macros["lexStabWorst"] = f"{worst:.2f}"
            macros["lexStabSd"] = f"{sd_mean:.3f}"
            macros["lexStabWorstSd"] = f"{sd_max:.2f}"
            macros["lexStabN"] = str(len(ms))
            macros["lexStabK"] = str(d["meta"]["k_resamples"])
        pr = d.get("pairs", {})
        big = pr.get("qwen15b_instruct|qwen15b_base", {})
        if big:
            macros["lexStabLargeGapVal"] = f"{big['ref_gap']:+.2f}"
            macros["lexStabLargeGapOrder"] = (
                f"{big['tuned_gt_base_resamples']}/{big['k']}")
        smalls = [v["tuned_gt_base_resamples"] for k, v in pr.items()
                  if k != "qwen15b_instruct|qwen15b_base"]
        if smalls:
            macros["lexStabSmallGapOrder"] = (
                f"{min(smalls)}--{max(smalls)} of {max(v['k'] for v in pr.values())}")
        q05 = pr.get("qwen05b_instruct|qwen05b_base", {})
        if q05:
            macros["lexStabDivOrder"] = (
                f"{q05['k'] - q05['tuned_gt_base_resamples']}/{q05['k']}")
    NUM.write_text("\n".join(
        f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
    print(f"macros: {len(macros)} -> {NUM}")
    print("tables ->", TAB)


if __name__ == "__main__":
    main()

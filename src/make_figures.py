"""Paper figure suite (publication style, vector PDF into paper/figures/).

Style: Okabe-Ito colorblind-safe palette, 8-9pt fonts matching the two-column
template, seed-spread bands, direct layer annotations. Every panel degrades
gracefully to "pending" markers when its data has not landed yet.

Run: .venv/bin/python -m src.make_figures [--out paper/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import math
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
P1 = ROOT / "artifacts" / "p1"
DPO = ROOT / "artifacts" / "dpo"
OUT = ROOT / "paper" / "figures"

def _lighten(hex_color: str, f: float = 0.38) -> str:
    """Blend a hex color toward white (fig-4-only pastel variant)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c + (255 - c) * f) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


OI = {  # Okabe-Ito
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}
FAMILIES = {  # pair key -> (pretty name, instruct color, base color)
    "smol135": ("SmolLM2-135M", OI["blue"], OI["orange"]),
    "smol360": ("SmolLM2-360M", OI["blue"], OI["orange"]),
    "qwen05": ("Qwen2.5-0.5B", OI["green"], OI["verm"]),
    "qwen15": ("Qwen2.5-1.5B", OI["purple"], OI["sky"]),
}
SLUGS = {
    "smol135": ("smol135m_instruct", "smol135m_base"),
    "smol360": ("smol360m_instruct", "smol360m_base"),
    "qwen05": ("qwen05b_instruct", "qwen05b_base"),
    "qwen15": ("qwen15b_instruct", "qwen15b_base"),
}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_agg():
    f = P1 / "aggregate.json"
    return json.loads(f.read_text()) if f.exists() else {"aggregates": {}, "pair_deltas": {}}


def curves_for(agg, slug, key_mean, key_perlayer=None):
    e = agg["aggregates"].get(slug)
    if e is None:
        return None, None, None
    x = np.arange(len(e["per_layer_auc_safety_mean"])) if key_perlayer else None
    if key_perlayer == "safety":
        y = np.array(e["per_layer_auc_safety_mean"])
    elif key_perlayer == "amp":
        y = np.array(e["amplification_mean"])
    else:
        y = None
    n = e["n_seeds"]
    # spread across seeds from the individual run files
    vals = []
    for f in sorted(P1.glob(f"{slug}/*_calibwikitext_w0.json")):
        r = json.loads(f.read_text())
        if key_perlayer == "safety":
            vals.append(r["per_layer_auc"]["safety"])
        elif key_perlayer == "amp":
            vals.append(r["amplification"])
    if vals:
        arr = np.array(vals, dtype=float)
        return x, arr.mean(0), arr.std(0)
    return x, y, None


def pending(ax, msg="data pending"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="gray", style="italic")
    ax.set_xticks([]); ax.set_yticks([])


def fig_schematic(path):
    """Fig 1: latent<->J-space pipeline — true TikZ vector reconstruction.

    Source of truth: paper/fig1_schematic.tex (Latin-Modern serif + typeset
    math, pastel fills, PI design spec 2026-09-30). Compiled here so the
    figure pipeline stays one command; the standalone class pins the page
    to the manuscript column width (394pt)."""
    import shutil, subprocess, tempfile
    src = Path(__file__).resolve().parent.parent / "paper" / "fig1_schematic.tex"
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"-output-directory={td}",
             str(src)], capture_output=True, text=True)
        pdf = Path(td) / "fig1_schematic.pdf"
        if r.returncode != 0 or not pdf.exists():
            raise RuntimeError(f"fig1 tikz compile failed:\n{r.stdout[-3000:]}")
        shutil.copy(pdf, path)
    _pdf_to_png(path)

def _pdf_to_png(pdf_path):
    """PNG twin for quick review / visual gate (same name as the PDF)."""
    import subprocess
    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "200", str(pdf_path),
         str(Path(pdf_path).with_suffix(""))], check=True)

def _save_twin(fig, path):
    """PNG twin for quick review / visual gate."""
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def fig_perlayer(agg, path, key, one_row=False):
    """Fig 2/3: per-layer curves, 2x2 family panels, instruct vs base.
    one_row=True renders the same four panels as a single 1x4 strip (TMLR
    single-column build); axes.flat below is layout-agnostic."""
    if one_row:
        # amp strips share one log y-axis: the ~5-char log tick labels
        # (3x10^0 ...) in EVERY panel compressed the boxes narrow-tall;
        # sharing + labelling once widens them square (matches fig-2 strip)
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharex=False,
                                 sharey=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), sharex=False)
    for ax, pk in zip(axes.flat, FAMILIES):
        name, ci, cb = FAMILIES[pk]
        si, sb = SLUGS[pk]
        drew = False
        for slug, color, label, ls in ((si, ci, "instruct", "-"), (sb, cb, "base", "--")):
            x, m, s = curves_for(agg, slug, None, "safety" if key == "auc" else "amp")
            if m is None:
                continue
            drew = True
            ax.plot(x, m, ls, color=color, lw=1.4, label=label)
            if s is not None:
                ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)
        if not drew:
            pending(ax, f"{name}\ndata pending")
            continue
        ax.set_title(name)
        ax.set_xlabel("layer $\\ell$")
        if key == "auc":
            ax.set_ylabel("per-layer SafetyAUC")
            ax.axhline(0.5, color="gray", lw=0.7, ls=":")
            ax.set_ylim(0.2, 1.0)
        else:
            ax.set_ylabel("transport amplification $A_\\ell$")
            ax.axhline(1.0, color="gray", lw=0.7, ls=":")
            ax.set_yscale("log")
        ax.legend(loc="best", frameon=False)
    if one_row:
        # one shared y-axis per strip: scale labeled once, leftmost panel
        for ax_ in list(axes.flat)[1:]:
            ax_.set_ylabel("")
            ax_.tick_params(labelleft=False)
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)


def fig_bridge(agg, path):
    """Fig 4: amplification vs per-layer SafetyAUC (the latent<->J-space bridge)."""
    # landscape boxes: smaller tick fonts + reduced height widen the boxes
    # relative to their height (PI request: square-ish but wide, light colors)
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.85), sharey=True)
    panels = [("smol135", "smol360"), ("qwen05", "qwen15")]
    for ax, pks in zip(axes, panels):
        slot = 0                    # one r-label slot per series: 4 labels
        for pk in pks:              # on 2 slots used to overprint
            name, ci, cb = FAMILIES[pk]
            si, sb = SLUGS[pk]
            for slug, color, label, mk in ((si, ci, f"{name} instruct", "o"),
                                           (sb, cb, f"{name} base", "^")):
                xi, mi, _ = curves_for(agg, slug, None, "amp")
                _, ms, _ = curves_for(agg, slug, None, "safety")
                if mi is None or ms is None:
                    continue
                ax.scatter(mi, ms, s=13, color=_lighten(color), marker=mk,
                           alpha=0.9, label=label, edgecolors="none")
                ok = np.isfinite(mi) & np.isfinite(ms)
                if ok.sum() > 4:
                    r = np.corrcoef(mi[ok], ms[ok])[0, 1]
                    ax.annotate(f"$r={r:+.2f}$",
                                xy=(0.97, 0.04 + 0.088 * slot),
                                xycoords="axes fraction", ha="right",
                                fontsize=7, color=_lighten(color, 0.22))
                slot += 1                    # per SERIES (member), not family
        ax.set_xscale("log")
        ax.set_xlabel("amplification $A_\\ell$ (log)")
        # explicit sparse ticks, plain %g labels, no minor labels: the
        # default log locator promotes adjacent minors (8x10^-2, 9x10^-1)
        # that collide at this panel size
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 4.0, 8.0)))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        # data-aware round ticks: the qwen panel spans only 0.8-1.7 (log),
        # so fixed candidates collapse to a single tick — sample the range
        lo, hi = ax.get_xlim()
        cands = sorted({round(v, 1) for v in np.geomspace(lo, hi, 14)}
                       | {0.5, 1.0, 2.0, 3.0})
        cands = [t for t in cands if lo * 1.02 <= t <= hi / 1.02]
        if len(cands) > 5:                    # thin to <= 5, evenly in log
            idx = np.linspace(0, len(cands) - 1, 5).round().astype(int)
            cands = [cands[i] for i in dict.fromkeys(idx)]
        ax.set_xticks(cands)
        ax.set_xticklabels([f"{t:g}" for t in cands])
        ax.tick_params(axis="x", labelsize=6.5)   # smaller x numbers
        ax.tick_params(axis="y", labelsize=7)
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    axes[0].set_ylabel("per-layer SafetyAUC")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.14))
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)


def fig_gap(agg, path):
    """Fig 5: paired base->instruct deltas with per-seed points."""
    metrics = [("auc_safety", "SafetyAUC"), ("auc_compliance", "ComplAUC"),
               ("headroom", "Safety Headroom")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5))
    for ax, (key, label) in zip(axes, metrics):
        for pk in FAMILIES:
            d = agg["pair_deltas"].get(pk, {})
            if key not in d or not d.get("seeds"):
                continue
            name, _, _ = FAMILIES[pk]
            _, cb = SLUGS[pk][1], None
            base_e = agg["aggregates"].get(SLUGS[pk][1])
            ins_e = agg["aggregates"].get(SLUGS[pk][0])
            if not base_e or not ins_e:
                continue
            yb, yi = base_e[key]["mean"], ins_e[key]["mean"]
            xs = {"smol135": 0, "smol360": 1, "qwen05": 2, "qwen15": 3}[pk]
            ax.plot([xs, xs], [yb, yi], "-", color="#BBBBBB", lw=1.2, zorder=1)
            ax.scatter([xs] * len(base_e[key]["values"]), base_e[key]["values"],
                       s=14, color=OI["orange"], marker="^", zorder=2,
                       edgecolors="none")
            ax.scatter([xs] * len(ins_e[key]["values"]), ins_e[key]["values"],
                       s=14, color=OI["blue"], marker="o", zorder=2,
                       edgecolors="none")
        ax.set_xticks(range(4))
        ax.set_xticklabels(["135M", "360M", "0.5B", "1.5B"])
        if key == "auc_safety":
            ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_title(label)
    axes[0].set_ylabel("AUC")
    axes[2].set_ylabel("SH")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=OI["blue"], label="instruct"),
               Line2D([], [], marker="^", ls="", color=OI["orange"], label="base")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)


def fig_robustness(agg, path, one_row=False):
    """Fig 6: H5 robustness — seeds, corpus, k, target window.
    one_row=True renders 1x4; the axes array is reshaped back to (2,2) so the
    panel body's axes[0,0]..[1,1] indexing maps onto the visual strip."""
    # pastel variants + landscape boxes in the strip (PI), matching fig 4
    C_I = _lighten(OI["blue"]) if one_row else OI["blue"]
    C_B = _lighten(OI["orange"]) if one_row else OI["orange"]
    if one_row:
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05))
        axes = np.asarray(axes).reshape(2, 2)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8))
    ax = axes[0, 0]
    # (a) seed spread, all models
    order = [s for p in SLUGS for s in SLUGS[p]]
    for i, slug in enumerate(order):
        vals = [json.loads(f.read_text())["auc_safety"]
                for f in sorted(P1.glob(f"{slug}/*_calibwikitext_w0.json"))]
        if not vals:
            continue
        ins = "instruct" in slug
        ax.scatter([i] * len(vals), vals, s=16,
                   color=C_I if ins else C_B,
                   marker="o" if ins else "^", edgecolors="none")
        ax.plot([i - 0.25, i + 0.25], [np.mean(vals)] * 2, lw=1.4,
                color=C_I if ins else C_B)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([s.replace("smol", "S").replace("qwen", "Q").replace("b_", "\n")
                        .replace("m_", "\n") for s in order], fontsize=5.5,
                       rotation=45, ha="right")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_ylabel("SafetyAUC")
    ax.set_title("(a) lens-fit seed spread", fontsize=8.5)

    # (b) calibration corpus ablation (135M pair)
    ax = axes[0, 1]
    kinds = ["wikitext", "generic", "safety"]
    width = 0.35
    for j, (slug, color, lab) in enumerate((("smol135m_instruct", C_I, "instruct"),
                                            ("smol135m_base", C_B, "base"))):
        means, stds = [], []
        for kind in kinds:
            vals = [json.loads(f.read_text())["auc_safety"]
                    for f in sorted(P1.glob(f"{slug}/*_calib{kind}_w0.json"))]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else np.nan)
        ax.bar(np.arange(3) + (j - 0.5) * width, means, width, yerr=stds,
               color=color, label=lab, error_kw={"lw": 0.8}, capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["wiki", "generic", "safety"])
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_ylabel("SafetyAUC"); ax.legend(frameon=False)
    ax.set_title("(b) calibration corpus", fontsize=8.5)

    # (c) k ablation from stored top tokens (135M pair, seed 0)
    ax = axes[1, 0]
    try:
        from .k_ablation import k_curve
        for slug, color, lab in (("smol135m_instruct", C_I, "instruct"),
                                 ("smol135m_base", C_B, "base")):
            ks, aucs = k_curve(slug, seed=0)
            if ks:
                ax.plot(ks, aucs, "-o", ms=3, color=color, label=lab)
    except Exception as e:
        pending(ax, f"k-ablation pending ({type(e).__name__})")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xlabel("readout depth $k$"); ax.set_ylabel("SafetyAUC")
    ax.legend(frameon=False); ax.set_title("(c) top-$k$ depth", fontsize=8.5)

    # (d) target window (135M pair, seed 0)
    ax = axes[1, 1]
    for j, (slug, color, lab) in enumerate((("smol135m_instruct", C_I, "instruct"),
                                            ("smol135m_base", C_B, "base"))):
        vals = []
        for w in (0, 1):
            fs = sorted(P1.glob(f"{slug}/*_calibwikitext_w{w}.json"))
            vals.append(np.mean([json.loads(f.read_text())["auc_safety"] for f in fs])
                        if fs else np.nan)
        ax.bar(np.arange(2) + (j - 0.5) * 0.35, vals, 0.35, color=color, label=lab)
    ax.set_xticks(range(2)); ax.set_xticklabels(["$W{=}0$ (same-pos)", "$W{=}1$ (future)"])
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_ylabel("SafetyAUC"); ax.legend(frameon=False)
    ax.set_title("(d) lens target window", fontsize=8.5)

    if one_row:
        # single-line labels collide in the narrow 1x4 panel
        axes[1, 1].set_xticklabels(["$W{=}0$\nsame-pos", "$W{=}1$\nfuture"])
        # one shared 'SafetyAUC' label (leftmost): the repeated rotated label
        # ate ~0.3in of every box's width -> tall-narrow boxes; panels keep
        # their own ticks (y-ranges differ, so no sharey here)
        for ax_ in (axes[0, 1], axes[1, 0], axes[1, 1]):
            ax_.set_ylabel("")
        for ax_ in np.ravel(axes):
            ax_.tick_params(axis="x", labelsize=6)   # smaller x numbers
            ax_.tick_params(axis="y", labelsize=6.5)
            leg = ax_.get_legend()
            if leg is not None:
                for t in leg.get_texts():
                    t.set_fontsize(6)
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)


def fig_dpo(path, pastel=False):
    """Fig 7: P2 — 4 model rows x 3 panels: training, pre/post metrics, amp shift.

    Rows: SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B, Qwen2.5-1.5B (the P2-at-scale
    GPU extension). A row renders only if its training artifacts exist, so the
    figure degrades gracefully before the qwen DPO runs land."""
    C_VAN = _lighten(OI["blue"]) if pastel else OI["blue"]
    C_JPN = _lighten(OI["verm"]) if pastel else OI["verm"]
    C_ORA = _lighten(OI["orange"]) if pastel else OI["orange"]
    models = [("smol135m", "SmolLM2-135M"), ("smol360m", "SmolLM2-360M"),
              ("qwen05b", "Qwen2.5-0.5B"), ("qwen15b", "Qwen2.5-1.5B")]
    letters = "abcdefghijkl"
    models = [(k, n) for k, n in models
              if (DPO / f"{k}_base_vanilla_lam1.0" / "loss_log.json").exists()
              or (DPO / f"{k}_base_jpen_lam1.0" / "loss_log.json").exists()]
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.0, 2.15 * n_rows + 0.4),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    axes = np.atleast_2d(axes)
    for r, (mkey, mname) in enumerate(models):
        # ---- (a/e/i/m) training curves: DPO-term loss comparable across variants
        ax = axes[r, 0]
        for v, color, lab in (("vanilla", C_VAN, "vanilla"),
                              ("jpen", C_JPN, "J-penalized")):
            f = DPO / f"{mkey}_base_{v}_lam1.0" / "loss_log.json"
            if not f.exists():
                continue
            log = json.loads(f.read_text())
            ax.plot([e["step"] for e in log], [e["loss_dpo"] for e in log],
                    lw=1.1, color=color, label=lab)
            if v == "jpen":
                ax2 = ax.twinx()
                ax2.plot([e["step"] for e in log], [e["penalty"] for e in log],
                         lw=0.9, ls=":", color=C_JPN, alpha=0.65)
                ax2.set_ylabel("penalty $\\lambda\\cdot$DCG", fontsize=6.5,
                               color=C_JPN)
                ax2.tick_params(labelsize=6)
                ax2.spines[["top"]].set_visible(False)
        ax.set_xlabel("step")
        if r == 0:
            ax.set_ylabel("DPO-term loss")
        ax.set_title(f"({letters[3*r]}) {mname}: training", fontsize=8.5)
        if r == 0:
            ax.legend(frameon=False, fontsize=6.2, loc="upper right")
        # ---- (b/f/j/n) pre/post bars
        ax = axes[r, 1]
        groups = [("pre", f"{mkey}_base_dpo_base"),
                  ("vanilla", f"{mkey}_base_dpo_vanilla_lam1.0"),
                  ("J-pen", f"{mkey}_base_dpo_jpen_lam1.0")]
        for i, (lab, slug) in enumerate(groups):
            f = P1 / slug / "seed0_calibwikitext_w0.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            ax.bar(i - 0.18, d["auc_safety"], 0.34, color=C_VAN)
            ax.bar(i + 0.18, d["auc_compliance"], 0.34, color=C_ORA,
                   alpha=0.75)
            rr = d.get("refusal_rate")
            if rr is not None:
                ax.text(i, 0.03, f"ref {rr:.2f}", ha="center", fontsize=6,
                        transform=ax.get_xaxis_transform())
        ax.axhline(0.5, color="gray", lw=0.7, ls=":")
        ax.set_xticks(range(3))
        ax.set_xticklabels([g[0] for g in groups], fontsize=6.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"({letters[3*r+1]}) {mname}: J-space pre/post", fontsize=8.5)
        # ---- (c/g/k/o) amplification shift
        ax = axes[r, 2]
        pre = P1 / f"{mkey}_base_dpo_base" / "seed0_calibwikitext_w0.json"
        if pre.exists():
            a0 = json.loads(pre.read_text())["amplification"]
            ax.plot(a0, lw=1.0, color="#888888", ls="--", label="pre (base)")
        for v, color, lab in (("vanilla", C_VAN, "vanilla"),
                              ("jpen", C_JPN, "J-pen")):
            f = P1 / f"{mkey}_base_dpo_{v}_lam1.0" / "seed0_calibwikitext_w0.json"
            if f.exists():
                ax.plot(json.loads(f.read_text())["amplification"], lw=1.1,
                        color=color, label=lab)
        ax.axhline(1.0, color="gray", lw=0.7, ls=":")
        ax.set_xlabel("layer")
        if r == 0:
            ax.set_ylabel("$A_\\ell$")
        ax.set_title(f"({letters[3*r+2]}) {mname}: $A_\\ell$ shift", fontsize=8.5)
        ax.legend(frameon=False, fontsize=6.2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)



def _kde(vals, xs):
    """Silverman-bandwidth Gaussian KDE (numpy-only; scipy not guaranteed)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    h = max(0.9 * sd * len(v) ** (-1 / 5), 1e-3)
    k = np.exp(-0.5 * ((xs[:, None] - v[None, :]) / h) ** 2).sum(1)
    return k / (len(v) * h * np.sqrt(2 * np.pi))



def fig_n50(path):
    """Fig 12 (PI review, not yet wired): program-17 n=50 widening results.

    (a) Beta(1+s, 1+f) posterior densities of the per-cell success rate
        (uniform prior, n=50) as ridgelines, vanilla vs silent per target.
    (b) paired n=15 -> n=50 dumbbells: the n=15 MLE (nested first-15 prompts
        of the same run) connected to the n=50 MLE, with Wilson 95% CIs.
    All numbers recomputed from the per-prompt JSONs (program 17)."""
    P3 = ROOT / "artifacts" / "p3"
    targets = [("smol135m_base_vanilla_lam1.0", "135M (DPO)"),
               ("Qwen2.5-0.5B-Instruct", "0.5B"),
               ("Qwen2.5-1.5B-Instruct", "1.5B")]
    C_VAN, C_SIL = _lighten(OI["blue"]), _lighten(OI["verm"])
    E_VAN, E_SIL = OI["blue"], OI["verm"]

    def wilson(s, n):
        p = s / n
        z = 1.96
        den = 1 + z*z/n
        c = (p + z*z/(2*n)) / den
        hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
        return max(0.0, c-hw), min(1.0, c+hw)

    def beta_pdf(xs, s, f):
        a, b = 1+s, 1+f
        lg = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b)
        return [math.exp(lg + (a-1)*math.log(x) + (b-1)*math.log(1-x)) for x in xs]

    cells = {}
    for slug, pretty in targets:
        for arm in ("vanilla", "silent"):
            fs = sorted((P3 / slug / f"{arm}_n50").glob("prompt*.json"))
            succ = [int(json.loads(f.read_text())["affirm_success"]) for f in fs]
            s50 = sum(succ[:50]); n50 = len(succ)
            s15 = sum(succ[:15]); n15 = 15
            cells[(pretty, arm)] = dict(s50=s50, n50=n50, s15=s15, n15=n15)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    xs = np.linspace(0.001, 0.999, 300)

    # ---- (a) posterior ridgelines ----
    ax = axes[0]
    y = 0.0
    yticks, ylabels = [], []
    handles_a = [plt.Line2D([], [], color=C_VAN, lw=6, alpha=0.55, label="vanilla"),
                 plt.Line2D([], [], color=C_SIL, lw=6, alpha=0.55, label="silent")]
    for slug, pretty in targets:
        top = y
        for arm, color, edge in (("vanilla", C_VAN, E_VAN),
                                 ("silent", C_SIL, E_SIL)):
            c = cells[(pretty, arm)]
            dens = beta_pdf(xs, c["s50"], c["n50"] - c["s50"])
            k = max(dens)
            ax.fill_between(xs, y, y + 0.62*np.array(dens)/k, color=color,
                            alpha=0.55, lw=0.7, ec=edge)
            ax.plot([c["s50"]/c["n50"]]*2, [y - 0.06, y + 0.62*0.9],
                    color=edge, lw=0.8)
            y -= 0.72
        yticks.append((top + y + 0.72)/2); ylabels.append(pretty)
        y -= 0.62
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("success rate\n(Beta posterior, uniform prior, $n{=}50$)",
                  fontsize=8, labelpad=8)
    ax.set_title("(a) posterior densities per cell", fontsize=8.5)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(handles=handles_a, frameon=False, fontsize=6.5,
              loc="upper right")
    ax.set_xlim(0, 1)

    # ---- (b) paired dumbbells n15 -> n50 ----
    ax = axes[1]
    y = 0.0
    yticks, ylabels = [], []
    for slug, pretty in targets:
        for arm, color, edge in (("vanilla", C_VAN, E_VAN),
                                 ("silent", C_SIL, E_SIL)):
            c = cells[(pretty, arm)]
            p15 = c["s15"]/c["n15"]; p50 = c["s50"]/c["n50"]
            lo, hi = wilson(c["s50"], c["n50"])
            ax.plot([p15, p50], [y, y], color=edge, lw=1.0, alpha=0.7)
            ax.plot([p15], [y], "o", ms=4.5, color=color,
                    markeredgecolor=edge, markeredgewidth=0.7)
            ax.plot([p50], [y], "o", ms=5.5, color="white",
                    markeredgecolor=edge, markeredgewidth=1.0)
            ax.plot([lo, hi], [y, y], "_", color=edge, lw=1.6, alpha=0.9)
            yticks.append(y); ylabels.append(f"{pretty} {arm}")
            y -= 1.0
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=6.8)
    ax.set_xlabel("success rate\n(filled: $n{=}15$ MLE, open: $n{=}50$ MLE,\n"
                  "whiskers: Wilson 95 percent CI)", fontsize=7.2, labelpad=8)
    ax.set_title("(b) small-cell stability ($n{=}15$ vs $n{=}50$)", fontsize=8.5)
    ax.set_xlim(-0.03, 1.03)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    fig.savefig(path)
    _pdf_to_png(path)
    plt.close(fig)


def _prompts(d):
    """Per-prompt attack records, ordered."""
    import glob as _g
    out = []
    for f in sorted(_g.glob(str(d / "prompt*.json"))):
        out.append(json.loads(open(f).read()))
    return out


def fig_p3(path):
    """Fig 10: P3/H4 — adversarial evasion of the J-space monitor.

    Per target x arm: behavioral success vs. prompt-side and response-side
    monitor alarm rates (left), and the monitor reading T_saf with the benign
    calibration band (right). Rows render only for targets with summaries."""
    P3 = ROOT / "artifacts" / "p3"
    targets = [("smol135m_base_vanilla_lam1.0", "135M (DPO)"),
               ("Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B"),
               ("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B")]
    rows = []
    taus = {}
    for slug, pretty in targets:
        per_arm = {}
        for arm in ("vanilla", "silent"):
            f = P3 / slug / arm / "summary.json"
            if f.exists():
                per_arm[arm] = json.loads(f.read_text())
        if per_arm:
            cal = P3 / slug / "monitor_calibration.json"
            if cal.exists():
                taus[pretty] = json.loads(cal.read_text()).get("tau95")
            rows.append((pretty, per_arm))
    if not rows:
        print("fig10 skipped (no p3 summaries)")
        return
    n = len(rows)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.1),
                             gridspec_kw={"width_ratios": [1.15, 1],
                                          "height_ratios": [1.0, 1.25]})
    ax = axes[0, 0]
    xs = np.arange(n)
    w = 0.27
    # pastel arm colors, consistent with figs 2-7; shared by panels a-d
    C_VAN, C_SIL = _lighten(OI["blue"]), _lighten(OI["verm"])
    arms = [("vanilla", "GCG (compliance)", C_VAN),
            ("silent", "silent (comp. $-$ monitor)", C_SIL)]
    for i, (arm, lab, color) in enumerate(arms):
        succ = [r[arm]["affirm_success_rate"] if arm in r else np.nan
                for _, r in rows]
        alarm = [r[arm].get("alarm_rate", np.nan) if arm in r else np.nan
                 for _, r in rows]
        ax.bar(xs + (i - 0.5) * w, succ, w * 0.92, color=color, label=lab)
        ax.bar(xs + (i - 0.5) * w, alarm, w * 0.92, facecolor="white",
               edgecolor=color, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([p for p, _ in rows], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("rate")
    ax.set_title("(a) behavioral success (solid) vs. monitor alarm (open)",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax = axes[0, 1]
    for i, (arm, lab, color) in enumerate(arms):
        T = [r[arm].get("T_saf_mean", np.nan) if arm in r else np.nan
             for _, r in rows]
        Tresp = [r[arm].get("T_saf_response_mean", np.nan) if arm in r else np.nan
                 for _, r in rows]
        ax.plot(xs + (i - 0.5) * w, T, "o--", color=color, lw=1.0, ms=4,
                label=f"{lab}: prompt")
        ax.plot(xs + (i - 0.5) * w, Tresp, "s:", color=color, lw=1.0, ms=4,
                alpha=0.75, label=f"{lab}: response")
    # per-target alarm thresholds: one short segment above each x position
    # (a single global line is wrong — tau95 differs per target and qwen15's
    # 25.5 pinned the axis). Label drawn once, left of the first segment.
    labelled = False
    for i, (pretty, _) in enumerate(rows):
        t = taus.get(pretty)
        if t is None:
            continue
        ax.plot([i - 0.32, i + 0.32], [t, t], color="gray", lw=1.1, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([p for p, _ in rows], fontsize=7.5)
    ax.set_ylabel("monitor reading $T_{\\mathrm{saf}}$")
    ax.set_title("(b) prompt-side vs. response-side monitor", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", ncol=1)

    # ---- (c) raincloud: prompt-side monitor-reading distributions ----
    # benign (n=250, from the 5% FPR calibration), vanilla (n=15), silent
    # (n=15) per target. Half-violin + thin box + raw points: the raw-data
    # style recommended for small-n comparisons (Allen et al. 2019).
    ax = axes[1, 0]
    C_BEN = "#9AA0A6"
    y = 0.0
    yticks, ylabels, tau_spans = [], [], []
    for ti, (pretty, _) in enumerate(rows):
        slug = [sl for sl, pr in targets if pr == pretty][0]
        cal = json.loads((P3 / slug / "monitor_calibration.json").read_text())
        ben = np.asarray(cal["benign_T"], float)
        block_top = y
        for gi, (vals, color) in enumerate((
                (ben, C_BEN),
                ([pp.get("T_saf_final") for pp in _prompts(P3 / slug / "vanilla")], C_VAN),
                ([pp.get("T_saf_final") for pp in _prompts(P3 / slug / "silent")], C_SIL))):
            vals = np.asarray([v for v in vals if v is not None], float)
            yc = y - 0.30
            # half-violin (upper half only), skipped for degenerate all-zero
            if vals.std() > 1e-9 and len(vals) >= 3:
                xmax = max(vals.max(), np.percentile(vals, 99) * 1.15)
                xs_ = np.linspace(0, xmax, 120)
                k = _kde(vals, xs_)
                k = k / k.max()
                ax.fill_between(xs_, yc, yc + 0.24 * k, color=color,
                                alpha=0.85, lw=0.5, ec=color)
                bp = ax.boxplot(vals, vert=False, positions=[yc + 0.015],
                                widths=0.07, showcaps=False, showfliers=False,
                                boxprops=dict(color=color, lw=0.8),
                                medianprops=dict(color=color, lw=0.9),
                                whiskerprops=dict(color=color, lw=0.7))
                for sp in bp["whiskers"]:
                    sp.set_linewidth(0.7); sp.set_color(color)
            # raw points under the violin
            jit = (np.random.RandomState(7 + ti * 10 + gi).uniform(-0.045, 0.045, len(vals)))
            ax.plot(vals, yc - 0.13 + jit, "o", ms=2.6, color=color, alpha=0.75,
                    markeredgecolor="none")
            yticks.append(yc)
            ylabels.append("")
            y -= 0.62
        # tau95 marker for this target block
        t = taus.get(pretty)
        if t is not None:
            ax.plot([t, t], [block_top - 0.02, y + 0.30], color="gray", lw=1.0,
                    ls="--")
        # target label at block center
        yticks.append((block_top + y + 0.30) / 2)
        ylabels.append(pretty)
        y -= 0.55
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.5)
    ax.set_xlabel("prompt-side monitor reading $T_{\\mathrm{saf}}$", fontsize=8.5)
    ax.set_title("(c) monitor-reading distributions (raincloud)", fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles_c = [plt.Line2D([], [], marker="s", ls="", ms=6, color=C_BEN, label="benign (n=250)"),
                 plt.Line2D([], [], marker="s", ls="", ms=6, color=C_VAN, label="vanilla (n=15)"),
                 plt.Line2D([], [], marker="s", ls="", ms=6, color=C_SIL, label="silent (n=15)")]
    ax.legend(handles=handles_c, frameon=False, fontsize=6.3, loc="upper right")
    ax.text(0.985, 0.02, "dashed: $\\tau_{95}$", transform=ax.transAxes,
            fontsize=6.3, color="gray", ha="right")

    # ---- (d) dissociation scatter: emptied monitor vs retained latent ----
    ax = axes[1, 1]
    mk = {"135M (DPO)": "o", "Qwen2.5-0.5B": "s", "Qwen2.5-1.5B": "^"}
    for pretty, _ in rows:
        slug = [sl for sl, pr in targets if pr == pretty][0]
        cal = json.loads((P3 / slug / "monitor_calibration.json").read_text())
        med, sd, tau = cal["benign_median"], cal["benign_sd"], cal["tau95"]
        zt = (tau - med) / sd
        ax.axvline(zt, color="gray", lw=0.8, ls=":", alpha=0.8)
        for arm, color in (("vanilla", C_VAN), ("silent", C_SIL)):
            pp = _prompts(P3 / slug / arm)
            x = [(p.get("T_saf_final") - med) / sd for p in pp]
            yv = [p.get("latent_final") for p in pp]
            ax.plot(x, yv, mk[pretty], ms=4.2, color=color, alpha=0.8,
                    markeredgecolor="white", markeredgewidth=0.4,
                    label=f"{pretty}: {arm}" if pretty == "135M (DPO)" else None)
    ax.axvline(0, color="black", lw=0.8)
    ax.text(0.02, 0.98, "benign median", transform=ax.get_xaxis_transform(),
            fontsize=6.3, color="black", ha="left", va="top", rotation=90)
    ax.set_xlabel("$z(T_{\\mathrm{saf}})$  (benign median $= 0$, $\\tau_{95}$ dotted)",
                  fontsize=8.5)
    ax.set_ylabel("latent danger score", fontsize=8.5)
    ax.set_title("(d) dissociation: accessible vs. latent", fontsize=8.5)
    ax.legend(frameon=False, fontsize=5.6, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(path)
    _save_twin(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--one-row", action="store_true",
                    help="render fig2/fig3/fig6 panels as 1x4 strips "
                         "(TMLR single-column layout)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    agg = load_agg()

    fig_schematic(out / "fig1_schematic.pdf")
    print("fig1 done")
    fig_perlayer(agg, out / "fig2_perlayer_auc.pdf", "auc",
                 one_row=args.one_row)
    print("fig2 done")
    fig_perlayer(agg, out / "fig3_amplification.pdf", "amp",
                 one_row=args.one_row)
    print("fig3 done")
    fig_bridge(agg, out / "fig4_bridge.pdf")
    print("fig4 done")
    fig_gap(agg, out / "fig5_accessibility_gap.pdf")
    print("fig5 done")
    fig_robustness(agg, out / "fig6_robustness.pdf", one_row=args.one_row)
    print("fig6 done")
    fig_dpo(out / "fig7_dpo.pdf", pastel=True)
    print("fig7 done")
    fig_p3(out / "fig10_p3.pdf")
    fig_n50(out / "fig12_n50.pdf")
    print("fig12 done (PI review: not wired into the paper yet)")
    print("fig10 done")


if __name__ == "__main__":
    main()

# From Latent Space to Jacobian Space: Measuring, Evading, and Training Against Safety-Content Accessibility

---

## Overview

This repository contains the source code and per-run artifacts for the
paper. Safety alignment is verified behaviorally, yet a language model
computes its answer before it emits it: all deliberation, including any
harmful reasoning, occurs in internal activations. The paper measures how
safety training moves safety-relevant content between **latent space** (the
full set of internal representations) and **Jacobian space** (the subset
that is causally poised to surface in outputs), trains against that
accessibility through a J-space-penalized preference objective, and audits
the resulting monitor adversarially with a two-arm, monitor-aware attack.

The three principal findings are: (i) tuned and base models differ in
J-space accessibility in a training-provenance-dependent manner, with a
recognition/behavior divergence at 0.5B where DPO installs refusal while
J-space recognition falls to chance; (ii) a monitor-aware attack suppresses
the prompt-side monitor reading at statistically indistinguishable
behavioral success, while a learned monitor rung resists its own adaptive
re-attack; and (iii) a training-time defense fails its own re-attack at
every penalty weight tested, and the amplification profile is descriptive
rather than causal.

## Repository contents

```
src/          the full pipeline: lens fitting (Hutchinson-style estimator),
              readout and six-axis DCG scoring, metrics, LoRA-DPO training
              (vanilla and J-space-penalized), two-arm GCG-style attacks,
              the J-space-LAT defense, causal steering, rubric judges, and
              the table/figure generation scripts
scripts/      the experiment programs exactly as executed
artifacts/    per-run JSON records, token-level NPZ sidecars, and console
              logs for every experiment reported in the paper
environment.txt   exact Python environment (pip freeze)
LICENSE       MIT
```

## Requirements

- Python 3.12; package versions are pinned in `environment.txt`.
- The primary evaluation protocol is **fp32 on CPU**; no GPU is required to
  reproduce the headline tables from the stored artifacts.
- Optional GPU acceleration (fp16/bf16/NF4/INT8 deployment regimes and
  training) requires an NVIDIA GPU with 24 GB of memory for the 4B/9B
  models.

## Reproduction

Every table and figure in the paper is generated from the stored artifacts
under `artifacts/` by the generation scripts in `src/`; no number is
entered by hand. Re-running the generation scripts on this repository
reproduces the paper's tables and figures exactly. Re-running the
experiments themselves is supported stage-by-stage: each stage skips work
whose output artifact already exists (JSON completion markers), so partial
re-execution is safe.

The artifact record is per-run and immutable: fp32/CPU artifacts are the
primary record and are tagged separately from any GPU-run outputs.

## Data and models

All models and evaluation sets are publicly available (SmolLM2, Qwen2.5,
Qwen3, Gemma-2 families; StrongREJECT; XSTest; wikitext-103). Prompts are
fictional and contain no personal data or human-subject content.



## License

Released under the MIT License (see `LICENSE`).

#!/bin/bash
# GPU program 12 — program-11 REMAINDER (after the gemma-fp32 OOM burned the
# watchdog's restarts) + the original M2 work:
#   A1. ladder re-eval restricted to the three fp32-fittable targets
#       (run_p3_ladder --targets fp32; smol135 ladder never completed).
#   A2. transfer matrix re-run, same exclusion (--eval-targets fp32) —
#       gemma-2-9b-it REMAINS a transfer SOURCE (cross-family rows).
#   A3. R6 J-space-LAT defense (smol135dpo, 3 rounds w/ adversarial refresh)
#       + full attack suite vs the defended checkpoint.
#   A4. R7 causal steering (qwen15 pair).
#   B.  M2 cross-family judge: re-grade the SAME regenerated completions with
#       gemma-2-9b-it (skip-gen) — judge-independence of the R13 conclusions.
#       (gemma as judge here is bf16 ~= 18 GB and FITS; it is only the fp32
#       J-lens eval that does not.)
#   C.  PEZ continuous-prefix baseline (kept from dropped program 10).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program12.log) 2>&1
echo $$ > artifacts/pid_gpu_program12

echo "=== GPU_PROGRAM12_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A1. ladder re-eval (fp32-fittable targets only) ----
$PY -m src.run_p3_ladder --device cuda --targets fp32 || exit 1

# ---- A2. transfer matrix re-run (fp32 targets; gemma stays a source) ----
$PY -m src.run_p3_transfer --device cuda --eval-targets fp32 || exit 1

# ---- A3. R6 defense + re-attack of the defended model ----
$PY -m src.run_p3_defense --target smol135dpo --rounds 3 --device cuda || exit 1
$PY -m src.run_p3_gcg --target-path artifacts/defense/smol135m_base_vanilla_lam1.0_defended_lam1.0/merged \
    --arms both --n-prompts 15 --n-steps 100 --device cuda --run-tag defended || exit 1

# ---- A4. R7 steering (qwen15 pair) ----
$PY -m src.run_steering --pairs qwen15 --device cuda || exit 1

# ---- B. cross-family judge (M2): gemma-2-9b-it re-grades stored completions ----
$PY -m src.run_judge --device cuda --skip-gen \
    --judge google/gemma-2-9b-it --grades-dirname grades_gemma9 || exit 1

# ---- C. PEZ continuous-prefix baseline (from dropped program 10) ----
for T in smol135dpo qwen05instr; do
  $PY -m src.run_p3_pez --model $T --arms both --n-prompts 15 --n-steps 100 \
      --device cuda || exit 1
done

echo "GPU_PROGRAM12_COMPLETE $(date -Is)"

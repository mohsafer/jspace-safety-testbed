#!/bin/bash
# GPU program 13 — program-12 remainder after the R7 crash (corrcoef 27-vs-28
# per-layer mismatch; R6 itself COMPLETED: defended smol135 re-attacked —
# silent arm holds T_saf at 0.00 with success 0.53, tau95 recalibrated to 4.77;
# see artifacts/defense/*/defense_log.json). Stages:
#   A. R7 causal steering (pairing fixed: profile[i] <-> ampl[i+2])
#   B. M2 cross-family judge: gemma-2-9b-it re-grades stored completions
#   C. PEZ continuous-prefix baseline (from dropped program 10)
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program13.log) 2>&1
echo $$ > artifacts/pid_gpu_program13

echo "=== GPU_PROGRAM13_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. R7 steering (both members of the qwen15 pair) ----
$PY -m src.run_steering --pairs qwen15 --device cuda || exit 1

# ---- B. cross-family judge (M2) ----
$PY -m src.run_judge --device cuda --skip-gen \
    --judge google/gemma-2-9b-it --grades-dirname grades_gemma9 || exit 1

# ---- C. PEZ baseline ----
for T in smol135dpo qwen05instr; do
  $PY -m src.run_p3_pez --model $T --arms both --n-prompts 15 --n-steps 100 \
      --device cuda || exit 1
done

echo "GPU_PROGRAM13_COMPLETE $(date -Is)"

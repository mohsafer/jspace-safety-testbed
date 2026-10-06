#!/bin/bash
# GPU program 14 — the last two stages of the campaign (program-13 remainder):
#   A. M2 cross-family judge: gemma-2-9b-it re-grades the SAME regenerated
#      completions (skip-gen); the earlier run crashed in aggregate() because
#      the glob swept its own judge_grades.json output — fixed in run_judge.
#   B. PEZ continuous-prefix baseline (from dropped program 10).
# R7 steering is DONE (artifacts/steering/*.json, corr(prof,amp) -0.47/-0.07);
# R6 defense + re-attack DONE in program 13 attempt 3.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program14.log) 2>&1
echo $$ > artifacts/pid_gpu_program14

echo "=== GPU_PROGRAM14_START $(date -Is) ==="
nvidia-smi | head -12

$PY -m src.run_judge --device cuda --skip-gen \
    --judge google/gemma-2-9b-it --grades-dirname grades_gemma9 || exit 1

for T in smol135dpo qwen05instr; do
  $PY -m src.run_p3_pez --model $T --arms both --n-prompts 15 --n-steps 100 \
      --device cuda || exit 1
done

echo "GPU_PROGRAM14_COMPLETE $(date -Is)"

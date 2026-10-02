#!/bin/bash
# GPU program 15 — R6 lambda-def sweep (reviewer attack #3: "maybe you
# under-trained the defense"). For each lambda_def in {0.5, 2.0, 4.0} around
# the canonical 1.0: train the J-space-LAT defense (3 rounds, adversarial
# refresh) and run the FULL-budget re-attack (15 prompts x both arms x 100
# steps). Resume guards make restarts safe. lambda=1.0 results already exist
# (program 13: p3/merged/{vanilla,silent}_defended).
# Collected at integration time into artifacts/defense/r6_sweep.json.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program15.log) 2>&1
echo $$ > artifacts/pid_gpu_program15

echo "=== GPU_PROGRAM15_START $(date -Is) ==="
nvidia-smi | head -12

for L in 0.5 2.0 4.0; do
  TAG=$(echo "$L" | tr -d '.')
  $PY -m src.run_p3_defense --lam-def "$L" --rounds 3 --device cuda || exit 1
  $PY -m src.run_p3_gcg \
      --target-path "artifacts/defense/smol135m_base_vanilla_lam1.0_defended_lam${L}/merged" \
      --arms both --n-prompts 15 --n-steps 100 --device cuda \
      --run-tag "dlam${TAG}" || exit 1
done

echo "GPU_PROGRAM15_COMPLETE $(date -Is)"

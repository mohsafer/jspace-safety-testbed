#!/bin/bash
# GPU program 9 — NEW-SERVER campaign Phase 6 part 2 (runbook research/10):
# R3, the headline upgrade: gemma-2-9b-it full JADR cross-check on the A30
# (24 GB): bf16 P1 protocol (3 seeds), NF4 axis (seeds 0 1), then the 9B GCG
# attack at chunk 8 (~9 h). The GPU IS the primary protocol at 9B (CPU fp32
# infeasible) — dtype recorded in the usual JSON fields.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program9.log) 2>&1
echo $$ > artifacts/pid_gpu_program9

echo "=== GPU_PROGRAM9_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. P1 protocol, bf16, 3 seeds ----
$PY -m src.run_p1 --pairs gemma29 --seeds 0 1 2 --device cuda --dtype bfloat16 || exit 1

# ---- B. NF4 quantization axis (9B fits ~5 GB weights) ----
$PY -m src.run_p1 --pairs gemma29 --seeds 0 1 --device cuda --quant nf4 || exit 1

# ---- C. 9B attack: chunk 8, bf16 weights ----
$PY -m src.run_p3_gcg --model gemma29it --arms both --n-prompts 15 --n-steps 100 \
    --chunk 8 --device cuda --dtype bfloat16 || exit 1

echo "GPU_PROGRAM9_COMPLETE $(date -Is)"

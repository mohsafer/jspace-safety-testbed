#!/bin/bash
# GPU program 6 — NEW-SERVER campaign, Phase 1 (runbook research/10): R1,
# P2-at-scale completion: Qwen2.5-1.5B DPO vanilla + jpen (OOM'd on the P100
# only under contention; routine on the A30), evaluated under the primary
# fp32/CPU protocol. The base eval (qwen15b_base_dpo_base) is already complete
# from the P100 era — its JSON carries the refusal rate; do not force-rerun.
# No paper build here: numbers land via the campaign-wide regeneration pass.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a artifacts/gpu_program6.log) 2>&1
echo $$ > artifacts/pid_gpu_program6

echo "=== GPU_PROGRAM6_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. R1: qwen15 DPO vanilla (GPU train) + CPU-fp32 eval (primary protocol) ----
$PY -m src.run_dpo  --model qwen15 --variant vanilla --lam 1.0 --device cuda || exit 1
$PY -m src.eval_dpo --model qwen15 --variant vanilla --force || exit 1

# ---- B. R1: qwen15 DPO jpen (GPU train; fits the frozen base lens first) ----
$PY -m src.run_dpo  --model qwen15 --variant jpen --lam 1.0 --device cuda || exit 1
$PY -m src.eval_dpo --model qwen15 --variant jpen --force || exit 1

echo "GPU_PROGRAM6_COMPLETE $(date -Is)"

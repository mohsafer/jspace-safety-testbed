#!/bin/bash
# GPU program 8 — NEW-SERVER campaign Phase 6 part 1 (runbook research/10):
#   A. R2: qwen34 FULL protocol on GPU in true bf16 (100-prompt calib, 3
#      seeds) — the 4B pair moves from reduced-anchor to main-results row
#   B. R5: public abliterated checkpoint (huihui Qwen3-1.7B v2) + base, bf16
#      seeds 0 1 then NF4 seeds 0 1 — JADR's abliterated-recovery headline
#   C. R4: INT8 quantization grid (impossible on Pascal; completes the
#      precision axis INT8/NF4/bf16/fp32)
# bf16 on A30 = true-bf16 weights (no fp16-style overflow; the run_p1 `amp`
# gate intentionally does not trigger for bfloat16).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
exec > >(tee -a artifacts/gpu_program8.log) 2>&1
echo $$ > artifacts/pid_gpu_program8

echo "=== GPU_PROGRAM8_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. R2: qwen34 full protocol, true bf16, 3 seeds ----
$PY -m src.run_p1 --pairs qwen34 --seeds 0 1 2 --device cuda --dtype bfloat16 || exit 1

# ---- B. R5: abliterated recovery (bf16 + NF4) ----
$PY -m src.run_p1 --pairs qwen317abl --seeds 0 1 --device cuda --dtype bfloat16 || exit 1
$PY -m src.run_p1 --pairs qwen317abl --seeds 0 1 --device cuda --quant nf4 || exit 1

# ---- C. R4: INT8 grid, matched seed 10 ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 --device cuda --quant int8 || exit 1

echo "GPU_PROGRAM8_COMPLETE $(date -Is)"

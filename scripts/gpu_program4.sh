#!/bin/bash
# GPU program 4 (after program 3): final parity fixes.
#   1. fp16 parity redo under the FINAL method (fp32 master weights + autocast;
#      the 4 SmolLM files from the partial run used true-fp16 weights)
#   2. Qwen3-4B cross-check (crashed in program 2 on fp16 overflow)
#   3. p3_posthoc rerun (path bug fixed) — response-side monitor results
#   4. final aggregate + tables + figures
# NO paper build: v2.0 is built once after the manual consistency pass.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
exec > >(tee -a artifacts/gpu_program4.log) 2>&1
echo "=== GPU_PROGRAM4_START $(date -Is) ==="

$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --dtype float16 --force
$PY -m src.run_p1 --pairs qwen34 --seeds 0 1 --n-calib 64 --device cuda --dtype float16
$PY -m src.p3_posthoc --device cuda
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures

echo "GPU_PROGRAM4_COMPLETE $(date -Is)"

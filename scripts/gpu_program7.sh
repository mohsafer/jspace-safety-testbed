#!/bin/bash
# GPU program 7 — NEW-SERVER campaign Phases 2-4 (runbook research/10):
#   A. R13 (W3): LLM-judge behavioral grading (src/run_judge.py)
#   B. R11 (W2): instrument triangulation (src/run_instruments.py)
#   C. R12 (W4): monitor-strength ladder (src/run_p3_ladder.py)
# All stages are resumable; artifacts under artifacts/{p3/judge,instruments,p3/ladder}.
# Ladder-table + judge columns enter the paper at the campaign regeneration pass.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
exec > >(tee -a artifacts/gpu_program7.log) 2>&1
echo $$ > artifacts/pid_gpu_program7

echo "=== GPU_PROGRAM7_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. R13 judge (regenerates full continuations, then grades) ----
$PY -m src.run_judge --device cuda || exit 1

# ---- B. R11 instrument triangulation (runbook scope: 4 P1 pairs + qwen34;
#         gemma29/qwen317abl are excluded — 9B fp32 does not fit 24 GB here
#         and the abliterated pair is an R5 row, not a triangulation target) ----
$PY -m src.run_instruments --pairs smol135,smol360,qwen05,qwen15,qwen34 --seeds 0 --device cuda || exit 1

# ---- C. R12 monitor-strength ladder (stored attacks + clean + benign) ----
$PY -m src.run_p3_ladder --device cuda || exit 1

echo "GPU_PROGRAM7_COMPLETE $(date -Is)"

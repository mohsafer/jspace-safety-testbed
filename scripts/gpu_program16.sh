#!/usr/bin/env bash
# program16: CPU recompute of qwen15b_base seed0 (Phase-1 style; skips completed runs)
# + aggregate. Relaunched detached after the 2026-09-29 21:08 silent kill of the
# attached task (no OOM/traceback — harness reaped it mid-eval).
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=artifacts/gpu_program16.log

echo "GPU_PROGRAM16_START $(date -Is)" | tee -a "$LOG"

$PY -m src.run_p1 --pairs qwen15 --seeds 0 --members both 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "GPU_PROGRAM16_FAIL rc=$rc $(date -Is)" | tee -a "$LOG"
  exit "$rc"
fi

$PY -m src.run_p1 --aggregate 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "GPU_PROGRAM16_FAIL_AGG rc=$rc $(date -Is)" | tee -a "$LOG"
  exit "$rc"
fi

echo "GPU_PROGRAM16_COMPLETE $(date -Is)" | tee -a "$LOG"

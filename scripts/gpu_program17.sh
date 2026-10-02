#!/usr/bin/env bash
# program17: widen the P3 attack cells from n=15 to n=50 (reviewer item:
# small cells / wide Wilson CIs). Same deterministic prompt order, so the
# original 15 prompts are a nested subset; fresh arm dirs via --run-tag n50
# keep the primary tables untouched until the PI folds them in.
# Costs (measured): 135M ~3.4h, 0.5B ~12.3h, 1.5B ~33h. Resumable per prompt.
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
LOG=artifacts/gpu_program17.log

echo "GPU_PROGRAM17_START $(date -Is)" | tee -a "$LOG"

for M in smol135dpo qwen05instr qwen15instr; do
  echo "=== target $M START $(date -Is)" | tee -a "$LOG"
  $PY -m src.run_p3_gcg --model "$M" --arms both --n-prompts 50 --n-steps 100 \
      --run-tag n50 --device cuda 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "GPU_PROGRAM17_FAIL $M rc=$rc $(date -Is)" | tee -a "$LOG"
    exit "$rc"
  fi
  echo "=== target $M DONE $(date -Is)" | tee -a "$LOG"
done

echo "GPU_PROGRAM17_COMPLETE $(date -Is)" | tee -a "$LOG"

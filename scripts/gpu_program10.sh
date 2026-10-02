#!/bin/bash
# GPU program 10 — NEW-SERVER campaign Phase 5 (runbook research/10): R15.
#   A. Budget–success curves: smol135dpo + qwen05instr, both arms, at
#      50/200/400 steps (the 100-step primary runs already exist) — tests
#      whether sub-1.0 success is under-convergence. Tagged artifacts
#      (vanilla_b50 etc.) stay out of the primary P3 tables.
#   B. PEZ continuous-prefix baseline at the same budget, both arms, both
#      targets — tests whether the hard-token constraint causes the
#      prompt-side monitor blindness.
# The strongest-rung re-attacks (R12 stage 2) run as program 11 after the
# ladder table selects the rung.
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
exec > >(tee -a artifacts/gpu_program10.log) 2>&1
echo $$ > artifacts/pid_gpu_program10

echo "=== GPU_PROGRAM10_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. budget curves ----
for T in smol135dpo qwen05instr; do
  CHUNK=32
  [ "$T" = "qwen05instr" ] && CHUNK=16
  for S in 50 200 400; do
    $PY -m src.run_p3_gcg --model $T --arms both --n-prompts 15 --n-steps $S \
        --chunk $CHUNK --device cuda --run-tag b$S || exit 1
  done
done

# ---- B. PEZ continuous-prefix baseline (same budget: 100 steps) ----
for T in smol135dpo qwen05instr; do
  $PY -m src.run_p3_pez --model $T --arms both --n-prompts 15 --n-steps 100 \
      --device cuda || exit 1
done

echo "GPU_PROGRAM10_COMPLETE $(date -Is)"

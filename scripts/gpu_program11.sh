#!/bin/bash
# GPU program 11 — NEW-SERVER campaign, R12 stage 2 + R17 re-run + R6 + R7
# (PI: run the re-attacks anyway — new findings may emerge).
#   A. ADAPTIVE RE-ATTACKS: silent arm re-optimized against each non-trivial
#      monitor rung (multi8, learned) on all three targets, same budget
#      (15 x 100 steps), tagged dirs (silent_m8 / silent_lrn) — tests whether
#      suppression of the last-token reading TRANSFERS to the stronger rungs,
#      or whether some rung resists the adaptive attack.
#   B. Ladder RE-EVALUATION including the tagged runs (per-rung alarm rates on
#      adaptively-attacked prompts — the R12 stage-2 table).
#   C. Transfer matrix re-run.
#   D. R6 J-space-LAT defense (smol135dpo, 3 rounds with adversarial refresh)
#      + full attack suite against the defended checkpoint.
#   E. R7 causal steering (qwen15 pair: alpha sweeps at amp-peak + latent-only
#      layers, fixed-alpha layer profile).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
export HF_HOME=/proj/cloudfaas-PG0/hf-cache
exec > >(tee -a artifacts/gpu_program11.log) 2>&1
echo $$ > artifacts/pid_gpu_program11

echo "=== GPU_PROGRAM11_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. adaptive re-attacks (silent arm vs each rung) ----
for T in smol135dpo qwen05instr qwen15instr; do
  CHUNK=32; [ "$T" = "smol135dpo" ] && CHUNK=32 || CHUNK=16
  for M in multi8 learned; do
    TAG=m8; [ "$M" = "learned" ] && TAG=lrn
    $PY -m src.run_p3_gcg --model $T --arms silent --n-prompts 15 --n-steps 100 \
        --chunk $CHUNK --device cuda --monitor $M --run-tag $TAG || exit 1
  done
done

# ---- B. ladder re-evaluation (now includes silent_m8 / silent_lrn sets) ----
$PY -m src.run_p3_ladder --device cuda || exit 1

# ---- C. transfer matrix re-run ----
$PY -m src.run_p3_transfer --device cuda || exit 1

# ---- D. R6 defense + re-attack of the defended model ----
$PY -m src.run_p3_defense --target smol135dpo --rounds 3 --device cuda || exit 1
$PY -m src.run_p3_gcg --target-path artifacts/defense/smol135m_base_vanilla_lam1.0_defended_lam1.0/merged \
    --arms both --n-prompts 15 --n-steps 100 --device cuda --run-tag defended || exit 1

# ---- E. R7 steering (qwen15 pair) ----
$PY -m src.run_steering --pairs qwen15 --device cuda || exit 1

echo "GPU_PROGRAM11_COMPLETE $(date -Is)"

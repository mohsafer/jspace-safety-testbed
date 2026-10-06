#!/bin/bash
# Autonomous GPU research program (Tesla P100, 2026-09-24):
#   P3/H4 GCG evasion (smol135dpo -> qwen05instr -> qwen15instr)
#   -> P2 multi-seed on GPU (135M/360M, vanilla+jpen, 3 seeds)
#   -> fp16 parity (4 pairs, seed 10, GPU)
#   -> NF4 quantization axis (4 pairs + 1.5B, seed 10)
#   -> Qwen3-4B reduced JADR cross-check (fp16, 2 seeds)
#   -> aggregate + tables + figures + paper build (v2.0)
# All output tees to artifacts/gpu_program.log (artifact log, per PI request).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
exec > >(tee -a artifacts/gpu_program.log) 2>&1

echo "=== GPU_PROGRAM_START $(date -Is) ==="
nvidia-smi | head -12

# ---- 0. P3 smoke: mechanics check (1 prompt x 10 steps, both arms) ----
$PY -m src.run_p3_gcg --model smol135dpo --arms both --n-prompts 1 --n-steps 10 \
    --device cuda --force || exit 1

# ---- 1. P3/H4 full: smol135dpo (15 prompts x 2 arms x 100 steps) ----
$PY -m src.run_p3_gcg --model smol135dpo --arms both --n-prompts 15 --n-steps 100 --force \
    --device cuda || exit 1

# ---- 2. P2 multi-seed on GPU (H3 robustness; seed0 vanilla already done) ----
$PY -m src.run_dpo --model smol135 --variant jpen   --lam 1.0 --device cuda
$PY -m src.eval_dpo --model smol135 --variant jpen  --lam 1.0 --device cuda
for S in 1 2; do
  for V in vanilla jpen; do
    $PY -m src.run_dpo  --model smol135 --variant $V --lam 1.0 --seed $S --device cuda
    $PY -m src.eval_dpo --model smol135 --variant $V --lam 1.0 --seed $S --device cuda
  done
done
$PY -m src.eval_dpo --model smol135 --variant base --device cuda
for S in 0 1 2; do
  for V in vanilla jpen; do
    $PY -m src.run_dpo  --model smol360 --variant $V --lam 1.0 --seed $S --device cuda
    $PY -m src.eval_dpo --model smol360 --variant $V --lam 1.0 --seed $S --device cuda
  done
done
$PY -m src.eval_dpo --model smol360 --variant base --device cuda

# ---- 3. P3/H4: qwen05b_instruct (real RLHF'd target) ----
$PY -m src.run_p3_gcg --model qwen05instr --arms both --n-prompts 15 --n-steps 100 --force \
    --chunk 16 --device cuda || exit 1

# ---- 4. fp16 parity (device+precision variance axis; seed 10 = parity seed) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --dtype float16
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 11 \
    --device cuda   # fp32-cuda control at the same novel seeds

# ---- 5. NF4 quantization axis (JADR quant x accessibility) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --quant nf4

# ---- 6. P3/H4: qwen15b_instruct (paper's strongest pair; the headline attack) ----
$PY -m src.run_p3_gcg --model qwen15instr --arms both --n-prompts 15 --n-steps 100 --force \
    --chunk 16 --device cuda || exit 1

# ---- 7. Qwen3-4B reduced JADR cross-check (fp16 GPU; 4B fp32 > 12GB VRAM) ----
$PY -m src.run_p1 --pairs qwen34 --seeds 0 1 --n-calib 64 --device cuda --dtype float16

# ---- 8. regenerate aggregates + paper v2.0 ----
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures
(cd paper && bash build.sh)

echo "GPU_PROGRAM_COMPLETE $(date -Is)"

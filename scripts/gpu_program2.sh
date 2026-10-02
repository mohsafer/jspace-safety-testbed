#!/bin/bash
# GPU program 2 (after gpu_program.sh): completes everything program 1 could not
# run (jpen DPO device bug, qwen05 silent-arm vocab bug, parity/quant stages that
# never started), then P2 at scale for the Qwen pairs (fig7 rows), then
# aggregate + tables + figures. NO paper build: v2.0 is built once, after the
# manual consistency pass (PI: the next versioned paper is the complete one).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
exec > >(tee -a artifacts/gpu_program2.log) 2>&1
echo $$ > artifacts/pid_gpu_program2

echo "=== GPU_PROGRAM2_START $(date -Is) ==="
nvidia-smi | head -12

# ---- A. rerun failed jpen DPO (device bug fixed) + evals ----
for M in smol135 smol360; do
  for S in 0 1 2; do
    $PY -m src.run_dpo  --model $M --variant jpen --lam 1.0 --seed $S --device cuda
    $PY -m src.eval_dpo --model $M --variant jpen --lam 1.0 --seed $S --device cuda
  done
done

# ---- B. P3 qwen05 silent arm (vanilla resume-skips: 15 prompts exist) ----
$PY -m src.run_p3_gcg --model qwen05instr --arms both --n-prompts 15 --n-steps 100 \
    --chunk 16 --device cuda || exit 1

# ---- C. fp16 parity (seed 10) + fp32-cuda controls (seed 11) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --dtype float16
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 11 --device cuda

# ---- D. NF4 quantization axis (seed 10) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --quant nf4

# ---- E. P3 qwen15 (paper's strongest pair; the headline attack) ----
$PY -m src.run_p3_gcg --model qwen15instr --arms both --n-prompts 15 --n-steps 100 \
    --chunk 16 --device cuda || exit 1

# ---- F. Qwen3-4B reduced JADR cross-check (fp16 GPU; 4B fp32 > 12GB VRAM) ----
$PY -m src.run_p1 --pairs qwen34 --seeds 0 1 --n-calib 64 --device cuda --dtype float16

# ---- G. P2 at scale: Qwen DPO (fig7 rows; H3 at 0.5B/1.5B) ----
$PY -m src.run_dpo  --model qwen05 --variant vanilla --lam 1.0 --device cuda
$PY -m src.eval_dpo --model qwen05 --variant vanilla --device cuda
$PY -m src.run_dpo  --model qwen05 --variant jpen   --lam 1.0 --device cuda
$PY -m src.eval_dpo --model qwen05 --variant jpen   --device cuda
$PY -m src.eval_dpo --model qwen05 --variant base   --device cuda
$PY -m src.run_dpo  --model qwen15 --variant vanilla --lam 1.0 --device cuda
$PY -m src.eval_dpo --model qwen15 --variant vanilla --device cuda
$PY -m src.run_dpo  --model qwen15 --variant jpen   --lam 1.0 --device cuda
$PY -m src.eval_dpo --model qwen15 --variant jpen   --device cuda
$PY -m src.eval_dpo --model qwen15 --variant base   --device cuda

# ---- H. regenerate aggregates + tables + figures (fig7 gains Qwen rows) ----
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures

echo "GPU_PROGRAM2_COMPLETE $(date -Is)"

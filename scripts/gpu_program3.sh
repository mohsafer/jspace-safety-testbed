#!/bin/bash
# GPU program 3 (after gpu_program2.sh): final GPU stage before the paper build.
#   1. p3_posthoc: unify latent/G stats across targets + response-side monitor
#   2. fp32-GPU P1 at seed 10 (matched-seed control for the fp16/NF4 parity
#      table; seed 11 alone would confound device with seed)
#   3. regenerate aggregates/tables/figures with everything in
# NO paper build here either: v2.0 is built once after the manual consistency
# pass (PI instruction).
set -x
cd /users/mosafer/spacepriv8
PY=.venv/bin/python
exec > >(tee -a artifacts/gpu_program3.log) 2>&1
echo $$ > artifacts/pid_gpu_program3

echo "=== GPU_PROGRAM3_START $(date -Is) ==="

# ---- 0. best-effort extras from research/06-gpu-tests.md ----
# INT8 attempt: expected to fail on Pascal (int8 tensor cores need sm>=75);
# the log entry itself is the evidence for the paper's limitation claim.
$PY -m src.run_p1 --pairs smol135 --members instruct --seeds 10 --device cuda \
    --quant int8 || echo "INT8_FAILED_AS_EXPECTED_ON_PASCAL (see traceback above)"

# Abliterated-model recovery under NF4 (JADR's headline quantization result):
# download + fp16 and NF4 P1 runs on the abliterated Qwen2.5-1.5B-Instruct.
$PY - <<'EOF' || echo "ABLITERATED_DOWNLOAD_FAILED (name may not exist; skipped)"
from huggingface_hub import snapshot_download
for cand in ["huihui-ai/Qwen2.5-1.5B-Instruct-abliterated",
             "failfinder/SmolLM2-1.7B-instruct-abliterated"]:
    try:
        snapshot_download(cand, ignore_patterns=["*.gguf", "*.onnx*", "*consolidated*"])
        print("ABLITERATED_OK:", cand)
        break
    except Exception as e:
        print("download failed:", cand, e)
EOF
$PY - <<'EOF' || true
# register the abliterated checkpoint as a one-off P1 member if present
import json, os
from pathlib import Path
from huggingface_hub import scan_cache_dir
names = {r.repo_id for r in scan_cache_dir().repos}
for cand in ["huihui-ai/Qwen2.5-1.5B-Instruct-abliterated"]:
    if cand in names:
        import sys
        sys.path.insert(0, ".")
        from src.run_p1 import MODELS_P1
        MODELS_P1["abliter"] = [(cand, "qwen15b_abliterated", True)]
        from src.run_p1 import run_one
        run_one(cand, "qwen15b_abliterated", True, 10, "wikitext", 100, 0, 8,
                200, 100, force=False, device="cuda", dtype_name="float16")
        run_one(cand, "qwen15b_abliterated", True, 10, "wikitext", 100, 0, 8,
                200, 100, force=False, device="cuda", dtype_name="float16",
                quant="nf4")
        print("ABLITERATED_P1_DONE")
        break
EOF

# ---- 1. p3_posthoc: unify latent/G stats across targets + response-side monitor
$PY -m src.p3_posthoc --device cuda
# ---- 1b. rerun parity stages that crashed on mixed dtypes (fixed in jlens) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --dtype float16
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 \
    --device cuda --quant nf4

# ---- 2. fp32-GPU P1 at seed 10 (matched-seed control for fp16/NF4 parity) ----
$PY -m src.run_p1 --pairs smol135,smol360,qwen05,qwen15 --seeds 10 --device cuda

# ---- 3. uniform CPU-protocol evaluation of ALL DPO models (training happened
# on GPU; the paper's P1-eval protocol is fp32/CPU -- one measurement protocol
# for every P2 number). Existing CPU evals skip via the json guard. ----
for M in smol135 smol360; do
  for S in 0 1 2; do
    for V in vanilla jpen; do
      $PY -m src.eval_dpo --model $M --variant $V --lam 1.0 --seed $S --force
    done
  done
  $PY -m src.eval_dpo --model $M --variant base --force
done
for M in qwen05 qwen15; do
  for V in vanilla jpen; do
    $PY -m src.eval_dpo --model $M --variant $V --lam 1.0 --force
  done
  $PY -m src.eval_dpo --model $M --variant base --force
done

# ---- 4. regenerate aggregates/tables/figures with everything in ----
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures

echo "GPU_PROGRAM3_COMPLETE $(date -Is)"

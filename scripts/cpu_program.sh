#!/bin/bash
# Autonomous CPU research program: waits for P1 (pid $1), then:
#   ablations -> DPO@135M (vanilla+jpen) + evals -> build v(n+1)
#   -> DPO@360M + evals -> build v(n+2)
set -x
cd /users/mosafer/code/jspace
PY=.venv/bin/python

# ---- wait for the already-running P1 ----
while kill -0 "$1" 2>/dev/null; do sleep 60; done

$PY -m src.run_p1 --aggregate

# ---- H5 ablations (135M pair) ----
$PY -m src.run_p1 --pairs smol135 --members instruct --seeds 0 1 --calib generic
$PY -m src.run_p1 --pairs smol135 --members instruct --seeds 0 1 --calib safety
$PY -m src.run_p1 --pairs smol135 --members base    --seeds 0 1 --calib generic
$PY -m src.run_p1 --pairs smol135 --members base    --seeds 0 1 --calib safety
$PY -m src.run_p1 --pairs smol135 --seeds 0 --window 1            # wikitext seed 0
$PY -m src.run_p1 --pairs smol135 --seeds 3 4 --calib wikitext    # extra seeds
$PY -m src.run_p1 --aggregate

# ---- P2 @ 135M ----
$PY -m src.run_dpo --model smol135 --variant vanilla
$PY -m src.run_dpo --model smol135 --variant jpen --lam 1.0
$PY -m src.eval_dpo --model smol135 --variant base
$PY -m src.eval_dpo --model smol135 --variant vanilla
$PY -m src.eval_dpo --model smol135 --variant jpen

# ---- intermediate paper build (has everything except 360M DPO) ----
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures
(cd paper && bash build.sh)

# ---- P2 @ 360M ----
$PY -m src.run_dpo --model smol360 --variant vanilla
$PY -m src.run_dpo --model smol360 --variant jpen --lam 1.0
$PY -m src.eval_dpo --model smol360 --variant base
$PY -m src.eval_dpo --model smol360 --variant vanilla
$PY -m src.eval_dpo --model smol360 --variant jpen

# ---- final regeneration + build ----
$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures
(cd paper && bash build.sh)
echo "CPU_PROGRAM_COMPLETE"

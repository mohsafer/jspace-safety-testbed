#!/bin/bash
# DPO-only rerun (P1 + ablations already complete): train + eval both models,
# then regenerate tables/figures and build the paper.
set -x
cd /users/mosafer/code/jspace
PY=.venv/bin/python

$PY -m src.run_dpo --model smol135 --variant vanilla
$PY -m src.run_dpo --model smol135 --variant jpen --lam 1.0
$PY -m src.eval_dpo --model smol135 --variant base
$PY -m src.eval_dpo --model smol135 --variant vanilla
$PY -m src.eval_dpo --model smol135 --variant jpen

$PY -m src.run_dpo --model smol360 --variant vanilla
$PY -m src.run_dpo --model smol360 --variant jpen --lam 1.0
$PY -m src.eval_dpo --model smol360 --variant base
$PY -m src.eval_dpo --model smol360 --variant vanilla
$PY -m src.eval_dpo --model smol360 --variant jpen

$PY -m src.run_p1 --aggregate
$PY -m src.make_tables
$PY -m src.make_figures
(cd paper && bash build.sh)
echo "DPO_PROGRAM_COMPLETE"

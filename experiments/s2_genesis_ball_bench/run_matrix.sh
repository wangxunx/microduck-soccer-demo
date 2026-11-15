#!/bin/bash
# S2 matrix: one chain per idle GPU (HIP 0-3 = rocm-smi cards 3,2,0,1).
cd "$(dirname "$0")"
PY=../../../microduck-rl-genesis/.venv/bin/python
run() { gpu=$1; name=$2; shift 2; HIP_VISIBLE_DEVICES=$gpu $PY bench.py "$@" --out results/$name.json > results/$name.log 2>&1; }
( run 0 walk_4096 --model walk -B 4096; run 0 allcol_ball_1024 --model allcollisions --ball -B 1024 ) &
( run 1 allcol_4096 --model allcollisions -B 4096; run 1 allcol_ball_4096_pairs60 --model allcollisions --ball -B 4096 --max-pairs 60 ) &
( run 2 allcol_ball_4096 --model allcollisions --ball -B 4096 ) &
( run 3 allcol_ball_4096_noself --model allcollisions --ball -B 4096 --self-collision off ) &
wait

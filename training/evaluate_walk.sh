#!/bin/bash
# Evaluate a walking policy in the soccer game's MuJoCo runtime.
#   ./evaluate_walk.sh <exp-name> <ckpt>          # exports logs/<exp>/model_<ckpt>.pt first
#   ./evaluate_walk.sh --onnx <label> <file.onnx> # evaluate an existing ONNX (e.g. baselines)
#
# 1. S1 walking scenarios (14) + command sweep from standstill (19) + stop/turn scenarios (4) -> evals/walk/<id>/s1_*.json
# 2. full matches with our kicking policy (game evaluator, 11 scenarios) -> evals/walk/<id>/game_eval.json
#    FOOTBALL_ONNX=<file.onnx> selects another kicking policy (output dir gets a +<name> suffix)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/../../microduck-rl-genesis/.venv/bin/python"
S1="$HERE/../experiments/s1_sim2sim"
GAME="$HERE/../game"
FOOTBALL=${FOOTBALL_ONNX:-$HERE/exports/football-v4ft-v3-2998.onnx}   # kicking policy used in the matches

if [ "$1" = "--onnx" ]; then
  ID=$2; ONNX=$(realpath "$3")
else
  ID="$1-$2"; ONNX="$HERE/exports/$ID.onnx"
  (cd "$HERE" && "$PY" export_walk.py -e "$1" --ckpt "$2" -o "$ONNX" 2>&1 | grep -E "^wrote|Error")
fi
[ -n "${FOOTBALL_ONNX:-}" ] && ID="$ID+$(basename "$FOOTBALL" .onnx)"   # tag non-default kick policy
OUT="$HERE/evals/walk/$ID"; mkdir -p "$OUT"

echo "== S1 scenarios + sweep (game MuJoCo)"
(cd "$GAME" && node "$S1/run_mujoco.mjs" "$ID" "$ONNX" "$OUT/s1_mujoco.json" > "$OUT/s1_mujoco.log" &&
               S1_SCENARIO_FILE=probe_scenarios.json node "$S1/run_mujoco.mjs" "$ID" "$ONNX" "$OUT/s1_probe.json" > "$OUT/s1_probe.log" &&
               S1_SCENARIO_FILE=stop_scenarios.json node "$S1/run_mujoco.mjs" "$ID" "$ONNX" "$OUT/s1_stop.json" > "$OUT/s1_stop.log")

echo "== matches with our kicking policy (game evaluator)"
(cd "$GAME" && WALK_CANDIDATE="$ONNX" FOOTBALL_CANDIDATE="$FOOTBALL" FOOTBALL_CURRENT=1.75 FOOTBALL_WINDOW=1.2 \
   node scripts/evaluate-football.mjs "$OUT/game_eval.json" all > "$OUT/game_eval.log" 2>&1)
# the evaluator records the absolute game path; keep results free of local paths (public repo)
sed -i "s#$(cd "$HERE/../.." && pwd)/##g" "$OUT/game_eval.json"
echo "done: $OUT"

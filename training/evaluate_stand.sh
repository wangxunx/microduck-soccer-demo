#!/bin/bash
# Evaluate a get-up / standing policy in the soccer game's MuJoCo runtime.
#   ./evaluate_stand.sh <exp-name> <ckpt>          # exports logs/<exp>/model_<ckpt>.pt first
#   ./evaluate_stand.sh --onnx <label> <file.onnx> # evaluate an existing ONNX (e.g. the official one)
#
# 1. S6 get-up suite (60 cases), game default walk and walk-A after the recovery -> getup*.json
# 2. S4 kick cases (protocol C): post-kick handoff to this stand policy, with football-v4ft-v3
#    (official walk) and football-v5 (walk-A)                                       -> s4_*.json
# 3. full matches (game evaluator, 11 scenarios):
#      author set  = official walk + official football + this stand                -> game_eval_author.json
#      AMD set     = walk-A + football-v5 + this stand                              -> game_eval_amd.json
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/../../microduck-rl-genesis/.venv/bin/python"
GAME="$HERE/../game"
S4="$HERE/../experiments/s4_kick_sim2sim"
S6="$HERE/../experiments/s6_getup"
WALK_A="$HERE/exports/walk-A-5998.onnx"

if [ "$1" = "--onnx" ]; then
  ID=$2; ONNX=$(realpath "$3")
else
  ID="$1-$2"; ONNX="$HERE/exports/$ID.onnx"
  (cd "$HERE" && "$PY" export_walk.py -e "$1" --ckpt "$2" -o "$ONNX" 2>&1 | grep -E "^wrote|Error")
fi
OUT="$HERE/evals/stand/$ID"; mkdir -p "$OUT"
STRIP="s#$(cd "$HERE/../.." && pwd)/##g"   # keep results free of local paths (public repo)

echo "== 1. S6 get-up suite"
(cd "$GAME" && STAND_CANDIDATE="$ONNX" node "$S6/run_getup.mjs" "$OUT/getup.json" > "$OUT/getup.log" &&
               STAND_CANDIDATE="$ONNX" WALK_CANDIDATE="$WALK_A" node "$S6/run_getup.mjs" "$OUT/getup_walkA.json" > "$OUT/getup_walkA.log")

echo "== 2. S4 post-kick handoff"
(cd "$GAME" && STAND_CANDIDATE="$ONNX" FOOTBALL_CANDIDATE="$HERE/exports/football-v4ft-v3-2998.onnx" S4_PROTOCOLS=C_football \
                 node "$S4/run_mujoco.mjs" "$OUT/s4_v4ft.json" > "$OUT/s4_v4ft.log" &&
               STAND_CANDIDATE="$ONNX" WALK_CANDIDATE="$WALK_A" FOOTBALL_CANDIDATE="$HERE/exports/football-v5-4497.onnx" \
                 S4_PROTOCOLS=C_football node "$S4/run_mujoco.mjs" "$OUT/s4_v5_walkA.json" > "$OUT/s4_v5_walkA.log")

echo "== 3. matches"
(cd "$GAME" && STAND_CANDIDATE="$ONNX" node scripts/evaluate-football.mjs "$OUT/game_eval_author.json" all > "$OUT/game_eval_author.log" 2>&1 &&
               STAND_CANDIDATE="$ONNX" WALK_CANDIDATE="$WALK_A" FOOTBALL_CANDIDATE="$HERE/exports/football-v5-4497.onnx" \
                 FOOTBALL_CURRENT=1.75 FOOTBALL_WINDOW=1.2 node scripts/evaluate-football.mjs "$OUT/game_eval_amd.json" all > "$OUT/game_eval_amd.log" 2>&1)
for f in "$OUT"/*.json; do sed -i "$STRIP" "$f"; done
"$PY" "$HERE/summarize_stand_eval.py" "$OUT"

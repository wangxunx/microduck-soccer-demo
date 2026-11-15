#!/bin/bash
# Export a football checkpoint and evaluate it end to end.
#   ./evaluate_candidate.sh <exp-name> <ckpt-iteration> [hip-gpu=1]
#
#  1. export ONNX + game manifest            -> exports/<exp>-<ckpt>.{onnx,json}
#  2. S4 kick cases, game MuJoCo (protocol C) -> evals/<exp>-<ckpt>/s4_mujoco.json
#  3. S4 kick cases, Genesis replica          -> evals/<exp>-<ckpt>/s4_genesis.json
#  4. paired S4 report                        -> evals/<exp>-<ckpt>/s4_report.md
#  5. the game's own evaluator (all suites, real matches with AI)
#                                             -> evals/<exp>-<ckpt>/game_eval.json
#     FOOTBALL_CURRENT=1.75 is REQUIRED: without it evaluate-football.mjs runs a
#     candidate with no current limit, which is not the deployment contract.
set -euo pipefail
EXP=$1; CKPT=$2; GPU=${3:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
PY="$HERE/../../microduck-rl-genesis/.venv/bin/python"
S4="$HERE/../experiments/s4_kick_sim2sim"
GAME="$HERE/../game"
ID="$EXP-$CKPT"; OUT="$HERE/evals/$ID"; ONNX="$HERE/exports/$ID.onnx"
mkdir -p "$OUT"

echo "== 1. export"
(cd "$HERE" && "$PY" export_football.py -e "$EXP" --ckpt "$CKPT" -o "$ONNX" 2>&1 | grep -E "^wrote|Error")

echo "== 2. S4 MuJoCo (game path)"
(cd "$GAME" && FOOTBALL_CANDIDATE="$ONNX" S4_PROTOCOLS=C_football node "$S4/run_mujoco.mjs" "$OUT/s4_mujoco.json" > "$OUT/s4_mujoco.log")

echo "== 3. S4 Genesis replica (HIP $GPU)"
(cd "$S4" && HIP_VISIBLE_DEVICES=$GPU "$PY" run_genesis.py --football "$ONNX" --protocols C_football \
   --out "$OUT/s4_genesis.json" 2>&1 | grep -v watertight | tail -3)

echo "== 4. S4 report"
(cd "$S4" && "$PY" analyze.py --mujoco "$OUT/s4_mujoco.json" --genesis "$OUT/s4_genesis.json" \
   --out "$OUT/s4_report.md" > /dev/null) && sed -n '3,12p' "$OUT/s4_report.md"

echo "== 5. game evaluator (all suites)"
(cd "$GAME" && FOOTBALL_CANDIDATE="$ONNX" FOOTBALL_CURRENT=1.75 FOOTBALL_WINDOW=1.2 \
   node scripts/evaluate-football.mjs "$OUT/game_eval.json" all > "$OUT/game_eval.log" 2>&1)
"$PY" "$HERE/summarize_game_eval.py" "$OUT/game_eval.json" "$HERE/evals/baseline-author/game_eval.json"

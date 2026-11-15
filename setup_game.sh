#!/bin/bash
# Recreate game/: the upstream MicroDuck Robot Soccer browser game, pinned, plus our
# model-switch patch and the AMD/Genesis-trained policies.
#
#   ./setup_game.sh                 # -> ./game
#   GAME_DIR=/tmp/game ./setup_game.sh
#
# The game itself is NOT vendored in this repository: upstream
# (https://github.com/wander521/microduck-robot-soccer) has no license file, so we
# only keep our own changes (game-patches/) and fetch the original at a fixed commit.
#
# Requires: git, Node.js 22+ / npm, and a microduck-rl-genesis checkout for the
# Genesis-trained walking policies (default: ../microduck-rl-genesis,
# override with MICRODUCK_GENESIS_REPO).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
GAME_DIR=${GAME_DIR:-$HERE/game}
UPSTREAM=https://github.com/wander521/microduck-robot-soccer.git
COMMIT=b0aee3d4c12d8c3969e4c4cfcb2a214477429cf1
GENESIS_REPO=${MICRODUCK_GENESIS_REPO:-$HERE/../microduck-rl-genesis}
FOOTBALL=football-v4ft-v3-2998

if [ -e "$GAME_DIR" ]; then
  echo "error: $GAME_DIR already exists (remove it or set GAME_DIR)" >&2
  exit 1
fi

echo "== clone upstream game @ ${COMMIT:0:7}"
git clone -q "$UPSTREAM" "$GAME_DIR"
git -C "$GAME_DIR" checkout -q "$COMMIT"

echo "== apply game-patches/"
for p in "$HERE"/game-patches/*.patch; do
  git -C "$GAME_DIR" apply "$p"
  echo "   applied $(basename "$p")"
done

echo "== copy Genesis-trained policies"
DST="$GAME_DIR/public/assets/policies/genesis"
mkdir -p "$DST"
for f in velocity velocity-reproduced rough backlash; do   # walking (?walk=genesis*)
  cp "$GENESIS_REPO/policies/$f.onnx" "$DST/"
done
cp "$HERE/training/exports/$FOOTBALL.onnx" "$HERE/training/exports/$FOOTBALL.json" "$DST/"   # kicking (?football=genesis)

echo "== npm ci"
(cd "$GAME_DIR" && npm ci --silent)

cat <<EOF

Done. Run the game:
  cd $GAME_DIR && npm run dev
  open http://localhost:5173/?lang=en&walk=genesis&football=genesis
EOF

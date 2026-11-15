"""Soccer RL tasks for the Microduck on Genesis (AMD ROCm).

Imports the `microduck` package (BAM actuator, velocity env, constants) from a
microduck-rl-genesis checkout, which is used read-only. Override the location
with MICRODUCK_GENESIS_REPO.
"""
import os
import sys
from pathlib import Path

GENESIS_REPO = Path(os.environ.get(
    "MICRODUCK_GENESIS_REPO",
    Path(__file__).resolve().parents[3] / "microduck-rl-genesis",
)).resolve()
if not (GENESIS_REPO / "microduck" / "velocity_env.py").is_file():
    raise ImportError(f"microduck-rl-genesis not found at {GENESIS_REPO} (set MICRODUCK_GENESIS_REPO)")
if str(GENESIS_REPO) not in sys.path:
    sys.path.insert(0, str(GENESIS_REPO))

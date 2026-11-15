"""Generate the shared S1 command timelines (identical for MuJoCo and Genesis).

Each scenario is a list of segments [t_start, vx, wz] (vy is always 0, as in
the game). Commands are body-frame velocity targets in m/s and rad/s.
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DT = 0.02  # 50 Hz control

fixed = {
    # name: (vx, wz, duration)
    "stand": (0.0, 0.0, 20.0),
    "fwd_0.15": (0.15, 0.0, 20.0),
    "fwd_0.30": (0.30, 0.0, 20.0),
    "back_0.30": (-0.30, 0.0, 20.0),
    "turn_+1.0": (0.0, 1.0, 20.0),
    "turn_-1.0": (0.0, -1.0, 20.0),
    # Game max turn rate; outside the Genesis training range (|wz| <= 1.0).
    "turn_+1.5": (0.0, 1.5, 20.0),
    "arc_0.2_0.5": (0.2, 0.5, 20.0),
}

scenarios = {}
for name, (vx, wz, dur) in fixed.items():
    scenarios[name] = {"duration": dur, "segments": [[0.0, vx, wz]], "kind": "fixed"}

# Game-like random command sequences: joystick-style levels, 1-3 s holds,
# turn rate kept inside the Genesis training range so the comparison is fair.
FWD_LEVELS = [-1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.0]
TURN_LEVELS = [-1.0, -0.5, 0.0, 0.0, 0.0, 0.5, 1.0]
for seed in range(6):
    rng = random.Random(1000 + seed)
    t, segs = 0.0, []
    while t < 60.0:
        vx = 0.3 * rng.choice(FWD_LEVELS)
        wz = 1.0 * rng.choice(TURN_LEVELS)
        segs.append([round(t, 2), vx, wz])
        t += round(rng.uniform(1.0, 3.0) / DT) * DT
    scenarios[f"random_{seed}"] = {"duration": 60.0, "segments": segs, "kind": "random"}

out = HERE / "scenarios.json"
out.write_text(json.dumps({"dt": DT, "scenarios": scenarios}, indent=1))
print(f"wrote {out} ({len(scenarios)} scenarios)")

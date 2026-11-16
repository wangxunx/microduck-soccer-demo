"""Compare walking policies evaluated in the game's MuJoCo runtime (S1 harness output).

    python summarize_walk_eval.py label=evals/walk/<id> [label=...]

Each directory holds s1_mujoco.json (S1 scenarios) and s1_probe.json (command sweep from
standstill). Metrics reuse experiments/s1_sim2sim/analyze.py.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "s1_sim2sim"))
import analyze as s1  # noqa: E402

SCEN = json.loads((HERE.parent / "experiments/s1_sim2sim/scenarios.json").read_text())["scenarios"]
PROBE = json.loads((HERE.parent / "experiments/s1_sim2sim/probe_scenarios.json").read_text())["scenarios"]


def last5(run):
    t = run["traj"]
    x, y, yaw = np.array(t["x"]), np.array(t["y"]), np.unwrap(t["yaw"])
    i = len(x) - 250
    c, s = math.cos(yaw[i]), math.sin(yaw[i])
    return (c * (x[-1] - x[i]) + s * (y[-1] - y[i])) / 5, (yaw[-1] - yaw[i]) / 5


def main():
    rows = {}
    for arg in sys.argv[1:]:
        label, d = arg.split("=", 1)
        d = Path(d)
        main_runs = json.loads((d / "s1_mujoco.json").read_text())["runs"]
        probe_runs = json.loads((d / "s1_probe.json").read_text())["runs"]
        m = {s: s1.metrics(s, SCEN[s], main_runs[s]) for s in SCEN}
        rnd = [s for s in SCEN if SCEN[s]["kind"] == "random"]
        rows[label] = {
            "falls": sum(v["falls"] for v in m.values()) + sum(r["falls"] for r in probe_runs.values()),
            "fwd_0.30": m["fwd_0.30"]["vx"], "fwd_0.15": m["fwd_0.15"]["vx"], "back_0.30": m["back_0.30"]["vx"],
            "turn_1.0": m["turn_+1.0"]["wz"], "turn_1.5": m["turn_+1.5"]["wz"],
            "rmse_vx": float(np.mean([m[s]["vx_rmse"] for s in rnd])), "rmse_wz": float(np.mean([m[s]["wz_rmse"] for s in rnd])),
            "stand_drift_cm": 100 * m["stand"]["drift_m"],
            "probe": {s: last5(r) for s, r in probe_runs.items()},
        }
    labels = list(rows)
    print("## Walking in the game's MuJoCo runtime\n")
    print("| metric | command | " + " | ".join(labels) + " |")
    print("|---|---|" + "---:|" * len(labels))
    for key, cmd, fmt in [("fwd_0.15", "vx 0.15", "{:.3f}"), ("fwd_0.30", "vx 0.30", "{:.3f}"), ("back_0.30", "vx -0.30", "{:.3f}"),
                          ("turn_1.0", "wz 1.0", "{:.2f}"), ("turn_1.5", "wz 1.5", "{:.2f}"),
                          ("rmse_vx", "random seq.", "{:.3f}"), ("rmse_wz", "random seq.", "{:.3f}"),
                          ("stand_drift_cm", "stand 20 s", "{:.1f}"), ("falls", "all 33 runs", "{}")]:
        print(f"| {key} | {cmd} | " + " | ".join(fmt.format(rows[l][key]) for l in labels) + " |")
    print("\n## From standstill (last 5 s of 10 s): measured vx (m/s) / wz (rad/s)\n")
    print("| command | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(labels))
    for s in PROBE:
        cells = []
        for l in labels:
            vx, wz = rows[l]["probe"][s]
            cells.append(f"{vx:+.3f} / {wz:+.2f}")
        print(f"| {s} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

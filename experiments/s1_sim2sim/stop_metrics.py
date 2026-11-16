"""Stopping / in-place-turn metrics from run_mujoco.mjs output on stop_scenarios.json.

    python stop_metrics.py label=file.json [...]
  stop_dist   distance travelled in the 3 s after a 0.3 m/s -> 0 command (go_stop)
  stop_time   time until body speed (0.2 s window) stays below 0.03 m/s
  turn_drift  translation per second while turning in place at 0.54 rad/s (turn_place, t >= 1 s)
  turn_rate   measured yaw rate in turn_place
  sgs_dist    distance travelled during the three 0.5 s 'stop' windows of stop_go_stop
"""
import json, math, sys
import numpy as np

def xy(r):
    t = r["traj"]; return np.array(t["x"]), np.array(t["y"]), np.unwrap(t["yaw"])

def stop_metrics(runs):
    """runs: run_mujoco.mjs output 'runs' for stop_scenarios.json -> dict of metrics (SI units)."""
    x, y, _ = xy(runs["go_stop"]); i0 = 150
    d = float(np.hypot(np.diff(x[i0:]), np.diff(y[i0:])).sum())
    v = np.hypot(x[10:] - x[:-10], y[10:] - y[:-10]) / 0.2
    stop_t = None
    for k in np.nonzero(v[i0:] < 0.03)[0]:
        if np.all(v[i0 + k:i0 + k + 25] < 0.03):
            stop_t = float(k * 0.02); break
    x2, y2, yaw2 = xy(runs["turn_place"]); a = 50
    drift = math.hypot(x2[-1] - x2[a], y2[-1] - y2[a]) / ((len(x2) - a) * 0.02)
    rate = (yaw2[-1] - yaw2[a]) / ((len(x2) - a) * 0.02)
    x3, y3, _ = xy(runs["stop_go_stop"]); sgs = 0.0
    for s0 in (50, 125, 200):
        sgs += float(np.hypot(np.diff(x3[s0:s0 + 25]), np.diff(y3[s0:s0 + 25])).sum())
    return {"stop_dist": d, "stop_time": stop_t, "turn_drift": float(drift), "turn_rate": float(rate), "sgs_dist": sgs}


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        m = stop_metrics(json.loads(open(path).read())["runs"])
        st = f"{m['stop_time']:.2f} s" if m["stop_time"] is not None else "  >3 s"
        print(f"{label:14s} stop_dist {m['stop_dist']*100:5.1f} cm  stop_time {st}  turn_drift {m['turn_drift']*100:4.1f} cm/s  "
              f"turn_rate {m['turn_rate']:.2f} rad/s  sgs_dist {m['sgs_dist']*100:5.1f} cm")

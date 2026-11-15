"""S1 analysis: same metric code for both engines.

    python3 analyze.py            # reads results/*.json, writes results/report.md

Metrics (frames flagged `recovering`, plus 1 s after, are excluded from
tracking/tilt metrics; falls are counted separately):
  falls        number of falls (game criterion: tilted/low for > 0.28 s)
  vx, wz       mean measured body-frame forward speed / yaw rate, t >= 3 s
               (fixed-command scenarios)
  vx_rmse,     per-segment tracking RMSE over random sequences: mean measured
  wz_rmse      velocity over the 2nd half of each command segment vs command
  drift        stand scenario: final xy displacement (m) and |yaw| change (deg)
  tilt_max     max trunk tilt in degrees
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
W = 25  # 0.5 s velocity window at 50 Hz


def load():
    runs = {}  # (engine_label, policy) -> {scenario: run}
    for f in sorted(RES.glob("mujoco_*.json")):
        d = json.loads(f.read_text())
        runs[("mujoco(game)", d["policy"])] = d["runs"]
    for f in sorted(RES.glob("genesis_*.json")):
        d = json.loads(f.read_text())
        label = f"genesis[{d['actuator']}]"
        for p, pd in d["policies"].items():
            runs[(label, p)] = pd["runs"]
    return runs


def body_vel(tr):
    x, y = np.asarray(tr["x"]), np.asarray(tr["y"])
    yaw = np.unwrap(np.asarray(tr["yaw"]))
    n = len(x)
    vx = np.full(n, np.nan)
    wz = np.full(n, np.nan)
    h = W // 2
    for i in range(h, n - h):
        a, b = i - h, i + h
        dx, dy = x[b] - x[a], y[b] - y[a]
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        T = (b - a) * 0.02
        vx[i] = (c * dx + s * dy) / T
        wz[i] = (yaw[b] - yaw[a]) / T
    return vx, wz


def valid_mask(tr):
    rec = np.asarray(tr["recovering"], dtype=bool)
    ok = ~rec
    # exclude 1 s after each recovery and the 0.3 s before it (pre-detection)
    idx = np.nonzero(rec)[0]
    for i in idx:
        ok[max(0, i - 15): i + 51] = False
    return ok


def metrics(scn_name, sc, run):
    tr = run["traj"]
    t = np.asarray(tr["t"])
    vx, wz = body_vel(tr)
    ok = valid_mask(tr) & np.isfinite(vx)
    gz = np.asarray(tr["gz"])
    tilt = np.degrees(np.arccos(np.clip(-gz, -1, 1)))
    m = {"falls": run["falls"],
         "tilt_max": float(np.max(tilt[valid_mask(tr)])) if valid_mask(tr).any() else float("nan")}
    if sc["kind"] == "fixed":
        sel = ok & (t >= 3.0)
        m["vx"] = float(np.mean(vx[sel])) if sel.any() else float("nan")
        m["wz"] = float(np.mean(wz[sel])) if sel.any() else float("nan")
        m["vx_cmd"], m["wz_cmd"] = sc["segments"][0][1], sc["segments"][0][2]
        if scn_name == "stand":
            m["drift_m"] = float(math.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]))
            m["yaw_drift_deg"] = float(abs(np.degrees(np.unwrap(tr["yaw"])[-1] - tr["yaw"][0])))
    else:
        segs = sc["segments"] + [[sc["duration"], 0, 0]]
        ex, ew = [], []
        for (t0, cvx, cwz), (t1, _, _) in zip(segs[:-1], segs[1:]):
            sel = ok & (t >= (t0 + t1) / 2) & (t < t1)
            if sel.sum() < 10:
                continue
            ex.append(np.mean(vx[sel]) - cvx)
            ew.append(np.mean(wz[sel]) - cwz)
        m["vx_rmse"] = float(np.sqrt(np.mean(np.square(ex)))) if ex else float("nan")
        m["wz_rmse"] = float(np.sqrt(np.mean(np.square(ew)))) if ew else float("nan")
        m["segments_scored"] = len(ex)
    return m


def fmt(v, nd=3):
    if isinstance(v, (int, np.integer)):
        return str(v)
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def main():
    scen = json.loads((HERE / "scenarios.json").read_text())["scenarios"]
    runs = load()
    allm = {k: {s: metrics(s, scen[s], r[s]) for s in scen if s in r} for k, r in runs.items()}
    (RES / "metrics.json").write_text(json.dumps({f"{e}|{p}": v for (e, p), v in allm.items()}, indent=1))

    engines = sorted({e for e, _ in allm}, key=lambda e: (not e.startswith("mujoco"), e))
    policies = sorted({p for _, p in allm}, key=lambda p: (p != "official_walk", p))
    L = ["# S1 sim2sim results\n"]

    # 1. falls
    L.append("## Falls per scenario (total over all scenarios)\n")
    L.append("| policy | " + " | ".join(engines) + " |")
    L.append("|---|" + "---:|" * len(engines))
    for p in policies:
        row = []
        for e in engines:
            m = allm.get((e, p))
            if not m:
                row.append("–"); continue
            tot = sum(v["falls"] for v in m.values())
            bad = [s for s, v in m.items() if v["falls"]]
            row.append(f"{tot}" + (f" ({', '.join(bad)})" if bad else ""))
        L.append(f"| {p} | " + " | ".join(row) + " |")

    # 2. fixed command tracking
    fixed = [s for s, v in scen.items() if v["kind"] == "fixed" and s != "stand"]
    L.append("\n## Fixed-command tracking: measured vx (m/s) / wz (rad/s), t ≥ 3 s\n")
    L.append("| scenario | cmd | policy | " + " | ".join(engines) + " |")
    L.append("|---|---|---|" + "---|" * len(engines))
    for s in fixed:
        c = scen[s]["segments"][0]
        for p in policies:
            row = []
            for e in engines:
                m = allm.get((e, p), {}).get(s)
                row.append("–" if not m else f"{fmt(m['vx'],2)} / {fmt(m['wz'],2)}" + (f" ⚠{m['falls']}F" if m["falls"] else ""))
            L.append(f"| {s} | {c[1]:.2f} / {c[2]:.1f} | {p} | " + " | ".join(row) + " |")

    # 3. stand drift
    L.append("\n## Standing still 20 s: drift (m) / yaw drift (deg) / max tilt (deg)\n")
    L.append("| policy | " + " | ".join(engines) + " |")
    L.append("|---|" + "---|" * len(engines))
    for p in policies:
        row = []
        for e in engines:
            m = allm.get((e, p), {}).get("stand")
            row.append("–" if not m else f"{fmt(m['drift_m'])} / {fmt(m['yaw_drift_deg'],1)} / {fmt(m['tilt_max'],1)}")
        L.append(f"| {p} | " + " | ".join(row) + " |")

    # 4. random sequences
    rnd = [s for s, v in scen.items() if v["kind"] == "random"]
    L.append(f"\n## Game-like random command sequences ({len(rnd)} × 60 s): mean vx RMSE / wz RMSE / falls / max tilt\n")
    L.append("| policy | " + " | ".join(engines) + " |")
    L.append("|---|" + "---|" * len(engines))
    for p in policies:
        row = []
        for e in engines:
            ms = [allm.get((e, p), {}).get(s) for s in rnd]
            ms = [m for m in ms if m]
            if not ms:
                row.append("–"); continue
            row.append(f"{fmt(np.nanmean([m['vx_rmse'] for m in ms]))} / "
                       f"{fmt(np.nanmean([m['wz_rmse'] for m in ms]))} / "
                       f"{sum(m['falls'] for m in ms)} / {fmt(max(m['tilt_max'] for m in ms),1)}")
        L.append(f"| {p} | " + " | ".join(row) + " |")

    out = RES / "report.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", out)


if __name__ == "__main__":
    main()

"""S4 analysis: paired per-case comparison MuJoCo(game) vs Genesis variants.

    ../../../microduck-rl-genesis/.venv/bin/python analyze.py   # -> results/report.md
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DT = 0.02


def metrics(r):
    tr = r["traj"]
    b, v = np.asarray(tr["ball"]), np.asarray(tr["bvel"])
    sp = np.linalg.norm(v[:, :2], axis=1)
    disp = b[:, :2] - np.array([r["bx"], r["by"]])
    moved_idx = np.nonzero(np.linalg.norm(disp, axis=1) > 0.005)[0]
    m = {"falls": r["falls"],
         "tilt_max_deg": float(np.degrees(np.arccos(np.clip(-np.asarray(tr["gz"]), -1, 1))).max()),
         "robot_drift_m": float(np.linalg.norm(np.asarray(tr["base"])[-1, :2])),
         "hit": bool(np.linalg.norm(disp[-1]) > 0.02)}
    pol = tr["policy"]
    tilt = np.degrees(np.arccos(np.clip(-np.asarray(tr["gz"]), -1, 1)))
    if "football" in pol and "stand" in pol:
        h = pol.index("stand")                      # handoff step (football -> stand)
        m["handoff_tilt_max_deg"] = float(tilt[h:h + 30].max())   # 0.6 s after the handoff
        m["window_tilt_max_deg"] = float(tilt[:h].max())
    if m["hit"]:
        i0 = moved_idx[0]
        m["t_first_move"] = (i0 + 1) * DT
        m["peak_speed"] = float(sp.max())
        j = min(len(disp) - 1, i0 + 25)  # direction from the first 0.5 s of travel
        m["dir_deg"] = float(math.degrees(math.atan2(disp[j, 1], disp[j, 0])))
        m["dist_end_m"] = float(np.linalg.norm(disp[-1]))
    return m


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mujoco", type=Path, default=RES / "mujoco.json")
    ap.add_argument("--genesis", type=Path, default=RES / "genesis.json")
    ap.add_argument("--out", type=Path, default=RES / "report.md")
    args = ap.parse_args()
    mj = json.loads(args.mujoco.read_text())["runs"]
    gen = json.loads(args.genesis.read_text())["variants"]
    M = {"mujoco(game)": {k: metrics(r) for k, r in mj.items()}}
    for v, runs in gen.items():
        M[v] = {k: metrics(r) for k, r in runs.items()}
    args.out.with_suffix(".metrics.json").write_text(json.dumps(M, indent=1))
    engines = list(M)
    protos = tuple(p for p in ("A_game_kick", "B_full_kick", "C_football") if any(k.startswith(p) for k in M["mujoco(game)"]))
    L = ["# S4 kick sim2sim: MuJoCo(game) vs Genesis\n"]

    L.append("## Per protocol summary (18 cases each = 2 feet x 9 ball positions)\n")
    L.append("| protocol | engine | falls | hits | median peak ball speed (m/s) | median dir (deg) | median dist 4 s (m) | max tilt (deg) |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for p in protos:
        for e in engines:
            ms = [m for k, m in M[e].items() if k.startswith(p)]
            hits = [m for m in ms if m["hit"]]
            med = lambda key: f"{np.median([m[key] for m in hits]):.2f}" if hits else "–"
            L.append(f"| {p} | {e} | {sum(m['falls'] for m in ms)} | {len(hits)}/{len(ms)} | {med('peak_speed')} | "
                     f"{med('dir_deg')} | {med('dist_end_m')} | {max(m['tilt_max_deg'] for m in ms):.1f} |")

    if any("handoff_tilt_max_deg" in m for m in M["mujoco(game)"].values()):
        L.append("\n## Football handoff: max tilt in the window / within 0.6 s after handing over to `stand` (deg)\n")
        L.append("| engine | window median | window max | after-handoff median | after-handoff p90 | after-handoff max | cases > 30 deg |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for e in engines:
            ms = [m for m in M[e].values() if "handoff_tilt_max_deg" in m]
            w = np.array([m["window_tilt_max_deg"] for m in ms]); a = np.array([m["handoff_tilt_max_deg"] for m in ms])
            L.append(f"| {e} | {np.median(w):.1f} | {w.max():.1f} | {np.median(a):.1f} | {np.percentile(a, 90):.1f} | {a.max():.1f} | {(a > 30).sum()}/{len(a)} |")
    L.append("\n## Paired comparison vs MuJoCo(game)\n")
    L.append("| protocol | engine | hit/miss agreement | peak-speed ratio gen/mj (median [min, max], both hit) | |Δdir| deg (median / max) | Δt first move s (median) |")
    L.append("|---|---|---:|---|---|---:|")
    for p in protos:
        for e in engines[1:]:
            keys = [k for k in M["mujoco(game)"] if k.startswith(p)]
            agree = sum(M[e][k]["hit"] == M["mujoco(game)"][k]["hit"] for k in keys)
            both = [k for k in keys if M[e][k]["hit"] and M["mujoco(game)"][k]["hit"]]
            if both:
                ratio = np.array([M[e][k]["peak_speed"] / M["mujoco(game)"][k]["peak_speed"] for k in both])
                ddir = np.array([abs((M[e][k]["dir_deg"] - M["mujoco(game)"][k]["dir_deg"] + 180) % 360 - 180) for k in both])
                dt = np.array([M[e][k]["t_first_move"] - M["mujoco(game)"][k]["t_first_move"] for k in both])
                L.append(f"| {p} | {e} | {agree}/{len(keys)} | {np.median(ratio):.2f} [{ratio.min():.2f}, {ratio.max():.2f}] | "
                         f"{np.median(ddir):.1f} / {ddir.max():.1f} | {np.median(dt):+.2f} |")
            else:
                L.append(f"| {p} | {e} | {agree}/{len(keys)} | – | – | – |")

    L.append("\n## Per case (peak ball speed m/s @ dir deg; F = fall; · = ball not hit)\n")
    L.append("| case | " + " | ".join(engines) + " |")
    L.append("|---|" + "---|" * len(engines))
    for k in M["mujoco(game)"]:
        row = []
        for e in engines:
            m = M[e][k]
            s = f"{m['peak_speed']:.2f} @ {m['dir_deg']:+.0f}" if m["hit"] else "·"
            row.append(s + (" F" if m["falls"] else ""))
        L.append(f"| {k} | " + " | ".join(row) + " |")
    args.out.write_text("\n".join(L) + "\n")
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()

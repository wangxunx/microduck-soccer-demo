"""S3 analysis: ball metrics per scenario for mujoco_python / genesis_native / genesis_rollfric.

    ../../../microduck-rl-genesis/.venv/bin/python analyze.py   # -> results/report.md
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DT = 0.005


def hspeed(r):
    return np.linalg.norm(np.asarray(r["v"])[:, :2], axis=1)


def metrics(name, r):
    p, v = np.asarray(r["p"]), np.asarray(r["v"])
    s = hspeed(r)
    t = (np.arange(len(s)) + 1) * DT
    at = lambda sec: float(s[min(len(s) - 1, round(sec / DT) - 1)])
    m = {}
    if name.startswith(("roll", "slide")):
        m["v@0.5s"], m["v@1s"], m["v@2s"], m["v@4s"] = at(0.5), at(1), at(2), at(4)
        stop = np.nonzero(s < 0.02)[0]
        m["stop_s"] = float(t[stop[0]]) if len(stop) else None
        m["dist_8s_m"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
        w = np.asarray(r["w"])
        R = 0.05
        slip = np.abs(np.linalg.norm(w[:, :2], axis=1) * R - s)
        m["slip@0.5s"] = float(slip[round(0.5 / DT) - 1])
    elif name.startswith("drop"):
        z = p[:, 2]
        first = np.argmax(z < 0.0505)
        m["bounce_apex_mm"] = float((z[first:].max() - 0.05) * 1000)
        m["settle_z_mm"] = float(z[-1] * 1000)
    elif name.startswith("wall"):
        vx = v[:, 0]
        hit = np.argmax(vx < 0) if (vx < 0).any() else None
        m["v_in"] = float(vx[max(0, (hit or 1) - 5)])
        m["v_out"] = float(-vx[hit:hit + 40].min()) if hit else 0.0
        m["restitution"] = m["v_out"] / m["v_in"] if hit else 0.0
    elif name.startswith("kick"):
        m["ball_peak_v"] = float(s[: round(0.5 / DT)].max())
        m["ball_v@1s"], m["ball_v@2s"] = at(1), at(2)
        m["ball_dist_3s_m"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
        m["striker_v_after"] = float(np.asarray(r["sv"])[round(0.3 / DT)])
        m["ball_lift_mm"] = float((p[:, 2].max() - 0.05) * 1000)
    return m


def main():
    variants = {}
    for f in ("mujoco.json", "genesis.json"):
        variants |= json.loads((RES / f).read_text())["variants"]
    scen = json.loads((RES / "mujoco.json").read_text())["scenarios"]
    allm = {v: {s: metrics(s, d["runs"][s]) for s in scen} for v, d in variants.items()}
    (RES / "metrics.json").write_text(json.dumps(allm, indent=1))
    order = ["mujoco_python", "genesis_native", "genesis_rollfric"]
    L = ["# S3 ball dynamics: MuJoCo (game reference) vs Genesis\n",
         "MuJoCo reference = game engine (Python 3.16 and game WASM 3.11 bit-identical, see crosscheck_wasm.json).\n"]
    for s in scen:
        keys = list(allm["mujoco_python"][s])
        L.append(f"\n### {s}\n")
        L.append("| metric | " + " | ".join(order) + " |")
        L.append("|---|" + "---:|" * len(order))
        for k in keys:
            row = []
            for v in order:
                x = allm[v][s][k]
                row.append("–" if x is None else f"{x:.3f}")
            L.append(f"| {k} | " + " | ".join(row) + " |")
    (RES / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

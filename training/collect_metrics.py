"""Recompute the standard metrics used in MODELS.md from the evaluation folders.

    ../../microduck-rl-genesis/.venv/bin/python collect_metrics.py            # markdown tables
    ../../microduck-rl-genesis/.venv/bin/python collect_metrics.py --json     # machine-readable

Kick policies:   evals/<id>/game_eval.json (+ .frames.json) and evals/<id>/s4_report.metrics.json
Walk policies:   evals/walk/<id>/{s1_mujoco,s1_probe,game_eval}.json
Missing raw files (frames, S1 trajectories are git-ignored) are reported as "–".

Metric definitions (keep stable so versions stay comparable):
  goals / kicks / falls / resets   game evaluator totals over its 11 scenarios
  kick->goal                       share of kicks followed by a goal within 5 s
  kick falls                       falls starting within 1.6 s after a kick/handoff phase
  S4 handoff tilt                  median max tilt (deg) in the 0.6 s after handing over to `stand`
                                   (18 S4 cases, game MuJoCo)
  S4 ball speed / dir              median peak ball speed (m/s) / direction (deg), game MuJoCo
  walk vx@0.3, wz@1.0              measured body speed for a fixed command (t >= 3 s), game MuJoCo
  walk rmse vx / wz                tracking error over 6 game-like random command sequences
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVALS = HERE / "evals"


def match_metrics(game_eval: Path):
    if not game_eval.exists():
        return None
    r = json.loads(game_eval.read_text())["results"]
    m = {"goals": sum(sum(x["goals"].values()) for x in r), "kicks": sum(x["kickAttempts"] for x in r),
         "falls": sum(x["recoveryEntries"] for x in r), "resets": sum(x["forcedResets"] for x in r),
         "kick_to_goal": None, "kick_falls": None}
    frames = game_eval.with_name(game_eval.name + ".frames.json")
    if frames.exists():
        fr = {x["name"]: x["series"] for x in json.loads(frames.read_text())}
        scored = n = kf = 0
        for s in fr.values():
            for pi in range(len(s[0]["players"])):
                prev = None
                for i in range(1, len(s)):
                    p, q = s[i]["players"][pi], s[i - 1]["players"][pi]
                    if p["kick"] > 0 and q["kick"] == 0:
                        n += 1
                        scored += sum(s[min(len(s) - 1, i + 25)]["scores"].values()) > sum(s[i]["scores"].values())
                    rec = p.get("recovery")
                    if rec and not prev and any(s[j]["players"][pi]["kick"] > 0 for j in range(max(0, i - 8), i)):
                        kf += 1
                    prev = rec
        m["kick_to_goal"] = scored / max(n, 1)
        m["kick_falls"] = kf
    return m


def s4_metrics(path: Path):
    if not path.exists():
        return None
    M = json.loads(path.read_text()).get("mujoco(game)", {})
    ms = [v for v in M.values() if "handoff_tilt_max_deg" in v]
    hits = [v for v in M.values() if v.get("hit")]
    return {"handoff_tilt": float(np.median([v["handoff_tilt_max_deg"] for v in ms])) if ms else None,
            "s4_falls": sum(v["falls"] for v in M.values()),
            "speed": float(np.median([v["peak_speed"] for v in hits])) if hits else None,
            "dir": float(np.median([v["dir_deg"] for v in hits])) if hits else None}


def walk_metrics(d: Path):
    s1p = d / "s1_mujoco.json"
    out = {"match": match_metrics(d / "game_eval.json")}
    if s1p.exists():
        sys.path.insert(0, str(HERE.parent / "experiments" / "s1_sim2sim"))
        import analyze as s1
        scen = json.loads((HERE.parent / "experiments/s1_sim2sim/scenarios.json").read_text())["scenarios"]
        runs = json.loads(s1p.read_text())["runs"]
        m = {s: s1.metrics(s, scen[s], runs[s]) for s in scen}
        rnd = [s for s in scen if scen[s]["kind"] == "random"]
        out |= {"vx_0.30": m["fwd_0.30"]["vx"], "wz_1.0": m["turn_+1.0"]["wz"], "wz_1.5": m["turn_+1.5"]["wz"],
                "rmse_vx": float(np.mean([m[s]["vx_rmse"] for s in rnd])),
                "rmse_wz": float(np.mean([m[s]["wz_rmse"] for s in rnd])),
                "walk_falls": sum(v["falls"] for v in m.values())}
    return out


def f(v, fmt="{:.2f}"):
    if v is None:
        return "–"
    return fmt.format(v) if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    kick = {}
    for d in sorted(p for p in EVALS.iterdir() if p.is_dir() and p.name != "walk"):
        kick[d.name] = {"match": match_metrics(d / "game_eval.json"), "s4": s4_metrics(d / "s4_report.metrics.json")}
    walk = {d.name: walk_metrics(d) for d in sorted((EVALS / "walk").iterdir()) if d.is_dir()}
    if args.json:
        print(json.dumps({"kick": kick, "walk": walk}, indent=1))
        return
    print("## Kick policies (game evaluator with the official walking policy; S4 in game MuJoCo)\n")
    print("| eval id | goals | kicks | kick->goal | falls | kick falls | resets | S4 handoff tilt | S4 falls | S4 speed | S4 dir |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in kick.items():
        m, s = v["match"] or {}, v["s4"] or {}
        print(f"| {k} | {f(m.get('goals'))} | {f(m.get('kicks'))} | {f(m.get('kick_to_goal'), '{:.0%}')} | {f(m.get('falls'))} | "
              f"{f(m.get('kick_falls'))} | {f(m.get('resets'))} | {f(s.get('handoff_tilt'), '{:.1f}°')} | {f(s.get('s4_falls'))} | "
              f"{f(s.get('speed'))} | {f(s.get('dir'), '{:+.1f}°')} |")
    print("\n## Walk policies and walk+kick pairs (game MuJoCo; match = game evaluator)\n")
    print("| eval id | vx @0.3 | wz @1.0 | wz @1.5 | rmse vx | rmse wz | walk falls | goals | kick->goal | falls | kick falls | resets |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in walk.items():
        m = v["match"] or {}
        print(f"| {k} | {f(v.get('vx_0.30'), '{:.3f}')} | {f(v.get('wz_1.0'))} | {f(v.get('wz_1.5'))} | {f(v.get('rmse_vx'), '{:.3f}')} | "
              f"{f(v.get('rmse_wz'), '{:.3f}')} | {f(v.get('walk_falls'))} | {f(m.get('goals'))} | {f(m.get('kick_to_goal'), '{:.0%}')} | "
              f"{f(m.get('falls'))} | {f(m.get('kick_falls'))} | {f(m.get('resets'))} |")


if __name__ == "__main__":
    main()

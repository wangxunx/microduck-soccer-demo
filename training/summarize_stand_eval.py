"""Summarise evaluate_stand.sh outputs (one or more evals/stand/<id> folders, side by side).

    ../../microduck-rl-genesis/.venv/bin/python summarize_stand_eval.py evals/stand/official evals/stand/stand-A-9999
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments" / "s6_getup"))
sys.path.insert(0, str(HERE.parent / "experiments" / "s4_kick_sim2sim"))
from summarize import summarize  # noqa: E402  (S6)
from analyze import metrics as s4_case  # noqa: E402  (S4)
from collect_metrics import match_metrics  # noqa: E402


def stand_metrics(d: Path) -> dict:
    """Metrics from the raw files; cached in stand_metrics.json (raw trajectories are git-ignored)."""
    cache = d / "stand_metrics.json"
    if not (d / "getup.json").exists() and cache.exists():
        return json.loads(cache.read_text())
    m = _stand_metrics(d)
    cache.write_text(json.dumps(m, indent=1) + "\n")
    return m


def _stand_metrics(d: Path) -> dict:
    m = {}
    for tag, f in (("getup", "getup.json"), ("getup_walkA", "getup_walkA.json")):
        if (d / f).exists():
            m[tag] = summarize(d / f)[1]["all"]
    for tag in ("v4ft", "v5_walkA"):
        f = d / f"s4_{tag}.json"
        if f.exists():
            runs = json.loads(f.read_text())["runs"].values()
            cs = [s4_case(r) for r in runs]
            tilts = [c["handoff_tilt_max_deg"] for c in cs if "handoff_tilt_max_deg" in c]
            m[f"s4_{tag}"] = {"tilt": float(np.median(tilts)) if tilts else None,
                              "tilt_max": float(np.max(tilts)) if tilts else None,
                              "falls": sum(c["falls"] for c in cs), "n": len(cs)}
    for tag in ("author", "amd"):
        m[f"match_{tag}"] = match_metrics(d / f"game_eval_{tag}.json")
    return m


def f(v, fmt="{:.2f}"):
    return "–" if v is None else (fmt.format(v) if isinstance(v, float) else str(v))


def main():
    rows = {Path(a).name: stand_metrics(Path(a)) for a in sys.argv[1:]}
    print("| eval id | get-up recovered | resets | relapses | stand->upright med/max (s) | same, walk-A: recovered / relapses | "
          "S4 handoff tilt med/max v4ft | S4 falls v4ft | handoff tilt v5+walk-A | S4 falls v5 | "
          "author set: goals / falls / resets | AMD set: goals / kick->goal / falls / resets |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, m in rows.items():
        g, gw = m.get("getup") or {}, m.get("getup_walkA") or {}
        s4a, s4b = m.get("s4_v4ft") or {}, m.get("s4_v5_walkA") or {}
        a, b = m.get("match_author") or {}, m.get("match_amd") or {}
        print(f"| {k} | {g.get('recovered', '–')}/{g.get('fell', '–')} | {g.get('resets', '–')} | {g.get('relapse', '–')} | "
              f"{g.get('t_up', '–')}/{f(g.get('t_up_max'))} | {gw.get('recovered', '–')}/{gw.get('fell', '–')} / {gw.get('relapse', '–')} | "
              f"{f(s4a.get('tilt'), '{:.1f}°')}/{f(s4a.get('tilt_max'), '{:.1f}°')} | {f(s4a.get('falls'))} | "
              f"{f(s4b.get('tilt'), '{:.1f}°')}/{f(s4b.get('tilt_max'), '{:.1f}°')} | {f(s4b.get('falls'))} | "
              f"{f(a.get('goals'))} / {f(a.get('falls'))} / {f(a.get('resets'))} | "
              f"{f(b.get('goals'))} / {f(b.get('kick_to_goal'), '{:.0%}')} / {f(b.get('falls'))} / {f(b.get('resets'))} |")


if __name__ == "__main__":
    main()

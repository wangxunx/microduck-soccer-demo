"""Summarise S6 get-up runs per start family (one or more result files side by side).

    python3 summarize.py results/official.json results/stand-A-9999.json
"""
import json
import statistics as st
import sys
from pathlib import Path

FAMILIES = ("face_down", "face_up", "side", "sitting", "drop", "shove")


def med(xs):
    xs = [x for x in xs if x is not None]
    return f"{st.median(xs):.2f}" if xs else "–"


def summarize(path):
    d = json.loads(Path(path).read_text())
    rows = {}
    for fam in FAMILIES + ("all",):
        rs = [r for r in d["runs"].values() if fam == "all" or r["family"] == fam]
        fell = [r for r in rs if r["recoveryEntries"] > 0]
        rows[fam] = {
            "n": len(rs), "fell": len(fell),
            "recovered": sum(r["recovered"] for r in fell),
            "resets": sum(r["forcedResets"] for r in rs),
            "relapse": sum(r["recoveryEntries"] > 1 for r in rs),
            "t_up": med([r["standToUpright"] for r in fell]),
            "t_up_max": max([r["standToUpright"] for r in fell if r["standToUpright"] is not None], default=None),
            "tilt_after": med([r["maxTiltAfterExit"] for r in fell if r["recoveryEntries"] == 1]),
            "upright_end": sum(r["uprightAtEnd"] for r in rs),
        }
    return d, rows


def main():
    for path in sys.argv[1:]:
        d, rows = summarize(path)
        print(f"\n### {Path(path).stem}  (stand: {Path(d['stand']).name}, walk: {Path(d['walk']).name})\n")
        print("| family | cases | needed stand | recovered | forced resets | relapses | stand->upright med / max (s) | max tilt 2 s after hand-back, med (deg) | upright at 10 s |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for fam, r in rows.items():
            mx = f"{r['t_up_max']:.2f}" if r["t_up_max"] is not None else "–"
            print(f"| {fam} | {r['n']} | {r['fell']} | {r['recovered']}/{r['fell']} | {r['resets']} | {r['relapse']} | "
                  f"{r['t_up']} / {mx} | {r['tilt_after']} | {r['upright_end']}/{r['n']} |")


if __name__ == "__main__":
    main()

"""Compare game evaluate-football.mjs results: candidate vs baseline.

    python summarize_game_eval.py evals/<id>/game_eval.json evals/baseline-author/game_eval.json
"""
import json
import sys
from pathlib import Path

KEYS = [("goals", "goals"), ("kickAttempts", "kicks"), ("kicksWithSampledBallContact", "kick contact"),
        ("recoveryEntries", "falls/recov"), ("forcedResets", "resets"), ("fallenRobotSeconds", "fallen s"),
        ("ballPathMetres", "ball path m"), ("firstGoalSeconds", "1st goal s")]


def load(p):
    return {r["name"]: r for r in json.loads(Path(p).read_text())["results"]}


def val(r, k):
    v = r.get(k)
    if k == "goals":
        return f"{v.get('blue', 0)}-{v.get('orange', 0)}"
    if v is None:
        return "–"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def total(rs, k):
    if k == "goals":
        return sum(sum(r["goals"].values()) for r in rs.values())
    return round(sum((r.get(k) or 0) for r in rs.values()), 1)


def main():
    cand = load(sys.argv[1])
    base = load(sys.argv[2]) if len(sys.argv) > 2 and Path(sys.argv[2]).exists() else {}
    print("| scenario | " + " | ".join(lbl for _, lbl in KEYS) + " |")
    print("|---|" + "---|" * len(KEYS))
    for name, r in cand.items():
        row = []
        for k, _ in KEYS:
            c = val(r, k)
            b = val(base[name], k) if name in base else None
            row.append(c if b is None or b == c else f"{c} ({b})")
        print(f"| {name} | " + " | ".join(row) + " |")
    print("\nTotals (candidate vs baseline):")
    for k, lbl in KEYS:
        if k in ("firstGoalSeconds",):
            continue
        print(f"  {lbl:14s} {total(cand, k):>8}  vs {total(base, k) if base else '–':>8}")
    print("finite:", all(r.get("finite", True) for r in cand.values()))


if __name__ == "__main__":
    main()

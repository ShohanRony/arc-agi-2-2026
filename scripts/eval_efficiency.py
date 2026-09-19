"""Accuracy-vs-compute sweep on the held-out evaluation split (120 tasks, never seen by the
ranker). Gate is now BEFORE any search: skip depth-1 and depth-2 entirely on tasks the
calibrated model rates unlikely to be solvable by our DSL library at all (previous version
gated only the depth-2 step, which turned out to have no learnable signal - see git history).

Single pass computes depth-1 and (if needed) depth-2 once per task, caching timing and the
calibrated probability, then the threshold sweep is simulated analytically from that cache -
no extra search time spent per threshold.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_split
from src.features import extract
from src.search import find_program, apply_program

ROOT = Path(__file__).resolve().parent.parent


def load_ranker():
    d = np.load(ROOT / "ranker.npz")
    return d["w"], float(d["b"]), float(d["a"]), float(d["c"])


def calibrated_prob(feats, w, b, a, c):
    z = np.dot(w, feats) + b
    return 1.0 / (1.0 + np.exp(-(a * z + c)))


def main():
    w, b, a, c = load_ranker()
    tasks = load_split("evaluation")

    rows = []
    for tid, task in tasks.items():
        prob = calibrated_prob(np.array(extract(task["train"])), w, b, a, c)

        t0 = time.time()
        prog1 = find_program(task, max_depth=1)
        t_d1 = time.time() - t0
        solved_d1 = prog1 is not None and all(
            apply_program(prog1, gin) == gout for gin, gout in task["test"]
        )

        t_d2 = 0.0
        solved_d2 = False
        if prog1 is None:
            t0 = time.time()
            prog2 = find_program(task, max_depth=2)
            t_d2 = time.time() - t0
            solved_d2 = prog2 is not None and all(
                apply_program(prog2, gin) == gout for gin, gout in task["test"]
            )
        rows.append(dict(tid=tid, prob=prob, t_search=t_d1 + t_d2,
                          solved=solved_d1 or solved_d2))

    n = len(rows)
    base_time = sum(r["t_search"] for r in rows)
    base_solved = sum(r["solved"] for r in rows)
    print(f"unrestricted (always search): solved {base_solved}/{n} ({base_solved / n:.1%}), "
          f"total search time {base_time:.1f}s")

    print("\nthreshold  tasks_searched  solved  solve_rate  total_time  time_saved")
    for tau in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.01]:
        searched = solved = 0
        total_time = 0.0
        for r in rows:
            if r["prob"] >= tau:
                searched += 1
                total_time += r["t_search"]
                if r["solved"]:
                    solved += 1
        saved = 1 - total_time / base_time
        print(f"{tau:9.2f}  {searched:14d}  {solved:6d}  {solved / n:9.1%}  "
              f"{total_time:9.1f}s  {saved:9.1%}")


if __name__ == "__main__":
    main()

"""Accuracy-vs-compute sweep on the held-out evaluation split (120 tasks, never seen by the
ranker). Single pass computes depth-1 and (if needed) depth-2 once per task, records timing
and the calibrated probability, then simulates every threshold analytically from that cache -
so the sweep itself costs no extra search time.
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
        t0 = time.time()
        prog1 = find_program(task, max_depth=1)
        t_d1 = time.time() - t0
        solved_d1 = prog1 is not None and all(
            apply_program(prog1, gin) == gout for gin, gout in task["test"]
        )

        needs_d2 = prog1 is None
        prob = calibrated_prob(np.array(extract(task["train"])), w, b, a, c) if needs_d2 else None
        t_d2 = 0.0
        solved_d2 = False
        if needs_d2:
            t0 = time.time()
            prog2 = find_program(task, max_depth=2)
            t_d2 = time.time() - t0
            solved_d2 = prog2 is not None and all(
                apply_program(prog2, gin) == gout for gin, gout in task["test"]
            )
        rows.append(dict(tid=tid, t_d1=t_d1, solved_d1=solved_d1, needs_d2=needs_d2,
                          prob=prob, t_d2=t_d2, solved_d2=solved_d2))

    n = len(rows)
    base_time = sum(r["t_d1"] for r in rows) + sum(r["t_d2"] for r in rows)
    base_solved = sum(r["solved_d1"] or r["solved_d2"] for r in rows)
    print(f"unrestricted (always expand to depth-2): solved {base_solved}/{n} "
          f"({base_solved / n:.1%}), total search time {base_time:.1f}s")

    print("\nthreshold  solved  solve_rate  total_time  time_saved_vs_unrestricted")
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
        solved = 0
        total_time = sum(r["t_d1"] for r in rows)
        for r in rows:
            if r["solved_d1"]:
                solved += 1
                continue
            if r["needs_d2"] and r["prob"] is not None and r["prob"] >= tau:
                total_time += r["t_d2"]
                if r["solved_d2"]:
                    solved += 1
        saved = 1 - total_time / base_time
        print(f"{tau:9.2f}  {solved:6d}  {solved / n:9.1%}  {total_time:9.1f}s  {saved:22.1%}")


if __name__ == "__main__":
    main()

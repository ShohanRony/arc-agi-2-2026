"""Correctness/solve-rate check for the depth<=2 DSL baseline.

Usage: python scripts/eval_baseline.py --split training --limit 50
Loads tasks with known test outputs, searches for a program from the train pairs only,
then checks whether applying it to the held-out test input reproduces the real test output.
This is the "does the search actually work" self-check, not a claim about eval-set accuracy.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_split
from src.search import find_program, apply_program


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="training", choices=["training", "evaluation"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    tasks = load_split(args.split)
    task_ids = list(tasks)[: args.limit]

    solved = 0
    found_program = 0
    t0 = time.time()
    for tid in task_ids:
        task = tasks[tid]
        program = find_program(task, max_depth=args.depth)
        if program is not None:
            found_program += 1
            ok = all(
                apply_program(program, gin) == gout
                for gin, gout in task["test"]
            )
            if ok:
                solved += 1
    elapsed = time.time() - t0

    n = len(task_ids)
    print(f"split={args.split} n={n} programs_found={found_program} test_solved={solved} "
          f"solve_rate={solved / n:.1%} time={elapsed:.1f}s ({elapsed / n:.2f}s/task)")
    assert found_program > 0, "sanity check failed: the DSL search found zero programs at all"


if __name__ == "__main__":
    main()

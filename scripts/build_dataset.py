"""Build the ranker's training set: for every training-split task, record features and
whether expanding search from depth 1 to depth 2 was necessary and successful.

Only tasks where depth-1 already fails are useful supervision for "is depth-2 worth the
compute" (that's the actual decision the ranker gates at inference time).
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_split
from src.features import extract, FEATURE_NAMES
from src.search import find_program

OUT = Path(__file__).resolve().parent.parent / "dataset.csv"


def main():
    tasks = load_split("training")
    rows = []
    t0 = time.time()
    for i, (tid, task) in enumerate(tasks.items()):
        train_pairs = task["train"]
        feats = extract(train_pairs)
        d1 = find_program(task, max_depth=1)
        needs_depth2 = d1 is None
        found_at_depth2 = False
        if needs_depth2:
            d2 = find_program(task, max_depth=2)
            found_at_depth2 = d2 is not None
        rows.append([tid] + feats + [int(needs_depth2), int(found_at_depth2)])
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(tasks)} tasks, {time.time() - t0:.0f}s elapsed")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id"] + FEATURE_NAMES + ["needed_depth2", "depth2_found"])
        w.writerows(rows)

    n_needed = sum(r[-2] for r in rows)
    n_found = sum(r[-1] for r in rows)
    print(f"wrote {len(rows)} rows to {OUT}")
    print(f"depth-1 already sufficient: {len(rows) - n_needed}/{len(rows)}")
    print(f"of {n_needed} that needed depth-2, {n_found} found something "
          f"({n_found / n_needed:.1%} positive rate for the ranker's target)")


if __name__ == "__main__":
    main()

"""Load ARC-AGI-2 tasks from the cloned data_repo."""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data_repo", "data")


def to_grid(rows):
    return tuple(tuple(row) for row in rows)


def load_split(split):
    """split: 'training' or 'evaluation'. Returns {task_id: {"train": [(in,out),...], "test": [(in,out),...]}}"""
    d = os.path.join(DATA_DIR, split)
    tasks = {}
    for fname in sorted(os.listdir(d)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(d, fname)) as f:
            raw = json.load(f)
        tasks[fname[:-5]] = {
            "train": [(to_grid(p["input"]), to_grid(p["output"])) for p in raw["train"]],
            "test": [(to_grid(p["input"]), to_grid(p["output"])) for p in raw["test"]],
        }
    return tasks

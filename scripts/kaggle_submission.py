"""Self-contained Kaggle submission script for ARC-AGI-2 (paste as single notebook cell).

No external imports beyond stdlib (json/collections/pathlib) since Kaggle eval runs offline.
Mirrors src/dsl.py + src/search.py logic inline so this one file is the whole submission.

Adjust CHALLENGES_PATH below once dataset is attached in Kaggle (check the competition's
Data tab for the exact input filename, typically .../arc-agi_test_challenges.json).
"""
import json
from collections import Counter
from pathlib import Path

CHALLENGES_PATH = "/kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
OUTPUT_PATH = "/kaggle/working/submission.json"
MAX_FACTOR = 4
MAX_DEPTH = 2


# ---- DSL primitives (Grid = tuple[tuple[int, ...], ...]) ----
def identity(g): return g
def rot90(g): return tuple(zip(*g[::-1]))
def rot180(g): return tuple(row[::-1] for row in g[::-1])
def rot270(g): return tuple(zip(*g))[::-1]
def flip_h(g): return tuple(row[::-1] for row in g)
def flip_v(g): return g[::-1]
def transpose(g): return tuple(zip(*g))


def background_color(g):
    return Counter(cell for row in g for cell in row).most_common(1)[0][0]


def crop_to_content(g):
    bg = background_color(g)
    rows = [r for r, row in enumerate(g) if any(c != bg for c in row)]
    cols = [c for c in range(len(g[0])) if any(row[c] != bg for row in g)]
    if not rows or not cols:
        raise ValueError("empty after crop")
    return tuple(row[cols[0]:cols[-1] + 1] for row in g[rows[0]:rows[-1] + 1])


def scale_up(factor):
    def f(g):
        return tuple(tuple(c for c in row for _ in range(factor)) for row in g for _ in range(factor))
    return f


def scale_down(factor):
    def f(g):
        h, w = len(g), len(g[0])
        if h % factor or w % factor:
            raise ValueError("not divisible")
        out = []
        for r in range(0, h, factor):
            row = []
            for c in range(0, w, factor):
                block = {g[r + dr][c + dc] for dr in range(factor) for dc in range(factor)}
                if len(block) != 1:
                    raise ValueError("block not uniform")
                row.append(block.pop())
            out.append(tuple(row))
        return tuple(out)
    return f


def tile(ky, kx):
    def f(g):
        h, w = len(g), len(g[0])
        return tuple(tuple(g[r % h][c % w] for c in range(w * kx)) for r in range(h * ky))
    return f


def mirror_tile(ky, kx):
    def f(g):
        h, w = len(g), len(g[0])
        out = [[0] * (w * kx) for _ in range(h * ky)]
        for by in range(ky):
            for bx in range(kx):
                block = g
                if by % 2: block = flip_v(block)
                if bx % 2: block = flip_h(block)
                for r in range(h):
                    for c in range(w):
                        out[by * h + r][bx * w + c] = block[r][c]
        return tuple(tuple(row) for row in out)
    return f


ATOMIC = [identity, rot90, rot180, rot270, flip_h, flip_v, transpose, crop_to_content]


# ---- search ----
def infer_recolor(train_pairs):
    mapping = {}
    for gin, gout in train_pairs:
        if len(gin) != len(gout) or len(gin[0]) != len(gout[0]):
            return None
        for rin, rout in zip(gin, gout):
            for a, b in zip(rin, rout):
                if a in mapping and mapping[a] != b:
                    return None
                mapping[a] = b

    def recolor(g):
        return tuple(tuple(mapping.get(c, c) for c in row) for row in g)
    return recolor


def candidate_transforms(train_pairs):
    cands = list(ATOMIC)
    recolor = infer_recolor(train_pairs)
    if recolor:
        cands.append(recolor)
    for ky in range(1, MAX_FACTOR + 1):
        for kx in range(1, MAX_FACTOR + 1):
            if ky == 1 and kx == 1:
                continue
            cands.append(tile(ky, kx))
            cands.append(mirror_tile(ky, kx))
    for f in range(2, MAX_FACTOR + 1):
        cands.append(scale_up(f))
        cands.append(scale_down(f))
    return cands


def matches_all(fn, train_pairs):
    try:
        return all(fn(gin) == gout for gin, gout in train_pairs)
    except (ValueError, IndexError, ZeroDivisionError):
        return False


def find_program(train_pairs, max_depth=MAX_DEPTH):
    cands = candidate_transforms(train_pairs)
    for fn in cands:
        if matches_all(fn, train_pairs):
            return [fn]
    if max_depth >= 2:
        for f1 in cands:
            try:
                mid = [(f1(gin), gout) for gin, gout in train_pairs]
            except (ValueError, IndexError, ZeroDivisionError):
                continue
            for f2 in cands:
                if matches_all(f2, mid):
                    return [f1, f2]
    return None


def apply_program(program, grid):
    for fn in program:
        grid = fn(grid)
    return grid


def to_grid(rows):
    return tuple(tuple(row) for row in rows)


def to_list(grid):
    return [list(row) for row in grid]


def main():
    challenges = json.loads(Path(CHALLENGES_PATH).read_text())
    submission = {}
    for task_id, task in challenges.items():
        train_pairs = [(to_grid(p["input"]), to_grid(p["output"])) for p in task["train"]]
        program = find_program(train_pairs)
        entries = []
        for test_pair in task["test"]:
            gin = to_grid(test_pair["input"])
            try:
                attempt_1 = apply_program(program, gin) if program else gin
            except (ValueError, IndexError, ZeroDivisionError):
                attempt_1 = gin
            entries.append({"attempt_1": to_list(attempt_1), "attempt_2": to_list(gin)})
        submission[task_id] = entries
    Path(OUTPUT_PATH).write_text(json.dumps(submission))
    print(f"wrote {len(submission)} tasks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

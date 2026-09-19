"""Brute-force program search: compose DSL primitives until every training pair matches exactly.

This is the week-1 baseline solver (no learning yet) — depth-<=2 composition over a small
primitive library. It exists to (a) prove the end-to-end pipeline works and (b) serve as the
"no ranker" ablation once the learned ranker (weeks 2-3) is added to prioritize search order.
"""
from src.dsl import ATOMIC, FAMILIES

MAX_FACTOR = 4  # cap on tile/scale factors so the candidate list stays small


def infer_recolor(train_pairs):
    """If every pair has matching shapes and a single consistent color mapping, return it."""
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
    recolor.__name__ = "recolor(learned)"
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
            cands.append(FAMILIES[2](ky, kx))  # tile
            cands.append(FAMILIES[3](ky, kx))  # mirror_tile
    for f in range(2, MAX_FACTOR + 1):
        cands.append(FAMILIES[0](f))  # scale_up
        cands.append(FAMILIES[1](f))  # scale_down
    return cands


def _matches_all(fn, train_pairs):
    try:
        return all(fn(gin) == gout for gin, gout in train_pairs)
    except (ValueError, IndexError, ZeroDivisionError):
        return False


def find_program(task, max_depth=2):
    """Returns a list of primitive functions (applied in order) that solves every train pair,
    or None if nothing up to max_depth was found."""
    train_pairs = task["train"]
    cands = candidate_transforms(train_pairs)

    for fn in cands:
        if _matches_all(fn, train_pairs):
            return [fn]

    if max_depth >= 2:
        for f1 in cands:
            try:
                mid = [(f1(gin), gout) for gin, gout in train_pairs]
            except (ValueError, IndexError, ZeroDivisionError):
                continue
            for f2 in cands:
                if _matches_all(f2, mid):
                    return [f1, f2]

    return None


def apply_program(program, grid):
    for fn in program:
        grid = fn(grid)
    return grid


def predict_two(task, program):
    """Kaggle scoring wants exactly 2 candidate output grids per test input."""
    preds = []
    for gin, _ in task["test"]:
        if program is not None:
            try:
                p1 = apply_program(program, gin)
            except (ValueError, IndexError, ZeroDivisionError):
                p1 = gin
        else:
            p1 = gin
        p2 = gin  # weak second guess until the ranker (weeks 2-3) picks a real runner-up
        preds.append((p1, p2))
    return preds

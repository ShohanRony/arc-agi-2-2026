"""Per-task augmentation for test-time training: geometric transform (reused straight from
src/dsl.py - same functions the DSL baseline searched over, different job here) + a random
color permutation, applied consistently to every grid in a task. Both kinds of transform are
invertible, so a prediction made on an augmented task can be mapped back to the original frame.
"""
import random

from src.dsl import identity, rot90, rot180, rot270, flip_h, flip_v, transpose

# name -> (forward, inverse). Geometric and color transforms act on independent axes
# (position vs. value), so they commute - de-augmentation order doesn't matter.
GEOM_TRANSFORMS = {
    "identity": (identity, identity),
    "rot90": (rot90, rot270),
    "rot180": (rot180, rot180),
    "rot270": (rot270, rot90),
    "flip_h": (flip_h, flip_h),
    "flip_v": (flip_v, flip_v),
    "transpose": (transpose, transpose),
}


def random_color_perm(rng):
    perm = list(range(10))
    rng.shuffle(perm)
    return perm


def apply_color_perm(grid, perm):
    return tuple(tuple(perm[c] for c in row) for row in grid)


def invert_perm(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


class Augmentation:
    def __init__(self, geom_name, color_perm):
        self.geom_name = geom_name
        self.color_perm = color_perm
        self.fwd_geom, self.inv_geom = GEOM_TRANSFORMS[geom_name]
        self.inv_color = invert_perm(color_perm)

    def apply(self, grid):
        return apply_color_perm(self.fwd_geom(grid), self.color_perm)

    def invert(self, grid):
        return self.inv_geom(apply_color_perm(grid, self.inv_color))


def random_augmentation(rng):
    geom_name = rng.choice(list(GEOM_TRANSFORMS))
    return Augmentation(geom_name, random_color_perm(rng))


def make_augmented_views(train_pairs, test_input, n, rng=None, include_identity=True):
    """Returns a list of (Augmentation, augmented_train_pairs, augmented_test_input)."""
    rng = rng or random.Random(0)
    augs = [Augmentation("identity", list(range(10)))] if include_identity else []
    while len(augs) < n:
        augs.append(random_augmentation(rng))
    out = []
    for aug in augs:
        aug_train = [(aug.apply(gin), aug.apply(gout)) for gin, gout in train_pairs]
        aug_test = aug.apply(test_input)
        out.append((aug, aug_train, aug_test))
    return out


if __name__ == "__main__":
    rng = random.Random(42)
    grid = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    for _ in range(20):
        aug = random_augmentation(rng)
        assert aug.invert(aug.apply(grid)) == grid, f"round-trip failed for {aug.geom_name}"
    print("augment.py self-check OK")

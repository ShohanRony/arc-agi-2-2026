"""Minimal grid-transform primitive library (Hodel/icecuber-style DSL) for ARC-AGI-2.

Grid = tuple[tuple[int, ...], ...]. Every primitive is a pure function Grid -> Grid
(or raises ValueError if inapplicable to that grid), so the search module can compose
and dedupe them freely.
"""
from collections import Counter


def identity(g):
    return g


def rot90(g):
    return tuple(zip(*g[::-1]))


def rot180(g):
    return tuple(row[::-1] for row in g[::-1])


def rot270(g):
    return tuple(zip(*g))[::-1]


def flip_h(g):
    return tuple(row[::-1] for row in g)


def flip_v(g):
    return g[::-1]


def transpose(g):
    return tuple(zip(*g))


def background_color(g):
    counts = Counter(cell for row in g for cell in row)
    return counts.most_common(1)[0][0]


def crop_to_content(g):
    bg = background_color(g)
    rows = [r for r, row in enumerate(g) if any(c != bg for c in row)]
    cols = [c for c in range(len(g[0])) if any(row[c] != bg for row in g)]
    if not rows or not cols:
        raise ValueError("empty after crop")
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    return tuple(row[c0:c1 + 1] for row in g[r0:r1 + 1])


def scale_up(factor):
    def f(g):
        return tuple(
            tuple(cell for cell in row for _ in range(factor))
            for row in g for _ in range(factor)
        )
    f.__name__ = f"scale_up({factor})"
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
    f.__name__ = f"scale_down({factor})"
    return f


def tile(ky, kx):
    def f(g):
        return tuple(
            tuple(g[r % len(g)][c % len(g[0])] for c in range(len(g[0]) * kx))
            for r in range(len(g) * ky)
        )
    f.__name__ = f"tile({ky},{kx})"
    return f


def mirror_tile(ky, kx):
    """Tile ky x kx blocks, mirroring alternate blocks (common ARC symmetry-completion pattern)."""
    def f(g):
        h, w = len(g), len(g[0])
        out = [[0] * (w * kx) for _ in range(h * ky)]
        for by in range(ky):
            for bx in range(kx):
                block = g
                if by % 2:
                    block = flip_v(block)
                if bx % 2:
                    block = flip_h(block)
                for r in range(h):
                    for c in range(w):
                        out[by * h + r][bx * w + c] = block[r][c]
        return tuple(tuple(row) for row in out)
    f.__name__ = f"mirror_tile({ky},{kx})"
    return f


# Parameterless transforms tried at every search depth.
ATOMIC = [identity, rot90, rot180, rot270, flip_h, flip_v, transpose, crop_to_content]

# Parameterized transform families, expanded by the search module against a size cap.
FAMILIES = [scale_up, scale_down, tile, mirror_tile]

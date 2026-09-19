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


# ---- object segmentation (same-color, 4-connectivity) ----
def _components(g, bg):
    h, w = len(g), len(g[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r][c] == bg or seen[r][c]:
                continue
            color = g[r][c]
            stack, cells = [(r, c)], []
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and g[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((color, cells))
    return comps


def _bbox(cells):
    rs = [r for r, c in cells]
    cs = [c for r, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _crop_object(g, bg, cells):
    r0, r1, c0, c1 = _bbox(cells)
    cellset = set(cells)
    return tuple(
        tuple(g[r][c] if (r, c) in cellset else bg for c in range(c0, c1 + 1))
        for r in range(r0, r1 + 1)
    )


def crop_largest_object(g):
    bg = background_color(g)
    comps = _components(g, bg)
    if not comps:
        raise ValueError("no objects")
    _, cells = max(comps, key=lambda x: len(x[1]))
    return _crop_object(g, bg, cells)


def crop_smallest_object(g):
    bg = background_color(g)
    comps = _components(g, bg)
    if not comps:
        raise ValueError("no objects")
    _, cells = min(comps, key=lambda x: len(x[1]))
    return _crop_object(g, bg, cells)


def crop_unique_color_object(g):
    bg = background_color(g)
    comps = _components(g, bg)
    if not comps:
        raise ValueError("no objects")
    counts = Counter(color for color, _ in comps)
    uniques = [(color, cells) for color, cells in comps if counts[color] == 1]
    if len(uniques) != 1:
        raise ValueError("no single unique-color object")
    _, cells = uniques[0]
    return _crop_object(g, bg, cells)


def _keep_object(g, pick):
    bg = background_color(g)
    comps = _components(g, bg)
    if not comps:
        raise ValueError("no objects")
    _, cells = pick(comps, key=lambda x: len(x[1]))
    cellset = set(cells)
    return tuple(
        tuple(g[r][c] if (r, c) in cellset else bg for c in range(len(g[0])))
        for r in range(len(g))
    )


def keep_largest_object(g):
    return _keep_object(g, max)


def keep_smallest_object(g):
    return _keep_object(g, min)


# ---- gravity: pack non-background cells toward one edge, per row/column ----
def gravity_down(g):
    bg = background_color(g)
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    for c in range(w):
        vals = [g[r][c] for r in range(h) if g[r][c] != bg]
        for i, v in enumerate(vals):
            out[h - len(vals) + i][c] = v
    return tuple(tuple(row) for row in out)


def gravity_up(g):
    bg = background_color(g)
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    for c in range(w):
        vals = [g[r][c] for r in range(h) if g[r][c] != bg]
        for i, v in enumerate(vals):
            out[i][c] = v
    return tuple(tuple(row) for row in out)


def gravity_left(g):
    bg = background_color(g)
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    for r in range(h):
        vals = [g[r][c] for c in range(w) if g[r][c] != bg]
        for i, v in enumerate(vals):
            out[r][i] = v
    return tuple(tuple(row) for row in out)


def gravity_right(g):
    bg = background_color(g)
    h, w = len(g), len(g[0])
    out = [[bg] * w for _ in range(h)]
    for r in range(h):
        vals = [g[r][c] for c in range(w) if g[r][c] != bg]
        for i, v in enumerate(vals):
            out[r][w - len(vals) + i] = v
    return tuple(tuple(row) for row in out)


# ---- grid split by a uniform separator row/col + panel overlay ----
def _split_two_panels(g):
    h, w = len(g), len(g[0])
    for r in range(1, h - 1):
        if len(set(g[r])) == 1:
            top, bottom = g[:r], g[r + 1:]
            if len(top) == len(bottom):
                return top, bottom
    for c in range(1, w - 1):
        if len({g[r][c] for r in range(h)}) == 1:
            left = tuple(row[:c] for row in g)
            right = tuple(row[c + 1:] for row in g)
            if len(left[0]) == len(right[0]):
                return left, right
    raise ValueError("no clean 2-panel separator")


def _panel_paint_color(panel, bg):
    for row in panel:
        for v in row:
            if v != bg:
                return v
    return bg


def split_overlay_priority(g):
    bg = background_color(g)
    a, b = _split_two_panels(g)
    return tuple(
        tuple(a[r][c] if a[r][c] != bg else b[r][c] for c in range(len(a[0])))
        for r in range(len(a))
    )


def split_overlay_and(g):
    bg = background_color(g)
    a, b = _split_two_panels(g)
    pc = _panel_paint_color(a, bg)
    return tuple(
        tuple(pc if (a[r][c] != bg and b[r][c] != bg) else bg for c in range(len(a[0])))
        for r in range(len(a))
    )


def split_overlay_xor(g):
    bg = background_color(g)
    a, b = _split_two_panels(g)
    pc = _panel_paint_color(a, bg)
    return tuple(
        tuple(pc if ((a[r][c] != bg) != (b[r][c] != bg)) else bg for c in range(len(a[0])))
        for r in range(len(a))
    )


# Parameterless transforms tried at every search depth.
ATOMIC = [
    identity, rot90, rot180, rot270, flip_h, flip_v, transpose, crop_to_content,
    crop_largest_object, crop_smallest_object, crop_unique_color_object,
    keep_largest_object, keep_smallest_object,
    gravity_up, gravity_down, gravity_left, gravity_right,
    split_overlay_priority, split_overlay_and, split_overlay_xor,
]

# Parameterized transform families, expanded by the search module against a size cap.
FAMILIES = [scale_up, scale_down, tile, mirror_tile]

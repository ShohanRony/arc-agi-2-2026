"""Grid <-> text serialization for prompting a causal LM. Deliberately boring: rows as
space-separated digits, blank line between grids. Format bugs here silently break everything
downstream, so keep it simple and covered by the round-trip check at the bottom.
"""
import re

GRID_ROW_RE = re.compile(r"^[0-9]( [0-9])*$")


def grid_to_text(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def text_to_grid(text):
    lines = [line for line in text.strip("\n").split("\n") if line.strip()]
    if not lines:
        raise ValueError("empty grid text")
    rows = []
    width = None
    for line in lines:
        if not GRID_ROW_RE.match(line.strip()):
            raise ValueError(f"malformed grid row: {line!r}")
        row = tuple(int(x) for x in line.split())
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("ragged grid")
        rows.append(row)
    return tuple(rows)


def format_prompt(train_pairs, test_input):
    """Train pairs + test input, ending right where the model should continue generating."""
    parts = []
    for gin, gout in train_pairs:
        parts.append(f"Input:\n{grid_to_text(gin)}\nOutput:\n{grid_to_text(gout)}")
    parts.append(f"Input:\n{grid_to_text(test_input)}\nOutput:\n")
    return "\n\n".join(parts)


def format_completion(gin, gout):
    """One train pair as a standalone (prompt, target) example for LoRA fine-tuning."""
    prompt = f"Input:\n{grid_to_text(gin)}\nOutput:\n"
    target = grid_to_text(gout)
    return prompt, target


if __name__ == "__main__":
    g = ((1, 2, 3), (4, 5, 6))
    assert text_to_grid(grid_to_text(g)) == g, "round-trip failed"
    try:
        text_to_grid("1 2\n3")
        raise AssertionError("should have rejected ragged grid")
    except ValueError:
        pass
    print("serialize.py self-check OK")

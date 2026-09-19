"""Task-level features computed only from train pairs (available at test time too).
Small tabular feature set on purpose: logistic regression on ~1000 tasks overfits fast
with more than a handful of features, and the goal is a calibrated probability, not raw
predictive power.
"""
from src.search import infer_recolor

FEATURE_NAMES = [
    "num_train_pairs", "same_shape_frac", "avg_h_ratio", "avg_w_ratio",
    "h_ratio_consistent", "w_ratio_consistent", "avg_colors_in", "avg_colors_out",
    "color_set_equal_frac", "has_consistent_recolor", "avg_input_area",
]


def extract(train_pairs):
    n = len(train_pairs)
    same_shape = [len(gin) == len(gout) and len(gin[0]) == len(gout[0]) for gin, gout in train_pairs]
    h_ratios = [len(gout) / len(gin) for gin, gout in train_pairs]
    w_ratios = [len(gout[0]) / len(gin[0]) for gin, gout in train_pairs]
    colors_in = [len(set(c for row in gin for c in row)) for gin, gout in train_pairs]
    colors_out = [len(set(c for row in gout for c in row)) for gin, gout in train_pairs]
    color_set_equal = [
        set(c for row in gin for c in row) == set(c for row in gout for c in row)
        for gin, gout in train_pairs
    ]
    areas = [len(gin) * len(gin[0]) for gin, gout in train_pairs]

    def consistent(vals):
        return float(len(set(round(v, 3) for v in vals)) == 1)

    return [
        n,
        sum(same_shape) / n,
        sum(h_ratios) / n,
        sum(w_ratios) / n,
        consistent(h_ratios),
        consistent(w_ratios),
        sum(colors_in) / n,
        sum(colors_out) / n,
        sum(color_set_equal) / n,
        float(infer_recolor(train_pairs) is not None),
        sum(areas) / n,
    ]

"""Train + calibrate the "is this task solvable by our DSL library at all" ranker on
dataset.csv (all 1000 training tasks, not just the ones needing depth-2).

First target tried was "does depth-2 expansion help" (needed_depth2==1 rows only): 9/976
positive (0.9%) - too sparse, calibration was noise fit to the base rate, dropped. This
target - solved at depth<=2, any depth - is 33/1000 (3.3%), workable, and gates a bigger
compute cost: skipping the whole search (depth-1 and depth-2) on tasks predicted unlikely
to be solvable, not just skipping depth-2.

Two-stage fit so the final model is a portable (w, b, a, c) tuple, no sklearn needed at
inference time: base logistic regression gives raw score z = w.x + b; a *separate* 1D
Platt-scaling sigmoid p = sigmoid(a*z + c) is fit on a held-out calibration split (fitting
it on the same rows as w,b would just relearn the identity and hide any real miscalibration).

class_weight="balanced" on the base classifier matters here: at 3.3% positive rate, an
unweighted fit just predicts the base rate everywhere (looks "calibrated" by ECE, but has
zero ranking ability - mean AUC was ~0.5 that way). Balanced weighting recovers real signal
(5-fold CV AUC ~0.94); the Platt step afterwards then re-calibrates those now-informative
scores back into honest probabilities.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset.csv"
MODEL_OUT = ROOT / "ranker.npz"


def load_dataset():
    X, y = [], []
    with open(DATASET) as f:
        for row in csv.DictReader(f):
            solved = int(row["needed_depth2"]) == 0 or int(row["depth2_found"]) == 1
            X.append([float(row[name]) for name in FEATURE_NAMES])
            y.append(int(solved))
    return np.array(X), np.array(y)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def expected_calibration_error(y_true, p_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_pred >= lo) & (p_pred < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = p_pred[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc - conf)
    return ece


def main():
    X, y = load_dataset()
    print(f"dataset: {len(y)} rows, positive rate {y.mean():.1%}")

    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, random_state=0, stratify=y
    )
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest
    )

    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    base.fit(X_train, y_train)

    z_calib = base.decision_function(X_calib)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(z_calib.reshape(-1, 1), y_calib)
    a, c = platt.coef_[0, 0], platt.intercept_[0]

    z_test = base.decision_function(X_test)
    raw_probs = sigmoid(z_test)
    cal_probs = sigmoid(a * z_test + c)

    raw_ece = expected_calibration_error(y_test, raw_probs)
    cal_ece = expected_calibration_error(y_test, cal_probs)
    auc = roc_auc_score(y_test, z_test)
    print(f"test-split AUC (ranking ability): {auc:.3f}")
    print(f"raw logistic regression ECE:  {raw_ece:.4f}")
    print(f"Platt-calibrated ECE:         {cal_ece:.4f}")

    print("\nreliability table (calibrated model, held-out test rows):")
    print("pred_bin       n   mean_pred   empirical_freq")
    bins = np.linspace(0, 1, 6)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        print(f"[{lo:.1f},{hi:.1f})  {mask.sum():4d}   {cal_probs[mask].mean():.3f}      "
              f"{y_test[mask].mean():.3f}")

    # Refit base on train+calib (everything except the held-out test rows) for the saved model;
    # keep the Platt (a, c) fit above since it was already fit on a disjoint calibration split.
    X_fit = np.concatenate([X_train, X_calib])
    y_fit = np.concatenate([y_train, y_calib])
    base_final = LogisticRegression(max_iter=1000, class_weight="balanced")
    base_final.fit(X_fit, y_fit)

    np.savez(
        MODEL_OUT,
        w=base_final.coef_[0], b=base_final.intercept_[0],
        a=a, c=c, feature_names=FEATURE_NAMES,
    )
    print(f"\nsaved model (w, b, a, c) to {MODEL_OUT}")


if __name__ == "__main__":
    main()

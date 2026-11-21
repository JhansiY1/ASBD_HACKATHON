"""
Attribute-selection measures for decision-tree induction (Lecture 13),
implemented from first principles.

    Entropy(D)        = - sum_i p_i log2 p_i
    Info_A(D)         = sum_j |D_j|/|D| * Entropy(D_j)
    Gain(A)           = Entropy(D) - Info_A(D)                        (ID3)
    SplitInfo_A(D)    = - sum_j |D_j|/|D| log2(|D_j|/|D|)
    GainRatio(A)      = Gain(A) / SplitInfo_A(D)                      (C4.5)
    Gini(D)           = 1 - sum_i p_i^2
    Gini_A(D)         = |D1|/|D| Gini(D1) + |D2|/|D| Gini(D2)         (CART, binary)
    Delta Gini(A)     = Gini(D) - Gini_A(D)

For numeric attributes the best binary threshold (midpoints between sorted
distinct values) is searched, exactly as CART / C4.5 do for continuous
attributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def entropy(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return float(1 - (p ** 2).sum())


def evaluate_binary_split(x: np.ndarray, y: np.ndarray, thr: float) -> dict:
    left, right = y[x <= thr], y[x > thr]
    n = len(y)
    wl, wr = len(left) / n, len(right) / n
    info = wl * entropy(left) + wr * entropy(right)
    gain = entropy(y) - info
    split_info = -sum(w * np.log2(w) for w in (wl, wr) if w > 0)
    gini_split = wl * gini(left) + wr * gini(right)
    return dict(threshold=thr, info_gain=gain, split_info=split_info,
                gain_ratio=gain / split_info if split_info > 0 else 0.0,
                gini_split=gini_split, gini_reduction=gini(y) - gini_split,
                n_left=len(left), n_right=len(right))


def best_split(x: np.ndarray, y: np.ndarray, criterion: str = "info_gain", max_thresholds: int = 64) -> dict:
    vals = np.unique(x)
    if len(vals) < 2:
        return dict(threshold=np.nan, info_gain=0, split_info=0, gain_ratio=0, gini_split=gini(y),
                    gini_reduction=0, n_left=len(y), n_right=0)
    mids = (vals[:-1] + vals[1:]) / 2
    if len(mids) > max_thresholds:
        mids = np.quantile(x, np.linspace(0.02, 0.98, max_thresholds))
    best = None
    for t in mids:
        r = evaluate_binary_split(x, y, t)
        if best is None or r[criterion] > best[criterion]:
            best = r
    return best


def rank_attributes(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Root-node evaluation of every attribute with all three measures."""
    rows = []
    for c in X.columns:
        x = X[c].to_numpy(dtype=float)
        ig = best_split(x, y, "info_gain")
        # C4.5 practice for continuous attributes: the threshold is chosen by
        # information gain and the gain ratio is evaluated AT that threshold
        # (maximising the ratio itself favours tiny, uninformative edge splits)
        gi = best_split(x, y, "gini_reduction")
        rows.append(dict(attribute=c, info_gain=ig["info_gain"], ig_threshold=ig["threshold"],
                         gain_ratio=ig["gain_ratio"], split_info=ig["split_info"],
                         gini_reduction=gi["gini_reduction"], gini_threshold=gi["threshold"]))
    df = pd.DataFrame(rows)
    for m in ("info_gain", "gain_ratio", "gini_reduction"):
        df[f"rank_{m}"] = df[m].rank(ascending=False, method="min").astype(int)
    return df.sort_values("info_gain", ascending=False).reset_index(drop=True)

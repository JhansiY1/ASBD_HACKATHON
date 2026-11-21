"""Unit tests for split criteria, manual K-Means and similarity measures, plus
integration checks on the generated project outputs."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from src.classification.split_criteria import entropy, evaluate_binary_split, gini
from src.clustering.kmeans import kmeans_plus_plus_init, manual_kmeans

ROOT = Path(__file__).resolve().parents[1]


def test_entropy_and_gini_lecture13():
    y = np.array([1] * 9 + [0] * 5)                  # buys_computer: 9 yes, 5 no
    assert math.isclose(entropy(y), 0.940, abs_tol=1e-3)
    assert math.isclose(gini(y), 0.459, abs_tol=1e-3)  # slide value


def test_information_gain_and_gain_ratio():
    x = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    r = evaluate_binary_split(x, y, 0.5)
    assert math.isclose(r["info_gain"], 1.0) and math.isclose(r["split_info"], 1.0)
    assert math.isclose(r["gain_ratio"], 1.0) and math.isclose(r["gini_reduction"], 0.5)


def test_manual_kmeans_matches_sklearn():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.4, size=(40, 2)) for c in ((0, 0), (4, 4), (0, 5))])
    init = kmeans_plus_plus_init(X, 3, np.random.default_rng(1))
    labels, C, wcss, hist, _ = manual_kmeans(X, init)
    sk = KMeans(n_clusters=3, init=init, n_init=1, algorithm="lloyd").fit(X)
    assert adjusted_rand_score(labels, sk.labels_) == 1.0
    assert math.isclose(wcss, sk.inertia_, rel_tol=1e-9)
    assert all(a >= b - 1e-9 for a, b in zip(hist, hist[1:]))   # WCSS never increases


def test_lecture20_kmeans_example():
    # Lecture 20 worked example (7 individuals, K = 2): final clusters {1,2} and {3..7}
    X = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
    labels, C, *_ = manual_kmeans(X, np.array([[1.0, 1.0], [5.0, 7.0]]))
    assert set(np.where(labels == labels[0])[0]) == {0, 1}
    assert np.allclose(C[labels[0]], [1.25, 1.5]) and np.allclose(C[labels[3]], [3.9, 5.1])


def test_similarity_measures_lecture22():
    p1, p2 = np.array([1, 0, 1, 1, 1]), np.array([1, 0, 0, 1, 1])
    jac = (p1 & p2).sum() / (p1 | p2).sum()
    assert math.isclose(jac, 3 / 4)
    q1, q2 = np.array([0, 0, 1, 1, 1]), np.array([1, 0, 0, 1, 1])
    cos = q1 @ q2 / (np.linalg.norm(q1) * np.linalg.norm(q2))
    assert math.isclose(cos, 2 / 3)
    a, b = np.array([5, 5]), np.array([9, 8])
    assert np.linalg.norm(a - b) == 5 and np.abs(a - b).sum() == 7


# ---------------------------------------------------------------- integration
needs_outputs = pytest.mark.skipif(not (ROOT / "outputs" / "cofi" / "cofi_summary.json").exists(),
                                   reason="pipeline outputs not generated yet")


@needs_outputs
def test_cofi_validation_passed():
    s = json.loads((ROOT / "outputs" / "cofi" / "cofi_summary.json").read_text())
    v = s["validation"]
    assert v["V1_lecture11_example"]["passed"] and v["V2_bruteforce_random"]["passed"]
    assert v["V3_movie_cross_check"]["identical_to_apriori"] and v["V4_V5_closed_checks"]["passed"]


@needs_outputs
def test_dt_matrix_has_no_leakage_or_missing():
    X = pd.read_csv(ROOT / "data" / "processed" / "dt_ml_matrix.csv")
    assert not X.isna().any().any()
    assert not any(c.startswith(("oscar", "halliwell", "any_oscar", "was_remade", "times_remade")) for c in X.columns)


@needs_outputs
def test_at_least_ten_plot_types():
    meta = json.loads((ROOT / "outputs" / "plots_eda_meta.json").read_text())
    assert len({m["type"] for m in meta}) >= 10
    for m in meta:
        assert (ROOT / "outputs" / "plots" / m["file"]).exists()

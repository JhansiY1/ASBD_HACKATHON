"""
PHASE 8b - K-Means clustering of director career profiles (+ hierarchical
clustering as an additional analysis).

Why directors (not films)?  Film-level attributes are almost all nominal;
K-Means relies on Euclidean distance and centroids (means), which are only
meaningful in a numeric space.  Aggregating each director's films yields a
genuinely numeric profile: productivity, period, colour/TV share, critic
ratings, Oscar rate, cast size and genre proportions.  Identifier columns
are never used as features.

K-Means (Lecture 20):
    1. choose K initial centroids
    2. assignment: each point x goes to the cluster with the nearest centroid,
       Euclidean distance d(x, c) = sqrt(sum_j (x_j - c_j)^2)
    3. update: each centroid becomes the mean of its assigned points
    4. repeat 2-3 until assignments stop changing / centroid shift < tol
    Objective: within-cluster sum of squares  WCSS = sum_k sum_{x in C_k} ||x - c_k||^2

Choice of K: elbow (WCSS), silhouette  s = (b - a) / max(a, b),
Davies-Bouldin (lower is better), Calinski-Harabasz (higher is better).

Run:  python -m src.clustering.kmeans
"""
from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             silhouette_samples, silhouette_score)

from src.config import KMEANS_DIR, KMEANS_K_RANGE, PLOTS_DIR, PROCESSED_DIR, RANDOM_SEED, get_logger
from src.preprocessing.selection import KMEANS_FEATURES
from src.visualization.style import CAT, GREY, INK2, DIV, apply_style, save

log = get_logger("kmeans")


# --------------------------------------------------------------------------
# Manual K-Means (educational implementation)
# --------------------------------------------------------------------------
def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: first centre uniformly, then each next centre with
    probability proportional to D(x)^2 (squared distance to nearest chosen centre)."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centres)[None, :, :]) ** 2).sum(-1), axis=1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(centres)


def manual_kmeans(X: np.ndarray, init_centres: np.ndarray, max_iter: int = 300, tol: float = 1e-6):
    """Lloyd's algorithm.  Returns labels, centroids, wcss, history of WCSS per iteration."""
    C = init_centres.copy()
    history = []
    labels = None
    for it in range(max_iter):
        # assignment step - Euclidean distance to every centroid
        dist = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))
        new_labels = dist.argmin(1)
        wcss = float(((X - C[new_labels]) ** 2).sum())
        history.append(wcss)
        # update step - centroid = mean of assigned points (empty cluster keeps its centre)
        newC = np.array([X[new_labels == j].mean(0) if np.any(new_labels == j) else C[j] for j in range(len(C))])
        shift = float(np.sqrt(((newC - C) ** 2).sum(1)).max())
        C = newC
        if labels is not None and np.array_equal(labels, new_labels) and shift < tol:
            labels = new_labels
            break
        labels = new_labels
    wcss = float(((X - C[labels]) ** 2).sum())
    return labels, C, wcss, history, it + 1


def run() -> dict:
    log.info("PHASE 8b - K-Means started")
    apply_style()
    prof = pd.read_csv(PROCESSED_DIR / "director_profiles.csv")
    Z = pd.read_csv(PROCESSED_DIR / "kmeans_director_matrix_z.csv")
    names = Z.pop("director")
    X = Z[KMEANS_FEATURES].to_numpy()

    # ---- choose K
    rows = []
    for k in KMEANS_K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(X)
        rows.append(dict(k=k, wcss=km.inertia_, silhouette=silhouette_score(X, km.labels_),
                         davies_bouldin=davies_bouldin_score(X, km.labels_),
                         calinski_harabasz=calinski_harabasz_score(X, km.labels_),
                         min_cluster_size=int(np.bincount(km.labels_).min())))
    sel = pd.DataFrame(rows)
    # elbow: largest second difference of WCSS (point of maximum curvature)
    w = sel["wcss"].to_numpy()
    second_diff = np.r_[np.nan, w[:-2] - 2 * w[1:-1] + w[2:], np.nan]
    sel["wcss_second_difference"] = second_diff
    sel.to_csv(KMEANS_DIR / "k_selection_metrics.csv", index=False)
    elbow_k = int(sel.loc[np.nanargmax(second_diff), "k"])
    # decision rule: among K in [3, 6] (interpretable, no tiny clusters) pick the
    # best silhouette; DB/CH reported as corroborating evidence
    cand = sel[(sel["k"] >= 3) & (sel["k"] <= 6) & (sel["min_cluster_size"] >= 20)]
    K = int(cand.sort_values("silhouette", ascending=False).iloc[0]["k"])

    # ---- final model: sklearn and manual implementation from the same seeds
    rng = np.random.default_rng(RANDOM_SEED)
    init = kmeans_plus_plus_init(X, K, rng)
    t0 = time.perf_counter()
    m_labels, m_C, m_wcss, hist, n_iter = manual_kmeans(X, init)
    t_manual = time.perf_counter() - t0
    t0 = time.perf_counter()
    sk = KMeans(n_clusters=K, init=init, n_init=1, random_state=RANDOM_SEED, algorithm="lloyd").fit(X)
    t_sk_same = time.perf_counter() - t0
    best = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_SEED).fit(X)
    agreement = dict(ari_manual_vs_sklearn_same_init=adjusted_rand_score(m_labels, sk.labels_),
                     wcss_manual=m_wcss, wcss_sklearn_same_init=float(sk.inertia_),
                     ari_manual_vs_sklearn_best_of_20=adjusted_rand_score(m_labels, best.labels_),
                     wcss_sklearn_best_of_20=float(best.inertia_), manual_iterations=n_iter,
                     manual_time_s=t_manual, sklearn_time_s=t_sk_same)
    labels = best.labels_
    # relabel clusters by mean year so cluster 0 is the earliest (stable naming)
    order = np.argsort([prof.loc[labels == j, "mean_year"].mean() for j in range(K)])
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    centres_z = pd.DataFrame(best.cluster_centers_[order], columns=KMEANS_FEATURES)
    centres_z.index.name = "cluster"
    centres_z.to_csv(KMEANS_DIR / "cluster_centroids_standardised.csv")

    prof["cluster"] = labels
    prof.to_csv(KMEANS_DIR / "director_cluster_assignments.csv", index=False)
    profile_cols = ["n_films", "first_year", "mean_year", "career_span", "share_color", "share_tv", "mean_halliwell",
                    "share_rated", "oscar_nom_rate", "mean_cast_size", "share_Dram", "share_Comd", "share_Susp",
                    "share_Romt", "share_Horr", "share_Musc", "share_West", "share_Advt", "share_Actn", "share_Docu"]
    profiles = prof.groupby("cluster")[profile_cols].mean().round(3)
    profiles.insert(0, "size", prof["cluster"].value_counts().sort_index())
    examples = prof.sort_values("n_films", ascending=False).groupby("cluster")["director"].apply(lambda s: ", ".join(s.head(6)))
    profiles["example_directors"] = examples
    top_country = prof.groupby("cluster")["director_country"].agg(lambda s: s.value_counts().index[0] if s.notna().any() else "Unknown")
    profiles["most_common_country"] = top_country
    profiles["label"] = [name_cluster(profiles.loc[c]) for c in profiles.index]
    profiles.to_csv(KMEANS_DIR / "cluster_profiles.csv")

    # ---- hierarchical clustering (additional analysis, Lecture 21)
    hier = []
    D = pdist(X, metric="euclidean")
    links = {}
    for method in ("single", "complete", "average", "ward"):
        L = linkage(X, method=method)
        links[method] = L
        coph, _ = cophenet(L, D)
        lab = fcluster(L, K, criterion="maxclust")
        sizes = np.bincount(lab)[1:]
        hier.append(dict(linkage=method, cophenetic_correlation=coph, clusters=K, largest_cluster=int(sizes.max()),
                         smallest_cluster=int(sizes.min()),
                         silhouette=silhouette_score(X, lab) if len(set(lab)) > 1 else np.nan,
                         ari_vs_kmeans=adjusted_rand_score(labels, lab)))
    hier = pd.DataFrame(hier)
    hier.to_csv(KMEANS_DIR / "hierarchical_comparison.csv", index=False)

    figs = _plots(X, sel, K, elbow_k, labels, centres_z, profiles, hist, links, names, prof)
    sil = silhouette_score(X, labels)
    summary = dict(n_directors=len(X), n_features=len(KMEANS_FEATURES), features=KMEANS_FEATURES, chosen_k=K,
                   elbow_k=elbow_k, silhouette=sil, davies_bouldin=davies_bouldin_score(X, labels),
                   calinski_harabasz=calinski_harabasz_score(X, labels), wcss=float(best.inertia_),
                   cluster_sizes=prof["cluster"].value_counts().sort_index().to_dict(),
                   manual_vs_sklearn=agreement, hierarchical=hier.to_dict("records"), figures=figs,
                   labels=profiles["label"].to_dict())
    with open(KMEANS_DIR / "kmeans_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("PHASE 8b complete: K=%d silhouette=%.3f agreement=%s", K, sil, agreement)
    return summary


def name_cluster(p: pd.Series) -> str:
    """Descriptive label derived from the profile (data-driven, not hand-assigned)."""
    era = ("silent/early" if p["mean_year"] < 1935 else "classical" if p["mean_year"] < 1955 else
           "post-studio" if p["mean_year"] < 1970 else "modern")
    genre_cols = [c for c in p.index if c.startswith("share_") and c not in ("share_color", "share_tv", "share_rated")]
    from src.preprocessing.vocabularies import CATEGORY_NAMES
    g = p[genre_cols].astype(float).idxmax().replace("share_", "")
    traits = [era, f"{CATEGORY_NAMES.get(g, g).split('/')[0].lower()}-leaning"]
    if p["share_tv"] > 0.25:
        traits.append("TV-heavy")
    if p["mean_halliwell"] > 1.5:
        traits.append("critically acclaimed")
    if p["n_films"] > 20:
        traits.append("prolific")
    return ", ".join(traits)


def _plots(X, sel, K, elbow_k, labels, centres_z, profiles, hist, links, names, prof):
    out = []
    # elbow + indices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(sel["k"], sel["wcss"], marker="o", color=CAT[0])
    axes[0].axvline(K, color=INK2, ls="--", lw=1)
    axes[0].set_title("Elbow method (WCSS)"); axes[0].set_xlabel("K"); axes[0].set_ylabel("Within-cluster SS")
    axes[1].plot(sel["k"], sel["silhouette"], marker="o", color=CAT[1])
    axes[1].axvline(K, color=INK2, ls="--", lw=1)
    axes[1].set_title("Silhouette (higher = better)"); axes[1].set_xlabel("K"); axes[1].set_ylabel("Mean silhouette")
    axes[2].plot(sel["k"], sel["davies_bouldin"], marker="o", color=CAT[2], label="Davies-Bouldin (lower = better)")
    axes[2].axvline(K, color=INK2, ls="--", lw=1)
    axes[2].set_title("Davies-Bouldin index"); axes[2].set_xlabel("K"); axes[2].set_ylabel("DB index")
    fig.suptitle(f"Choosing K for director profiles (chosen K = {K}, dashed)", fontweight="bold")
    out.append(save(fig, PLOTS_DIR / "29_kmeans_elbow_silhouette_db.png").name)
    # silhouette per sample
    s = silhouette_samples(X, labels)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    y0 = 10
    for j in range(K):
        sj = np.sort(s[labels == j])
        ax.fill_betweenx(np.arange(y0, y0 + len(sj)), 0, sj, color=CAT[j], alpha=0.85, label=f"cluster {j}")
        y0 += len(sj) + 10
    ax.axvline(s.mean(), color=INK2, ls="--", lw=1)
    ax.text(s.mean() + 0.01, 5, f"mean = {s.mean():.3f}", color=INK2, fontsize=9)
    ax.set_title("Silhouette plot of the final K-Means clustering")
    ax.set_xlabel("Silhouette coefficient"); ax.set_ylabel("Directors (sorted within cluster)"); ax.set_yticks([])
    ax.legend(loc="lower right")
    out.append(save(fig, PLOTS_DIR / "30_kmeans_silhouette_plot.png").name)
    # PCA cluster visualisation
    pca = PCA(n_components=2, random_state=RANDOM_SEED).fit(X)
    P = pca.transform(X)
    Cp = pca.transform(centres_z.to_numpy())
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for j in range(K):
        ax.scatter(P[labels == j, 0], P[labels == j, 1], s=22, color=CAT[j], alpha=0.7, edgecolor="white",
                   linewidth=0.4, label=f"C{j}: {profiles.loc[j, 'label']} (n={int(profiles.loc[j, 'size'])})")
    ax.scatter(Cp[:, 0], Cp[:, 1], marker="X", s=220, color="black", edgecolor="white", linewidth=1.5, label="centroids")
    for nm in ["Hitchcock", "Cukor", "J.Ford", "Corman", "Lumet", "I.Bergman", "Griffith", "Henreid"]:
        idx = np.where(names == nm)[0]
        if len(idx):
            ax.annotate(nm, P[idx[0]], fontsize=8.5, color=INK2, xytext=(4, 4), textcoords="offset points")
    ax.set_title("K-Means clusters projected on the first two principal components")
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% variance)")
    ax.legend(fontsize=8, loc="best")
    out.append(save(fig, PLOTS_DIR / "31_kmeans_pca_clusters.png").name)
    # cluster sizes
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([f"C{j}" for j in profiles.index], profiles["size"], color=[CAT[j] for j in profiles.index], width=0.6)
    for j, v in enumerate(profiles["size"]):
        ax.text(j, v + 3, str(int(v)), ha="center", color=INK2)
    ax.set_title("Cluster size distribution"); ax.set_xlabel("Cluster"); ax.set_ylabel("Number of directors")
    out.append(save(fig, PLOTS_DIR / "32_kmeans_cluster_sizes.png").name)
    # centroid heatmap (standardised)
    fig, ax = plt.subplots(figsize=(12, 0.9 * K + 2.5))
    im = ax.imshow(centres_z.to_numpy(), cmap=DIV, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(KMEANS_FEATURES))); ax.set_xticklabels(KMEANS_FEATURES, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"C{j}" for j in range(K)])
    for i in range(K):
        for j in range(len(KMEANS_FEATURES)):
            ax.text(j, i, f"{centres_z.iloc[i, j]:.1f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="centroid z-score")
    ax.set_title("Cluster centroids in standardised feature space"); ax.set_xlabel("Feature"); ax.set_ylabel("Cluster")
    out.append(save(fig, PLOTS_DIR / "33_kmeans_centroid_heatmap.png").name)
    # parallel coordinates of centroids
    feats = ["mean_year", "career_span", "share_color", "share_tv", "mean_halliwell_filled", "oscar_nom_rate",
             "mean_cast_size", "share_Dram", "share_Comd", "share_Susp", "share_Horr", "share_West"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for j in range(K):
        ax.plot(range(len(feats)), centres_z.loc[j, feats], marker="o", color=CAT[j], label=f"C{j}")
    ax.axhline(0, color=GREY, lw=1)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=35, ha="right")
    ax.set_title("Parallel coordinates of cluster centroids (z-scores)")
    ax.set_xlabel("Feature"); ax.set_ylabel("Centroid value (z-score)"); ax.legend(ncol=K)
    out.append(save(fig, PLOTS_DIR / "34_kmeans_parallel_coordinates.png").name)
    # pair plot of key original features coloured by cluster
    import seaborn as sns
    pp = prof[["mean_year", "share_color", "mean_halliwell", "n_films", "cluster"]].copy()
    pp["mean_halliwell"] = pp["mean_halliwell"].fillna(-0.5)
    pp["cluster"] = "C" + pp["cluster"].astype(str)
    g = sns.pairplot(pp, hue="cluster", palette=CAT[:K], hue_order=[f"C{j}" for j in range(K)], corner=True,
                     plot_kws=dict(s=14, alpha=0.6, edgecolor="none"), diag_kind="kde", height=2.3)
    g.figure.suptitle("Pair plot of director profile features by K-Means cluster", y=1.02, fontweight="bold")
    g.savefig(PLOTS_DIR / "35_kmeans_pairplot.png", dpi=130, bbox_inches="tight"); plt.close(g.figure)
    out.append("35_kmeans_pairplot.png")
    # convergence of manual K-Means
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(range(1, len(hist) + 1), hist, marker="o", color=CAT[0])
    ax.set_title("Convergence of the manual K-Means implementation")
    ax.set_xlabel("Iteration"); ax.set_ylabel("WCSS after assignment step")
    out.append(save(fig, PLOTS_DIR / "36_kmeans_manual_convergence.png").name)
    # dendrograms (single / complete / average / ward)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, m in zip(axes.ravel(), ["single", "complete", "average", "ward"]):
        dendrogram(links[m], truncate_mode="lastp", p=30, ax=ax, color_threshold=0, above_threshold_color=CAT[0],
                   show_leaf_counts=True, leaf_font_size=7)
        ax.set_title(f"{m.capitalize()} linkage"); ax.set_xlabel("Cluster (leaf count in brackets)")
        ax.set_ylabel("Merge distance (height)")
    fig.suptitle("Agglomerative hierarchical clustering dendrograms of director profiles (last 30 merges)",
                 fontweight="bold")
    out.append(save(fig, PLOTS_DIR / "37_hierarchical_dendrograms.png").name)
    return out


if __name__ == "__main__":
    run()

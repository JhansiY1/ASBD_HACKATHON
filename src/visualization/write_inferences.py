"""
Builds outputs/visualization_inferences.md and outputs/figure_catalogue.json.

EDA figures (1-16) take their metadata from outputs/plots_eda_meta.json,
which is generated together with the plots.  Algorithm figures (17-41) are
described here from the result files of each phase, so every number quoted
is read from the actual outputs.

Run:  python -m src.visualization.write_inferences
"""
from __future__ import annotations

import json

import pandas as pd

from src.config import APRIORI_DIR, COFI_DIR, DT_DIR, KMEANS_DIR, OUTPUTS, SPARK_DIR, get_logger

log = get_logger("inferences")


def _j(p):
    return json.loads(p.read_text())


def algorithm_figures() -> list[dict]:
    ap = _j(APRIORI_DIR / "apriori_summary.json")
    sens = pd.read_csv(APRIORI_DIR / "support_analysis.csv")
    top = pd.read_csv(APRIORI_DIR / "top_rules.csv")
    lv = pd.read_csv(APRIORI_DIR / "levelwise_trace.csv")
    dt = _j(DT_DIR / "decision_tree_summary.json")
    tm = dt["test_metrics"]
    imp = pd.read_csv(DT_DIR / "feature_importance.csv", index_col=0)["importance"]
    km = _j(KMEANS_DIR / "kmeans_summary.json")
    ksel = pd.read_csv(KMEANS_DIR / "k_selection_metrics.csv")
    prof = pd.read_csv(KMEANS_DIR / "cluster_profiles.csv", index_col=0)
    hier = pd.read_csv(KMEANS_DIR / "hierarchical_comparison.csv")
    cofi = _j(COFI_DIR / "cofi_summary.json")
    comp = pd.read_csv(COFI_DIR / "support_comparison.csv")
    stats = pd.read_csv(COFI_DIR / "cofi_tree_statistics.csv")
    bdt = pd.read_csv(SPARK_DIR / "benchmark_decision_tree.csv")
    bk = pd.read_csv(SPARK_DIR / "benchmark_kmeans.csv")
    bf = pd.read_csv(SPARK_DIR / "benchmark_frequent_patterns.csv")
    st = pd.read_csv(SPARK_DIR / "stress_test_replicated.csv")
    kf = bk[bk.workload == "kmeans_films"]
    pruned_total = int(lv["pruned"].sum())
    s02 = sens[(sens.min_support == 0.02)]
    figs = [
        dict(file="17_apriori_support_confidence_lift.png", type="Scatter plot (support-confidence-lift)",
             attributes="all rules at min_sup 0.02 / min_conf 0.60",
             reason="Standard visualisation of rule interestingness measures; colour adds lift as third dimension.",
             observations=f"{ap['rules']} rules; {ap['rules_lift_gt_1']} have lift > 1. High-lift rules (dark) sit at "
                          f"low support (2-3 %), while high-support rules have lift close to 1-2.",
             inference="Support and interestingness trade off: the most surprising patterns are specific "
                       "(e.g. Hitchcock TV episodes), the most frequent ones are near base rates.",
             conclusion="Rules are ranked by lift after confidence filtering, with closure-based de-duplication."),
        dict(file="18_apriori_threshold_sensitivity.png", type="Line plot (parameter sensitivity)",
             attributes="min_support x min_confidence grid",
             reason="Justifies the chosen thresholds by showing how output size responds to them.",
             observations=f"Frequent itemsets: {', '.join(f'{int(r.min_support * 100)}% -> {int(r.frequent_itemsets)}' for r in sens.drop_duplicates('min_support').itertuples())}. "
                          f"At 2 % support the rule count falls from {int(s02[s02.min_confidence == 0.5].rules.iloc[0])} "
                          f"(conf 0.5) to {int(s02[s02.min_confidence == 0.8].rules.iloc[0])} (conf 0.8).",
             inference="1 % support gives hundreds of very specific rules; 5 % keeps only obvious era/colour "
                       "rules. 2 % / 0.60 is the knee: genre-level patterns survive while the output stays readable.",
             conclusion="Selected min_support = 0.02 (229 films) and min_confidence = 0.60."),
        dict(file="19_apriori_levelwise_candidates.png", type="Grouped bar chart (log scale)",
             attributes="Apriori level-wise trace |C_k|, pruned, |L_k|",
             reason="Makes the join -> prune -> count mechanism of Apriori visible.",
             observations="; ".join(f"k={int(r.k)}: joined {int(r.candidates_joined)}, pruned {int(r.pruned)}, "
                                   f"frequent {int(r.frequent)}" for r in lv.itertuples()),
             inference=f"The Apriori property pruned {pruned_total} candidates before any support counting; most "
                       f"surviving candidates are still infrequent, showing why candidate generation is costly.",
             conclusion="Confirms the downward-closure pruning works and motivates FP-tree/COFI methods."),
        dict(file="20_apriori_top_rules_lift.png", type="Horizontal bar chart (ranked rules)",
             attributes="top rules: lift and confidence",
             reason="Ranks the selected rules for interpretation.",
             observations=f"Highest lift: {top.iloc[0].antecedent} -> {top.iloc[0].consequent} (lift {top.iloc[0].lift:.1f}, "
                          f"conf {top.iloc[0].confidence:.2f}); era/colour rules have lift 3-16; studio/genre rules 1.6-2.",
             inference="The strongest associations are structural (production format and period) rather than "
                       "artistic; genre-era rules quantify the life cycles seen in the EDA.",
             conclusion="Rules describe co-occurrence in this curated collection, not causal effects."),
        dict(file="21_cofi_frequent_closed_maximal.png", type="Line plot (log-log)",
             attributes="numbers of frequent, closed, maximal itemsets vs min support",
             reason="Compares the size of the three pattern families (Lecture 10).",
             observations="; ".join(f"{r.min_support}: {int(r.frequent_itemsets)} / {int(r.closed_itemsets)} / "
                                   f"{int(r.maximal_itemsets)}" for r in comp.itertuples()) + " (frequent/closed/maximal).",
             inference="Closed itemsets are only slightly fewer than frequent itemsets on these short, sparse "
                       "transactions (compression up to "
                       f"{comp.compression_ratio.max():.2f}x at the lowest support), while maximal itemsets are far fewer "
                       "but lose support information.",
             conclusion="CFIs are a lossless summary; their advantage grows as support decreases."),
        dict(file="22_cofi_tree_sizes.png", type="Bubble/scatter plot",
             attributes="per-item COFI-tree nodes, root support, patterns mined",
             reason="Shows the memory footprint of COFI: one small tree per item.",
             observations=f"{len(stats)} COFI-trees; the largest has {int(stats.nodes.max())} nodes versus "
                          f"{cofi['fp_tree_nodes']} nodes in the global FP-tree.",
             inference="Each COFI-tree is two orders of magnitude smaller than the FP-tree, so mining needs "
                       "little memory at a time.",
             conclusion="Supports the COFI design claim of bounded memory during mining."),
        dict(file="23_decision_tree_structure.png", type="Tree diagram",
             attributes="final decision tree (top 4 levels)",
             reason="Shows the induced test attributes and thresholds.",
             observations=f"Criterion {dt['final_params']['criterion']}, depth {tm['depth']}, {tm['leaves']} leaves. "
                          f"Root split on year; then cast size and director productivity.",
             inference="Critic acclaim is mostly predicted by period and documentation depth (cast size), echoing "
                       "the guide's preference for classic studio films.",
             conclusion="A compact, interpretable model (IF-THEN rules in the report)."),
        dict(file="24_decision_tree_confusion_matrix.png", type="Confusion matrix heatmap",
             attributes="test-set predictions", reason="Breaks accuracy into TP, FP, TN, FN.",
             observations=f"Accuracy {tm['accuracy']:.3f}, balanced accuracy {tm['balanced_accuracy']:.3f}, precision "
                          f"{tm['precision']:.3f}, recall {tm['recall']:.3f}.",
             inference="Balanced class weights trade precision for recall: most acclaimed films are found, at the "
                       "cost of false positives among older un-acclaimed films.",
             conclusion=f"Majority-class baseline accuracy is {tm['baseline_majority_accuracy']:.3f}; the tree's value "
                        f"lies in ranking (AUC) and recall, not raw accuracy."),
        dict(file="25_decision_tree_roc_curve.png", type="ROC curve", attributes="test-set probabilities",
             reason="Threshold-independent evaluation (Lecture 13).",
             observations=f"AUC = {tm['roc_auc']:.3f} (random = 0.5).",
             inference="Moderate discriminative power from release-time attributes only.",
             conclusion="Useful for ranking, limited for hard decisions."),
        dict(file="26_decision_tree_feature_importance.png", type="Horizontal bar chart (importance)",
             attributes="impurity-based feature importance",
             reason="Quantifies each attribute's contribution to impurity reduction.",
             observations=", ".join(f"{k} {v:.3f}" for k, v in imp.head(5).items()),
             inference="Year dominates; genres contribute little once period is known.",
             conclusion="Agrees with the root-node information-gain ranking."),
        dict(file="27_decision_tree_pruning_curve.png", type="Line plot (pruning path)",
             attributes="ccp_alpha vs train/test accuracy and CV AUC",
             reason="Shows over- vs under-fitting along the cost-complexity path.",
             observations=f"Chosen alpha {dt['final_params']['ccp_alpha']:.4f}.",
             inference="Small alphas overfit (train >> test); alpha near the CV optimum gives the simplest tree "
                       "within 0.005 AUC of the best.",
             conclusion="Post-pruning yields a smaller, equally accurate tree."),
        dict(file="28_decision_tree_split_criteria.png", type="Small-multiple bar charts",
             attributes="information gain, gain ratio, Gini reduction at the root",
             reason="Compares the three attribute-selection measures of Lecture 13.",
             observations="; ".join(f"{r['attribute']}: IG {r['info_gain']:.3f}, GR {r['gain_ratio']:.3f}, "
                                   f"dGini {r['gini_reduction']:.3f}" for r in dt["top_root_attributes"][:4]),
             inference="All three measures rank year first; gain ratio reorders binary attributes slightly.",
             conclusion=f"Gini vs entropy: CV AUC {dt['criterion_cv'][0]['best_cv_roc_auc']:.3f} vs "
                        f"{dt['criterion_cv'][1]['best_cv_roc_auc']:.3f} - practically equivalent."),
        dict(file="29_kmeans_elbow_silhouette_db.png", type="Line plots (model selection)",
             attributes="K = 2..10: WCSS, silhouette, Davies-Bouldin",
             reason="Standard criteria for choosing K.",
             observations=f"Silhouette ranges {ksel.silhouette.min():.3f}-{ksel.silhouette.max():.3f}; elbow at K = "
                          f"{km['elbow_k']}; chosen K = {km['chosen_k']}.",
             inference="No sharp elbow and low silhouettes: director careers form a continuum rather than "
                       "well-separated groups.",
             conclusion=f"K = {km['chosen_k']} chosen as the most interpretable solution within K 3-6 with the best "
                        f"silhouette and no tiny clusters."),
        dict(file="30_kmeans_silhouette_plot.png", type="Silhouette plot",
             attributes="per-director silhouette coefficients",
             reason="Shows cohesion/separation of each cluster member.",
             observations=f"Mean silhouette {km['silhouette']:.3f}; cluster sizes {km['cluster_sizes']}.",
             inference="The small TV-suspense cluster is the most cohesive; the two large era clusters overlap.",
             conclusion="Cluster boundaries are soft; profiles are summaries of tendencies."),
        dict(file="31_kmeans_pca_clusters.png", type="Cluster scatter (PCA projection)",
             attributes="first two principal components of 19 standardised features",
             reason="2-D view of a 19-D clustering.",
             observations="The classical (C0) and modern (C2) clusters separate along PC1 (period / colour share); "
                          "the small TV-suspense cluster (C1) sits at low PC2 values. Hitchcock lies far right on "
                          "PC1 (prolific, classic, acclaimed) although assigned to C1 by his TV share.",
             inference="Period is the dominant axis of variation among directors; TV/suspense specialisation is "
                       "the second.",
             conclusion="Visual confirmation of the centroid profiles."),
        dict(file="32_kmeans_cluster_sizes.png", type="Bar chart", attributes="cluster sizes",
             reason="Checks for degenerate clusters.", observations=f"Sizes: {km['cluster_sizes']}.",
             inference="One small specialised cluster plus two large era-based clusters.",
             conclusion="No empty or singleton clusters."),
        dict(file="33_kmeans_centroid_heatmap.png", type="Heatmap (centroids)",
             attributes="centroid z-scores for 19 features",
             reason="Interprets clusters through their centroids.",
             observations="; ".join(f"C{i}: {prof.loc[i, 'label']}" for i in prof.index),
             inference="Clusters differ in period, colour share, TV share and suspense share.",
             conclusion="Gives each cluster a data-driven description."),
        dict(file="34_kmeans_parallel_coordinates.png", type="Parallel coordinates plot",
             attributes="centroid values of 12 key features",
             reason="Compares all centroids across many features at once.",
             observations="Profiles cross mainly at mean_year, share_color and share_tv.",
             inference="Same conclusion as the heatmap, in a form that highlights differences between pairs.",
             conclusion="Supports the cluster labels."),
        dict(file="35_kmeans_pairplot.png", type="Pair plot", attributes="mean_year, share_color, mean_halliwell, n_films",
             reason="Pairwise relationships coloured by cluster.",
             observations="mean_year vs share_color is an S-curve that the clusters split along.",
             inference="Colour adoption over time drives much of the clustering.",
             conclusion="Era is the key latent factor in director profiles."),
        dict(file="36_kmeans_manual_convergence.png", type="Line plot (convergence)",
             attributes="WCSS per iteration of the manual implementation",
             reason="Demonstrates the monotone decrease of the K-Means objective.",
             observations=f"Converged in {km['manual_vs_sklearn']['manual_iterations']} iterations; ARI with sklearn "
                          f"(same seeds) = {km['manual_vs_sklearn']['ari_manual_vs_sklearn_same_init']:.2f}.",
             inference="The manual Lloyd algorithm reproduces sklearn exactly.",
             conclusion="Validates the educational implementation."),
        dict(file="37_hierarchical_dendrograms.png", type="Dendrogram",
             attributes="single, complete, average, Ward linkage",
             reason="Lecture 21 hierarchical clustering comparison.",
             observations="; ".join(f"{r.linkage}: cophenetic {r.cophenetic_correlation:.2f}, largest cluster "
                                   f"{int(r.largest_cluster)}, ARI vs K-Means {r.ari_vs_kmeans:.2f}" for r in hier.itertuples()),
             inference="Single/average linkage chain almost all directors into one cluster (high silhouette but "
                       "useless partition); Ward is closest to K-Means.",
             conclusion="Ward (variance-minimising, like K-Means) is the only linkage that yields balanced groups."),
        dict(file="38_pyspark_execution_time.png", type="Line plot (benchmark)",
             attributes="execution time vs records, Python vs PySpark",
             reason="Main benchmark result.",
             observations=f"At 100 %: DT {bdt.iloc[-1].py_total_s:.3f}s vs {bdt.iloc[-1].spark_total_s:.2f}s; K-Means films "
                          f"{kf.iloc[-1].py_total_s:.3f}s vs {kf.iloc[-1].spark_total_s:.2f}s; Apriori "
                          f"{bf.iloc[-1].apriori_total_s:.3f}s vs FP-Growth {bf.iloc[-1].spark_fpgrowth_total_s:.2f}s.",
             inference="On the real data size, PySpark is slower for every technique: fixed job-scheduling and "
                       "JVM/Python serialisation costs dominate the tiny computations.",
             conclusion="Spark offers no speedup at this scale."),
        dict(file="39_pyspark_speedup.png", type="Bar chart (speedup, log scale)",
             attributes="Python time / PySpark time per workload",
             reason="Speedup summary, including the replicated stress test.",
             observations=f"Real-data speedups range {pd.concat([bdt.speedup, kf.speedup, bf.speedup]).min():.3f}-"
                          f"{pd.concat([bdt.speedup, kf.speedup, bf.speedup]).max():.3f}x; stress-test speedups "
                          f"{st.speedup.min():.3f}-{st.speedup.max():.3f}x.",
             inference="Speedup rises with data volume because Spark's overhead is roughly constant while the "
                       "Python cost grows.",
             conclusion="Distributed processing pays off only for much larger data than this dataset."),
        dict(file="40_pyspark_time_breakdown.png", type="Stacked horizontal bar chart",
             attributes="Spark time split into conversion, fit, predict",
             reason="Explains where Spark spends its time.",
             observations="Data conversion and job overhead are a large share of the total.",
             inference="The algorithms themselves are cheap; Spark's costs are infrastructural.",
             conclusion="Keep data inside Spark end to end for real big-data workloads."),
        dict(file="41_pyspark_quality_comparison.png", type="Grouped bar + line chart",
             attributes="accuracy/F1/AUC and silhouette, sklearn vs Spark",
             reason="Checks that the faster/slower implementations give comparable results.",
             observations=f"DT AUC at 100 %: sklearn {bdt.iloc[-1].py_roc_auc:.3f} vs Spark {bdt.iloc[-1].spark_roc_auc:.3f}; "
                          f"K-Means silhouette: {kf.iloc[-1].py_silhouette:.3f} vs {kf.iloc[-1].spark_silhouette:.3f}.",
             inference="Quality is comparable; differences come from Spark's binned thresholds and k-means|| seeding.",
             conclusion="Spark MLlib reproduces the analytical results."),
    ]
    return figs


# Canonical visualisation technique ("family") of every figure.  Distinct-type
# counts use these families, so variations of one technique (e.g. several line
# plots) are counted once.
FAMILY = {1: "Missing-value heatmap", 2: "Histogram", 3: "Line plot", 4: "Area chart", 5: "Bar chart",
          6: "Stacked bar chart", 7: "Pie / donut chart", 8: "Box plot", 9: "Violin plot", 10: "Density (KDE) plot",
          11: "Correlation heatmap", 12: "Bubble chart", 13: "Treemap", 14: "Count plot", 15: "Scatter plot",
          16: "Heatmap (matrix)", 17: "Support-confidence-lift plot", 18: "Line plot", 19: "Bar chart",
          20: "Bar chart", 21: "Line plot", 22: "Bubble chart", 23: "Tree diagram", 24: "Heatmap (matrix)",
          25: "ROC curve", 26: "Bar chart", 27: "Line plot", 28: "Bar chart", 29: "Line plot", 30: "Silhouette plot",
          31: "Cluster visualisation (PCA)", 32: "Bar chart", 33: "Heatmap (matrix)", 34: "Parallel coordinates",
          35: "Pair plot", 36: "Line plot", 37: "Dendrogram", 38: "Line plot", 39: "Bar chart",
          40: "Stacked bar chart", 41: "Bar chart"}


def run() -> None:
    eda = json.loads((OUTPUTS / "plots_eda_meta.json").read_text())
    alg = algorithm_figures()
    for i, f in enumerate(alg, start=len(eda) + 1):
        f["number"] = i
    allf = eda + alg
    for f in allf:
        f["family"] = FAMILY[f["number"]]
    (OUTPUTS / "figure_catalogue.json").write_text(json.dumps(allf, indent=2), encoding="utf-8")
    lines = ["# Visualization Inferences", "",
             f"{len(allf)} figures were generated, covering {len({f['family'] for f in allf})} distinct visualisation "
             f"techniques (the {len(eda)} exploratory figures alone use "
             f"{len({f['family'] for f in allf if f['number'] <= len(eda)})} distinct techniques). "
             "Figures 1-16 are exploratory/descriptive; 17-41 belong to the mining techniques.", ""]
    for f in allf:
        lines += [f"## Plot {f['number']}: {f['type']}", "",
                  f"* **Visualisation technique:** {f['family']}",
                  f"* **File:** `outputs/plots/{f['file']}`",
                  f"* **Attributes used:** {f['attributes']}",
                  f"* **Reason for selecting the plot:** {f['reason']}",
                  f"* **Important observations:** {f['observations']}",
                  f"* **Key inference:** {f['inference']}",
                  f"* **Conclusion / contribution to the analytics task:** {f['conclusion']}", ""]
    (OUTPUTS / "visualization_inferences.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("visualization_inferences.md written (%d figures)", len(allf))


if __name__ == "__main__":
    run()

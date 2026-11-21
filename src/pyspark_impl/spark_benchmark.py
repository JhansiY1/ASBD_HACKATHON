"""
PHASES 10-11 - PySpark implementations and Python-vs-PySpark benchmarking.

Techniques implemented in PySpark (Spark MLlib, local mode)
-----------------------------------------------------------
T1  Decision Tree classification   pyspark.ml.classification.DecisionTreeClassifier
T2  K-Means clustering             pyspark.ml.clustering.KMeans
S   (supplementary) frequent-pattern mining with pyspark.ml.fpm.FPGrowth.
    NOTE: Spark MLlib provides FP-Growth, NOT Apriori.  The two algorithms
    differ (FP-Growth builds a compressed FP-tree and grows patterns without
    candidate generation; Apriori is level-wise with candidate generation and
    one database scan per level) but they are *exact* for the same minimum
    support, so the frequent-itemset OUTPUT can be compared like-for-like.
    Spark's associationRules only produces rules with a single-item
    consequent, so rule counts are compared against the subset of our rules
    with |consequent| = 1.

Comparable configurations
-------------------------
Decision tree : entropy impurity, max depth 6, min 40 samples per leaf,
                balanced class weights (sklearn class_weight / Spark weightCol),
                identical stratified samples and 75/25 split per data size.
                Spark discretises continuous features into maxBins=256 candidate
                thresholds; sklearn evaluates every midpoint - small accuracy
                differences are expected.
K-Means       : same K, k-means++ (sklearn) vs k-means|| (Spark, its scalable
                k-means++ variant), max 100 iterations, tol 1e-4, one init.
                Quality (WCSS, silhouette) is computed with the same NumPy /
                sklearn code on both label sets.

Timing methodology
------------------
* Spark session start-up time is reported once and separately.
* One warm-up run per Spark workload (JIT / class loading) is discarded.
* Each measurement is the median of 3 repetitions (time.perf_counter).
* Spark time is split into: data conversion (pandas -> Spark DataFrame via
  Arrow, vector assembly, cache + count), fit, predict (transform + collect).
* Speedup = Python time / Spark time;  improvement % = (Py - Spark) / Py x 100.
* Memory: tracemalloc peak for the Python implementations; resident set size
  of the Spark JVM (psutil) after the workload.

Data sizes: 10 %, 25 %, 50 %, 100 % of the REAL data (random samples, seed 42).
A separate, clearly labelled stress test replicates the real rows x5 and x20
to study scaling only; replicated rows are not new information and are never
used for any analytical conclusion.

Run (from the space-free path):  cd D:\\ASBD_Hackathon && .venv\\Scripts\\python -m src.pyspark_impl.spark_benchmark
"""
from __future__ import annotations

import json
import statistics
import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans as SkKMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SkDT

from src.association_rules.apriori import apriori, generate_rules
from src.association_rules.run_apriori import load_transactions
from src.config import (APRIORI_MIN_CONFIDENCE, APRIORI_MIN_SUPPORT, PLOTS_DIR, PROCESSED_DIR, RANDOM_SEED,
                        SPARK_DIR, get_logger)
from src.pyspark_impl.spark_session import SPARK_CONF, build_spark, jvm_rss_mb, python_rss_mb
from src.visualization.style import CAT, GREY, INK2, apply_style, save

log = get_logger("spark_benchmark")
FRACTIONS = [0.10, 0.25, 0.50, 1.00]
REPEATS = 3
DT_PARAMS = dict(max_depth=6, min_leaf=40, max_bins=256)
KM_K_DIRECTORS = 3
KM_K_MOVIES = 6


def timed(fn, repeats=REPEATS):
    """Run fn() `repeats` times; return (median seconds, last result)."""
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def py_peak_mb(fn):
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def balanced_weights(y):
    n, pos = len(y), y.sum()
    w1, w0 = n / (2 * pos), n / (2 * (n - pos))
    return np.where(y == 1, w1, w0)


# =========================================================================
# T1 Decision tree
# =========================================================================
def bench_decision_tree(spark, Xall: pd.DataFrame, yall: np.ndarray, fractions, label="real"):
    from pyspark.ml.classification import DecisionTreeClassifier as SpDT
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    rows = []
    feats = list(Xall.columns)
    for frac in fractions:
        if frac < 1:
            X, _, y, _ = train_test_split(Xall, yall, train_size=frac, stratify=yall, random_state=RANDOM_SEED)
        else:
            X, y = Xall, yall
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)
        # ---- Python / scikit-learn
        sk = SkDT(criterion="entropy", max_depth=DT_PARAMS["max_depth"], min_samples_leaf=DT_PARAMS["min_leaf"],
                  class_weight="balanced", random_state=RANDOM_SEED)
        t_fit, _ = timed(lambda: sk.fit(Xtr, ytr))
        t_pred, prob = timed(lambda: sk.predict_proba(Xte)[:, 1])
        mem_py = py_peak_mb(lambda: SkDT(criterion="entropy", max_depth=6, min_samples_leaf=40,
                                         class_weight="balanced").fit(Xtr, ytr))
        pred = (prob >= 0.5).astype(int)
        # ---- Spark MLlib
        tr = Xtr.copy(); tr["label"] = ytr.astype(float); tr["w"] = balanced_weights(ytr)
        te = Xte.copy(); te["label"] = yte.astype(float)
        asm = VectorAssembler(inputCols=feats, outputCol="features")

        def convert():
            a = asm.transform(spark.createDataFrame(tr)).select("features", "label", "w").repartition(2).cache()
            b = asm.transform(spark.createDataFrame(te)).select("features", "label").repartition(2).cache()
            a.count(); b.count()
            return a, b
        t_conv, (sdf_tr, sdf_te) = timed(convert)
        sp = SpDT(featuresCol="features", labelCol="label", weightCol="w", impurity="entropy",
                  maxDepth=DT_PARAMS["max_depth"], minInstancesPerNode=DT_PARAMS["min_leaf"],
                  maxBins=DT_PARAMS["max_bins"], seed=RANDOM_SEED)
        sp.fit(sdf_tr)                                     # warm-up (discarded)
        t_sfit, model = timed(lambda: sp.fit(sdf_tr))
        t_spred, sp_out = timed(lambda: model.transform(sdf_te).select(
            vector_to_array("probability").getItem(1).alias("p1"), "label").toPandas())
        sp_prob = sp_out["p1"].to_numpy()
        sp_pred = (sp_prob >= 0.5).astype(int)
        sp_y = sp_out["label"].astype(int).to_numpy()
        rows.append(dict(workload=f"decision_tree_{label}", fraction=frac, records=len(X), train=len(Xtr),
                         test=len(Xte), features=len(feats), partitions=sdf_tr.rdd.getNumPartitions(),
                         py_fit_s=t_fit, py_predict_s=t_pred, py_total_s=t_fit + t_pred, py_peak_mem_mb=mem_py,
                         spark_convert_s=t_conv, spark_fit_s=t_sfit, spark_predict_s=t_spred,
                         spark_total_s=t_sfit + t_spred, spark_total_incl_convert_s=t_conv + t_sfit + t_spred,
                         spark_jvm_rss_mb=jvm_rss_mb(),
                         py_accuracy=accuracy_score(yte, pred), spark_accuracy=accuracy_score(sp_y, sp_pred),
                         py_f1=f1_score(yte, pred), spark_f1=f1_score(sp_y, sp_pred),
                         py_roc_auc=roc_auc_score(yte, prob), spark_roc_auc=roc_auc_score(sp_y, sp_prob),
                         py_leaves=int(sk.get_n_leaves()), spark_nodes=int(model.numNodes),
                         spark_depth=int(model.depth)))
        sdf_tr.unpersist(); sdf_te.unpersist()
        log.info("DT %s frac=%.2f done: py %.4fs spark %.4fs", label, frac, t_fit + t_pred, t_sfit + t_spred)
    return pd.DataFrame(rows)


# =========================================================================
# T2 K-Means
# =========================================================================
def wcss(X, labels):
    return float(sum(((X[labels == j] - X[labels == j].mean(0)) ** 2).sum() for j in np.unique(labels)))


def bench_kmeans(spark, Xall: np.ndarray, k: int, fractions, name: str):
    from pyspark.ml.clustering import KMeans as SpKMeans
    from pyspark.ml.feature import VectorAssembler
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    cols = [f"f{i}" for i in range(Xall.shape[1])]
    for frac in fractions:
        idx = rng.permutation(len(Xall))[: int(round(frac * len(Xall)))] if frac < 1 else np.arange(len(Xall))
        X = Xall[idx]
        sk = SkKMeans(n_clusters=k, init="k-means++", n_init=1, max_iter=100, tol=1e-4, random_state=RANDOM_SEED)
        t_fit, _ = timed(lambda: sk.fit(X))
        t_pred, lab_py = timed(lambda: sk.predict(X))
        mem_py = py_peak_mb(lambda: SkKMeans(n_clusters=k, n_init=1, max_iter=100, random_state=1).fit(X))
        pdf = pd.DataFrame(X, columns=cols)
        asm = VectorAssembler(inputCols=cols, outputCol="features")

        def convert():
            d = asm.transform(spark.createDataFrame(pdf)).select("features").repartition(2).cache()
            d.count()
            return d
        t_conv, sdf = timed(convert)
        sp = SpKMeans(k=k, initMode="k-means||", maxIter=100, tol=1e-4, seed=RANDOM_SEED, featuresCol="features")
        sp.fit(sdf)                                         # warm-up
        t_sfit, model = timed(lambda: sp.fit(sdf))
        # predictions are collected in the same row order as `pdf` (monotonic id)
        from pyspark.sql import functions as F
        sdf_ordered = asm.transform(spark.createDataFrame(pdf.assign(_rid=np.arange(len(pdf))))).select("features", "_rid")
        t_spred, lab_sp = timed(lambda: model.transform(sdf_ordered).select("_rid", "prediction")
                                .toPandas().sort_values("_rid")["prediction"].to_numpy())
        n_sil = min(len(X), 4000)
        sil_idx = np.random.default_rng(1).choice(len(X), n_sil, replace=False)
        from sklearn.metrics import adjusted_rand_score
        rows.append(dict(workload=f"kmeans_{name}", fraction=frac, records=len(X), features=X.shape[1], k=k,
                         partitions=sdf.rdd.getNumPartitions(), py_fit_s=t_fit, py_predict_s=t_pred,
                         py_total_s=t_fit + t_pred, py_peak_mem_mb=mem_py, spark_convert_s=t_conv,
                         spark_fit_s=t_sfit, spark_predict_s=t_spred, spark_total_s=t_sfit + t_spred,
                         spark_total_incl_convert_s=t_conv + t_sfit + t_spred, spark_jvm_rss_mb=jvm_rss_mb(),
                         py_wcss=wcss(X, lab_py), spark_wcss=wcss(X, lab_sp),
                         py_silhouette=silhouette_score(X[sil_idx], lab_py[sil_idx]),
                         spark_silhouette=silhouette_score(X[sil_idx], lab_sp[sil_idx]),
                         ari_py_vs_spark=adjusted_rand_score(lab_py, lab_sp), py_iterations=int(sk.n_iter_),
                         spark_iterations=int(model.summary.numIter)))
        sdf.unpersist()
        log.info("KMeans %s frac=%.2f done", name, frac)
    return pd.DataFrame(rows)


# =========================================================================
# S  Frequent patterns: Apriori (ours), mlxtend, Spark FP-Growth
# =========================================================================
def bench_fpm(spark, tx_all, fractions, label="real"):
    from mlxtend.frequent_patterns import fpgrowth as mx_fpgrowth
    from mlxtend.preprocessing import TransactionEncoder
    from pyspark.ml.fpm import FPGrowth
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for frac in fractions:
        idx = rng.permutation(len(tx_all))[: int(round(frac * len(tx_all)))] if frac < 1 else np.arange(len(tx_all))
        tx = [tx_all[i] for i in idx]
        t_ap, res = timed(lambda: apriori(tx, APRIORI_MIN_SUPPORT))
        t_rules, rules = timed(lambda: generate_rules(res, APRIORI_MIN_CONFIDENCE))
        mem_py = py_peak_mb(lambda: apriori(tx, APRIORI_MIN_SUPPORT))
        te = TransactionEncoder()
        Xb = pd.DataFrame(te.fit(tx).transform(tx), columns=te.columns_)
        t_mx, _ = timed(lambda: mx_fpgrowth(Xb, min_support=APRIORI_MIN_SUPPORT, use_colnames=True))
        pdf = pd.DataFrame({"items": tx})

        def convert():
            d = spark.createDataFrame(pdf).repartition(2).cache()
            d.count()
            return d
        t_conv, sdf = timed(convert)
        fp = FPGrowth(itemsCol="items", minSupport=APRIORI_MIN_SUPPORT, minConfidence=APRIORI_MIN_CONFIDENCE)
        fp.fit(sdf).freqItemsets.count()                   # warm-up

        def spark_run():
            m = fp.fit(sdf)
            fi = m.freqItemsets.toPandas()
            ar = m.associationRules.toPandas()
            return fi, ar
        t_sp, (fi, ar) = timed(spark_run)
        sp_sets = {frozenset(i): int(f) for i, f in zip(fi["items"], fi["freq"])}
        ours_single = rules[rules["consequent_len"] == 1]
        rows.append(dict(workload=f"frequent_patterns_{label}", fraction=frac, transactions=len(tx),
                         partitions=sdf.rdd.getNumPartitions(), apriori_s=t_ap, apriori_rules_s=t_rules,
                         apriori_total_s=t_ap + t_rules, apriori_peak_mem_mb=mem_py, mlxtend_fpgrowth_s=t_mx,
                         spark_convert_s=t_conv, spark_fpgrowth_total_s=t_sp,
                         spark_total_incl_convert_s=t_conv + t_sp, spark_jvm_rss_mb=jvm_rss_mb(),
                         apriori_itemsets=len(res.itemsets), spark_itemsets=len(sp_sets),
                         identical_itemsets_and_counts=sp_sets == res.itemsets,
                         apriori_rules_single_consequent=len(ours_single), spark_rules=len(ar)))
        sdf.unpersist()
        log.info("FPM %s frac=%.2f done", label, frac)
    return pd.DataFrame(rows)


def add_speedup(df, py_col, sp_col, prefix="speedup"):
    df[f"{prefix}"] = df[py_col] / df[sp_col]
    df[f"{prefix}_improvement_pct"] = (df[py_col] - df[sp_col]) / df[py_col] * 100
    return df


# =========================================================================
def run() -> dict:
    log.info("PHASE 10-11 - PySpark benchmarking started")
    apply_style()
    import psutil, platform
    env = dict(cpu="Intel Core i5-1235U (10 cores / 12 threads)", logical_cpus=psutil.cpu_count(),
               ram_gb=round(psutil.virtual_memory().total / 1e9, 1), os=platform.platform(),
               python=platform.python_version(), spark_conf=SPARK_CONF, repeats=REPEATS, fractions=FRACTIONS)
    rss0 = python_rss_mb()
    spark, startup = build_spark("ASBD-Benchmark")
    env.update(spark_version=spark.version, spark_startup_s=startup, python_rss_before_spark_mb=rss0)
    try:
        # ---- data
        dt = pd.read_csv(PROCESSED_DIR / "dt_ml_matrix.csv")
        dt.pop("film_id")
        ydt = dt.pop("acclaimed").astype(int).to_numpy()
        Zd = pd.read_csv(PROCESSED_DIR / "kmeans_director_matrix_z.csv").drop(columns="director").to_numpy()
        Zm = pd.read_csv(PROCESSED_DIR / "kmeans_movie_matrix.csv").to_numpy()
        tx = load_transactions()

        # ---- real-data benchmarks
        b_dt = add_speedup(bench_decision_tree(spark, dt, ydt, FRACTIONS), "py_total_s", "spark_total_s")
        b_dt = add_speedup(b_dt, "py_total_s", "spark_total_incl_convert_s", "speedup_incl_convert")
        b_dt.to_csv(SPARK_DIR / "benchmark_decision_tree.csv", index=False)
        b_kd = bench_kmeans(spark, Zd, KM_K_DIRECTORS, FRACTIONS, "directors")
        b_km = bench_kmeans(spark, Zm, KM_K_MOVIES, FRACTIONS, "films")
        b_k = add_speedup(pd.concat([b_kd, b_km], ignore_index=True), "py_total_s", "spark_total_s")
        b_k = add_speedup(b_k, "py_total_s", "spark_total_incl_convert_s", "speedup_incl_convert")
        b_k.to_csv(SPARK_DIR / "benchmark_kmeans.csv", index=False)
        b_f = add_speedup(bench_fpm(spark, tx, FRACTIONS), "apriori_total_s", "spark_fpgrowth_total_s")
        b_f.to_csv(SPARK_DIR / "benchmark_frequent_patterns.csv", index=False)

        # ---- stress test by replication (scaling study only)
        rep_rows = []
        for factor in (5, 20):
            dt_rep = pd.concat([dt] * factor, ignore_index=True)
            y_rep = np.tile(ydt, factor)
            r1 = bench_decision_tree(spark, dt_rep, y_rep, [1.0], label=f"replicated_x{factor}")
            r2 = bench_kmeans(spark, np.tile(Zm, (factor, 1)), KM_K_MOVIES, [1.0], f"films_replicated_x{factor}")
            r3 = bench_fpm(spark, tx * factor, [1.0], label=f"replicated_x{factor}")
            for r, py, sp in ((r1, "py_total_s", "spark_total_s"), (r2, "py_total_s", "spark_total_s"),
                              (r3, "apriori_total_s", "spark_fpgrowth_total_s")):
                r = add_speedup(r, py, sp)
                r["replication_factor"] = factor
                rep_rows.append(r)
        stress = pd.concat(rep_rows, ignore_index=True)
        stress.to_csv(SPARK_DIR / "stress_test_replicated.csv", index=False)
        env["spark_jvm_rss_end_mb"] = jvm_rss_mb()
    finally:
        spark.stop()
    env["python_rss_end_mb"] = python_rss_mb()
    with open(SPARK_DIR / "spark_configuration.json", "w") as fh:
        json.dump(env, fh, indent=2, default=str)

    comp = comparison_table(b_dt, b_k, b_f)
    comp.to_csv(SPARK_DIR / "performance_comparison_summary.csv", index=False)
    figs = plots(b_dt, b_k, b_f, stress)
    summary = dict(environment=env, figures=figs, comparison=comp.to_dict("records"))
    with open(SPARK_DIR / "benchmark_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    log.info("PHASE 10-11 complete")
    return summary


def comparison_table(b_dt, b_k, b_f):
    rows = []
    for _, r in b_dt.iterrows():
        rows.append(dict(technique="Decision tree", workload="labelled films", fraction=r.fraction, records=r.records,
                         python_s=r.py_total_s, pyspark_s=r.spark_total_s, pyspark_incl_convert_s=r.spark_total_incl_convert_s,
                         speedup=r.speedup, improvement_pct=r.speedup_improvement_pct,
                         python_quality=f"acc {r.py_accuracy:.3f} / F1 {r.py_f1:.3f} / AUC {r.py_roc_auc:.3f}",
                         pyspark_quality=f"acc {r.spark_accuracy:.3f} / F1 {r.spark_f1:.3f} / AUC {r.spark_roc_auc:.3f}"))
    for _, r in b_k.iterrows():
        rows.append(dict(technique="K-Means", workload=r.workload.replace("kmeans_", ""), fraction=r.fraction,
                         records=r.records, python_s=r.py_total_s, pyspark_s=r.spark_total_s,
                         pyspark_incl_convert_s=r.spark_total_incl_convert_s, speedup=r.speedup,
                         improvement_pct=r.speedup_improvement_pct,
                         python_quality=f"WCSS {r.py_wcss:.0f} / sil {r.py_silhouette:.3f}",
                         pyspark_quality=f"WCSS {r.spark_wcss:.0f} / sil {r.spark_silhouette:.3f}"))
    for _, r in b_f.iterrows():
        rows.append(dict(technique="Frequent patterns (Apriori vs Spark FP-Growth)", workload="film transactions",
                         fraction=r.fraction, records=r.transactions, python_s=r.apriori_total_s,
                         pyspark_s=r.spark_fpgrowth_total_s, pyspark_incl_convert_s=r.spark_total_incl_convert_s,
                         speedup=r.speedup, improvement_pct=r.speedup_improvement_pct,
                         python_quality=f"{r.apriori_itemsets} itemsets / {r.apriori_rules_single_consequent} rules",
                         pyspark_quality=f"{r.spark_itemsets} itemsets / {r.spark_rules} rules"))
    return pd.DataFrame(rows)


def plots(b_dt, b_k, b_f, stress):
    out = []
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = [(b_dt, "records", "py_total_s", "spark_total_s", "Decision tree (fit + predict)"),
              (b_k[b_k.workload == "kmeans_films"], "records", "py_total_s", "spark_total_s", "K-Means on films (fit + predict)"),
              (b_f, "transactions", "apriori_total_s", "spark_fpgrowth_total_s", "Apriori (ours) vs Spark FP-Growth")]
    for ax, (d, x, py, sp, title) in zip(axes, panels):
        ax.plot(d[x], d[py], marker="o", color=CAT[0], label="Python")
        ax.plot(d[x], d[sp], marker="o", color=CAT[1], label="PySpark (fit + predict)")
        if "spark_total_incl_convert_s" in d:
            ax.plot(d[x], d["spark_total_incl_convert_s"], marker="o", ls="--", color=CAT[1], alpha=0.6,
                    label="PySpark incl. data conversion")
        ax.set_title(title); ax.set_xlabel("Records"); ax.set_ylabel("Median time (s)"); ax.set_yscale("log")
        ax.legend(fontsize=8)
    fig.suptitle("Execution time on 10/25/50/100 % of the real data (local[2])", fontweight="bold")
    out.append(save(fig, PLOTS_DIR / "38_pyspark_execution_time.png").name)
    # speedup bars
    comp = pd.concat([
        pd.DataFrame({"label": [f"DT {int(100 * f)}%" for f in b_dt.fraction], "speedup": b_dt.speedup}),
        pd.DataFrame({"label": [f"KM films {int(100 * f)}%" for f in b_k[b_k.workload == 'kmeans_films'].fraction],
                      "speedup": b_k[b_k.workload == 'kmeans_films'].speedup}),
        pd.DataFrame({"label": [f"FPM {int(100 * f)}%" for f in b_f.fraction], "speedup": b_f.speedup}),
        pd.DataFrame({"label": [f"{w.split('_')[0].upper()} x{int(r)}" for w, r in zip(stress.workload, stress.replication_factor)],
                      "speedup": stress.speedup})], ignore_index=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [CAT[0] if s >= 1 else CAT[1] for s in comp.speedup]
    ax.bar(comp.label, comp.speedup, color=colors, width=0.65)
    ax.axhline(1, color=INK2, ls="--", lw=1)
    for i, s in enumerate(comp.speedup):
        ax.text(i, s * 1.08, f"{s:.2f}x", ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_title("Speedup = Python time / PySpark time (>1: Spark faster; x5/x20 = replicated stress test)")
    ax.set_xlabel("Workload"); ax.set_ylabel("Speedup (log scale)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    out.append(save(fig, PLOTS_DIR / "39_pyspark_speedup.png").name)
    # time breakdown of Spark at 100 %
    full = pd.DataFrame([
        dict(w="Decision tree", conv=b_dt.iloc[-1].spark_convert_s, fit=b_dt.iloc[-1].spark_fit_s, pred=b_dt.iloc[-1].spark_predict_s),
        dict(w="K-Means films", conv=b_k[b_k.workload == 'kmeans_films'].iloc[-1].spark_convert_s,
             fit=b_k[b_k.workload == 'kmeans_films'].iloc[-1].spark_fit_s, pred=b_k[b_k.workload == 'kmeans_films'].iloc[-1].spark_predict_s),
        dict(w="FP-Growth", conv=b_f.iloc[-1].spark_convert_s, fit=b_f.iloc[-1].spark_fpgrowth_total_s, pred=0.0)])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(full))
    for col, c, lab in [("conv", GREY, "data conversion (pandas -> Spark)"), ("fit", CAT[0], "fit / mine"),
                        ("pred", CAT[2], "predict + collect")]:
        ax.barh(full.w, full[col], left=left, color=c, label=lab, edgecolor="white", linewidth=1.5, height=0.55)
        left += full[col].to_numpy()
    ax.set_title("Where PySpark time goes (100 % real data)"); ax.set_xlabel("Seconds"); ax.set_ylabel("Workload")
    ax.legend(loc="lower right")
    out.append(save(fig, PLOTS_DIR / "40_pyspark_time_breakdown.png").name)
    # quality comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(b_dt)); w = 0.2
    for i, (m, lab) in enumerate([("accuracy", "accuracy"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]):
        axes[0].bar(x + (i - 1) * w - w / 2, b_dt[f"py_{m}"], w, color=CAT[i], label=f"sklearn {lab}")
        axes[0].bar(x + (i - 1) * w + w / 2, b_dt[f"spark_{m}"], w, color=CAT[i], alpha=0.5, hatch="//",
                    label=f"Spark {lab}")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{int(100 * f)}%" for f in b_dt.fraction])
    axes[0].set_ylim(0, 1); axes[0].set_title("Decision tree quality: sklearn vs Spark")
    axes[0].set_xlabel("Data fraction"); axes[0].set_ylabel("Score"); axes[0].legend(fontsize=7, ncol=2)
    kf = b_k[b_k.workload == "kmeans_films"]
    axes[1].plot(kf.records, kf.py_silhouette, marker="o", color=CAT[0], label="sklearn silhouette")
    axes[1].plot(kf.records, kf.spark_silhouette, marker="s", color=CAT[1], label="Spark silhouette")
    axes[1].set_title("K-Means (films) silhouette: sklearn vs Spark"); axes[1].set_xlabel("Records")
    axes[1].set_ylabel("Silhouette"); axes[1].legend()
    out.append(save(fig, PLOTS_DIR / "41_pyspark_quality_comparison.png").name)
    return out


if __name__ == "__main__":
    run()

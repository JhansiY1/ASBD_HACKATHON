"""
Generate the eight phase notebooks in notebooks/ (and optionally execute them).

    python -m src.utils.build_notebooks            # write notebooks
    python -m src.utils.build_notebooks --execute  # write and execute in place

Each notebook calls the same modules as run_all.py, so notebooks and
scripts can never diverge; the notebooks add narrative and inline display.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks"

SETUP = """import sys, os, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import pandas as pd
from IPython.display import Image, display, Markdown
pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 90)
def show(f, w=900): display(Image(filename=str(ROOT / 'outputs' / 'plots' / f), width=w))"""


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


NOTEBOOKS = {
    "01_dataset_exploration.ipynb": [
        md("# 01 - Dataset exploration and profiling\nUCI *Movie* dataset (id 132, Wiederhold). Raw files are read "
           "without modification to understand formats, encodings, schema, size and quality."),
        code(SETUP),
        code("from src.data_loading.html_table_loader import file_inventory, read_table\n"
             "pd.DataFrame(file_inventory())"),
        md("## Raw record format\nEach record is one `<tr><td> ... <td>|` line; the number of cells per row reveals "
           "malformed records."),
        code("main = read_table('main.html')\nprint(len(main), 'MAIN rows')\n"
             "print(pd.Series([len(r.fields) for r in main]).value_counts().sort_index())\n"
             "for r in main[:5]: print(r.fields)"),
        code("from src.preprocessing.pipeline import raw_frames, quality_summary, PLACEHOLDERS\n"
             "raws = raw_frames()\n{k: v.shape for k, v in raws.items()}"),
        code("quality_summary('MAIN', raws['MAIN'].drop(columns='_n_fields'), PLACEHOLDERS['MAIN'])"),
        code("pd.read_csv('outputs/data_dictionary.csv')"),
        code("print(open('outputs/dataset_overview.txt', encoding='utf-8').read())"),
    ],
    "02_preprocessing.ipynb": [
        md("# 02 - Preprocessing (cleaning, integration, selection, transformation)"),
        code(SETUP),
        code("from src.preprocessing import pipeline\nstats = pipeline.run()\nstats"),
        md("## Cleaning log - every operation and the records it affected"),
        code("pd.read_csv('outputs/cleaning_log.csv')"),
        md("## Outliers (flagged, not deleted) and smoothing"),
        code("pd.read_csv('outputs/outlier_summary.csv')"),
        code("sm = pd.read_csv('data/processed/length_min_binning_smoothing.csv', index_col=0)\n"
             "sm.groupby('bin').agg(n=('original','size'), lo=('original','min'), hi=('original','max'), mean=('by_mean','first'))"),
        md("## Attribute selection"),
        code("pd.read_csv('outputs/data_selection.csv')"),
        md("## Transformed outputs"),
        code("pd.read_csv('data/processed/films_transformed.csv').head()"),
        code("pd.read_csv('data/transactions/film_transactions.csv').head(10)"),
        code("print(open('outputs/preprocessing_report.md', encoding='utf-8').read()[:4000])"),
    ],
    "03_descriptive_analytics.ipynb": [
        md("# 03 - Descriptive analytics and visualisation"),
        code(SETUP),
        code("from src.analytics import descriptive\nfindings = descriptive.run()\nfindings"),
        code("pd.read_csv('outputs/descriptive/numeric_summary.csv')"),
        code("pd.read_csv('outputs/descriptive/categorical_cardinality.csv')"),
        code("pd.read_csv('outputs/descriptive/freq_genres_multilabel.csv').head(20)"),
        code("pd.read_csv('outputs/descriptive/categorical_associations.csv')"),
        code("from src.visualization import eda_plots\nmeta = eda_plots.run()\n"
             "for m in meta:\n    display(Markdown(f\"### Plot {m['number']}: {m['type']}\\n**Inference:** {m['inference']}\"))\n"
             "    show(m['file'], 800)"),
    ],
    "04_apriori.ipynb": [
        md("# 04 - Association rule mining with Apriori (from scratch)"),
        code(SETUP),
        code("from src.association_rules import run_apriori\nsummary = run_apriori.run()\nsummary"),
        code("pd.read_csv('outputs/apriori/levelwise_trace.csv')"),
        code("pd.read_csv('outputs/apriori/support_analysis.csv')"),
        code("pd.read_csv('outputs/apriori/top_rules.csv')[['antecedent','consequent','support','confidence','lift','leverage','conviction']]"),
        code("for f in ['17_apriori_support_confidence_lift.png','18_apriori_threshold_sensitivity.png',"
             "'19_apriori_levelwise_candidates.png','20_apriori_top_rules_lift.png']: show(f)"),
    ],
    "05_decision_tree.ipynb": [
        md("# 05 - Decision tree classification (target: Halliwell acclaim >= 2 stars)"),
        code(SETUP),
        code("from src.classification import decision_tree\ns = decision_tree.run()\ns['test_metrics']"),
        code("pd.read_csv('outputs/decision_tree/root_split_measures.csv').head(12)"),
        code("print(open('outputs/decision_tree/classification_report.txt').read())"),
        code("pd.read_csv('outputs/decision_tree/if_then_rules.csv').head(12)"),
        code("for f in ['23_decision_tree_structure.png','24_decision_tree_confusion_matrix.png','25_decision_tree_roc_curve.png',"
             "'26_decision_tree_feature_importance.png','27_decision_tree_pruning_curve.png','28_decision_tree_split_criteria.png']: show(f)"),
    ],
    "06_kmeans.ipynb": [
        md("# 06 - K-Means clustering of director profiles (+ hierarchical clustering)"),
        code(SETUP),
        code("from src.clustering import kmeans\ns = kmeans.run()\n{k: s[k] for k in ['chosen_k','silhouette','davies_bouldin','cluster_sizes','manual_vs_sklearn']}"),
        code("pd.read_csv('outputs/kmeans/k_selection_metrics.csv')"),
        code("pd.read_csv('outputs/kmeans/cluster_profiles.csv')"),
        code("pd.read_csv('outputs/kmeans/hierarchical_comparison.csv')"),
        code("for f in ['29_kmeans_elbow_silhouette_db.png','30_kmeans_silhouette_plot.png','31_kmeans_pca_clusters.png',"
             "'33_kmeans_centroid_heatmap.png','34_kmeans_parallel_coordinates.png','35_kmeans_pairplot.png',"
             "'36_kmeans_manual_convergence.png','37_hierarchical_dendrograms.png']: show(f)"),
    ],
    "07_cofi.ipynb": [
        md("# 07 - Closed frequent itemsets with the COFI-tree algorithm"),
        code(SETUP),
        code("from src.cofi.cofi import cofi_mine\nfrom src.cofi.run_cofi import LECTURE11_DB\n"
             "res = cofi_mine(LECTURE11_DB, min_count=3)\nprint('Lecture 11 closed sets:', [sorted(x) for x in res.closed])"),
        code("from src.cofi import run_cofi\ns = run_cofi.run()\ns['validation']"),
        code("print(open('outputs/cofi/cofi_execution_summary.txt', encoding='utf-8').read())"),
        code("pd.read_csv('outputs/cofi/closed_frequent_itemsets.csv').head(20)"),
        code("pd.read_csv('outputs/cofi/support_comparison.csv')"),
        code("for f in ['21_cofi_frequent_closed_maximal.png','22_cofi_tree_sizes.png']: show(f)"),
    ],
    "08_pyspark_benchmarking.ipynb": [
        md("# 08 - PySpark implementations and benchmarking\n"
           "The full benchmark (`src.pyspark_impl.spark_benchmark`) takes ~15-25 minutes on the target laptop; set "
           "`RUN_FULL_BENCHMARK = True` to rerun it. This notebook runs a live Spark demo of both techniques on the "
           "complete data and then loads the saved benchmark results.\n\n"
           "**Note:** on Windows, run Jupyter from the space-free path `D:\\ASBD_Hackathon` (Spark launch scripts "
           "break on paths with spaces)."),
        code(SETUP),
        code("RUN_FULL_BENCHMARK = False\nif RUN_FULL_BENCHMARK:\n    from src.pyspark_impl import spark_benchmark\n    spark_benchmark.run()"),
        code("""from src.pyspark_impl.spark_session import build_spark
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import roc_auc_score
import numpy as np, time
spark, startup = build_spark('notebook-demo'); print(f'Spark {spark.version} started in {startup:.1f}s')
dt = pd.read_csv('data/processed/dt_ml_matrix.csv').drop(columns='film_id')
feats = [c for c in dt.columns if c != 'acclaimed']
pos = dt.acclaimed.mean(); dt['w'] = np.where(dt.acclaimed == 1, 0.5 / pos, 0.5 / (1 - pos)); dt['label'] = dt.acclaimed.astype(float)
sdf = VectorAssembler(inputCols=feats, outputCol='features').transform(spark.createDataFrame(dt)).cache()
tr, te = sdf.randomSplit([0.75, 0.25], seed=42)
t0 = time.perf_counter(); m = DecisionTreeClassifier(weightCol='w', impurity='entropy', maxDepth=6, minInstancesPerNode=40, maxBins=256, seed=42).fit(tr)
out = m.transform(te).select(vector_to_array('probability').getItem(1).alias('p'), 'label').toPandas()
print(f'Spark DT fit+predict {time.perf_counter()-t0:.2f}s, test AUC {roc_auc_score(out.label, out.p):.3f}, nodes {m.numNodes}')
Z = pd.read_csv('data/processed/kmeans_director_matrix_z.csv').drop(columns='director')
zdf = VectorAssembler(inputCols=list(Z.columns), outputCol='features').transform(spark.createDataFrame(Z)).cache()
t0 = time.perf_counter(); km = KMeans(k=3, seed=42, maxIter=100).fit(zdf)
print(f'Spark K-Means fit {time.perf_counter()-t0:.2f}s, WCSS {km.summary.trainingCost:.1f}, sizes {km.summary.clusterSizes}')
spark.stop()"""),
        code("pd.read_csv('outputs/pyspark/performance_comparison_summary.csv')"),
        code("pd.read_csv('outputs/pyspark/benchmark_decision_tree.csv')"),
        code("pd.read_csv('outputs/pyspark/benchmark_kmeans.csv')"),
        code("pd.read_csv('outputs/pyspark/benchmark_frequent_patterns.csv')"),
        code("pd.read_csv('outputs/pyspark/stress_test_replicated.csv')"),
        code("for f in ['38_pyspark_execution_time.png','39_pyspark_speedup.png','40_pyspark_time_breakdown.png',"
             "'41_pyspark_quality_comparison.png']: show(f)"),
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        if args.only and name not in args.only:
            continue
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = NB / name
        nbf.write(nb, path)
        print("written", path.name)
        if args.execute:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", str(path)], check=True, cwd=NB)
            print("executed", path.name)


if __name__ == "__main__":
    main()

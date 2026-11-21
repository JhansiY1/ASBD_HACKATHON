# ASBD Movie Analytics: UCI Movies Data Set

End-to-end project for **Advanced Statistical and Big Data Analytics (ASBD)**. The pipeline runs in the KDD
order:
1. Preprocessing (cleaning, integration, selection, transformation)
2. Descriptive analytics
3. 41 visualisations covering 24 distinct techniques (16 distinct techniques among the exploratory plots alone)
4. Apriori association rules (from scratch)
5. Decision-tree classification
6. K-Means clustering (manual + scikit-learn) and hierarchical clustering
7. COFI-tree closed frequent itemset mining (from scratch, independently validated)
8. Spark MLlib implementations, benchmarked against Python

The project ends with a full academic report in DOCX and PDF.

## Dataset

* **Movie**, UCI Machine Learning Repository, dataset id 132. Creator: Gio Wiederhold. Licence: CC BY 4.0.
  DOI [10.24432/C5SW2R](https://doi.org/10.24432/C5SW2R).
* Page: <https://archive.ics.uci.edu/ml/datasets/Movie>. Download: <https://archive.ics.uci.edu/static/public/132/movie.zip>.
* Stored unmodified in `data/raw/` (`movie.zip` plus the extracted `movie/` folder).
* The dataset is relational: MAIN (films), CASTS, ACTORS, PEOPLE (directors etc.), REMAKES and STUDIOS. Each is a
  hand-written HTML table, documented in `data/raw/movie/doc.html`.

## Project structure

```
endsem-hackathon/              (also reachable as the space-free junction D:\ASBD_Hackathon)
├── run_all.py                 one-command pipeline (all phases)
├── requirements.txt / environment.yml
├── CLAUDE.md                  notes for AI-assisted maintenance
├── reference_materials/       notes.zip, extracted slides, lecture_text/, lecture_summary.md
├── data/
│   ├── raw/                   original UCI files (never modified)
│   ├── processed/             cleaned/integrated films, ML matrix, director profiles, smoothing outputs
│   └── transactions/          film baskets (JSON, CSV, binary matrix)
├── notebooks/                 01..08 executed notebooks (one per phase)
├── src/
│   ├── config.py              paths, seeds, shared parameters, logger
│   ├── data_loading/          raw HTML table reader (encoding detection, TeX accents)
│   ├── preprocessing/         cleaning, integration, selection, transformation, outliers/smoothing, pipeline
│   ├── analytics/             descriptive statistics
│   ├── visualization/         style, 16 EDA plots, inference writer
│   ├── association_rules/     Apriori from scratch + experiment runner
│   ├── classification/        split criteria (entropy, IG, gain ratio, Gini) + decision tree
│   ├── clustering/            manual & sklearn K-Means, hierarchical clustering
│   ├── cofi/                  FP-tree, COFI-tree mining, closed/maximal itemsets, validation
│   ├── pyspark_impl/          Spark session (laptop-tuned), Windows worker fix, benchmarks
│   ├── report/                DOCX builder + Word post-processing (equations, TOC, PDF)
│   └── utils/                 lecture text extraction, notebook builder
├── outputs/
│   ├── dataset_overview.txt, data_dictionary.csv, data_quality_summary.csv, cleaning_log.csv,
│   │   data_selection.csv, outlier_summary.csv, preprocessing_report.md, visualization_inferences.md
│   ├── descriptive/  plots/  apriori/  decision_tree/  kmeans/  cofi/  pyspark/  logs/
├── reports/
│   ├── ASBD_Movie_Analytics_Report.docx   (primary, editable)
│   ├── ASBD_Movie_Analytics_Report.pdf
│   ├── figures/  report_assets/
└── tests/                     test_preprocessing.py, test_apriori.py, test_cofi.py, test_algorithms.py
```

## Installation

Requirements:
* Python 3.12 (64-bit)
* **Java 17 or 21** for PySpark 4 (tested with Temurin 21.0.11)
* Microsoft Word (Windows) for the equation build-up and PDF export. Without Word, `docx2pdf` is used and the
  equations stay in linear form.

```powershell
cd "D:\sem-7-labs\Big Data labs\endsem-hackathon"
python -m venv .venv                       # the delivered venv used --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt
```

The environment is kept on D:. pip's cache was redirected to `D:\pip_cache` and Spark's scratch space is
`E:\spark_tmp` (configured in `src/config.py`), so nothing large is written to the nearly full C: drive.

### Windows notes for PySpark

1. **No spaces in the path.** Spark's launch scripts break on `...\Big Data labs\...`. A junction provides a
   space-free path:
   ```powershell
   cmd /c mklink /J D:\ASBD_Hackathon "D:\sem-7-labs\Big Data labs\endsem-hackathon"
   ```
   `run_all.py` automatically runs the Spark phase through this junction.
2. **Python-worker crash.** PySpark 4.0.1 workers on Windows exit before the JVM reads their reply.
   `src/pyspark_impl/win_worker/spark_win_worker.py` is a minimal entry point that waits for the JVM to close the
   socket. It is enabled automatically via `spark.python.worker.module` on Windows.
3. `winutils.exe` warnings can be ignored, because nothing is written through the Hadoop file system.

## Running

| What | Command (from `D:\ASBD_Hackathon`) | Output |
|---|---|---|
| Everything | `.venv\Scripts\python run_all.py` | all outputs + report |
| Everything except Spark (~2 min) | `.venv\Scripts\python run_all.py --skip-spark` | |
| Preprocessing | `.venv\Scripts\python -m src.preprocessing.pipeline` | `data/processed`, `outputs/*.csv` |
| Descriptive statistics | `.venv\Scripts\python -m src.analytics.descriptive` | `outputs/descriptive/` |
| Regenerate EDA plots | `.venv\Scripts\python -m src.visualization.eda_plots` | `outputs/plots/01-16` |
| Apriori | `.venv\Scripts\python -m src.association_rules.run_apriori` | `outputs/apriori/` |
| Decision tree | `.venv\Scripts\python -m src.classification.decision_tree` | `outputs/decision_tree/` |
| K-Means + hierarchical | `.venv\Scripts\python -m src.clustering.kmeans` | `outputs/kmeans/` |
| COFI + validation | `.venv\Scripts\python -m src.cofi.run_cofi` | `outputs/cofi/` |
| PySpark benchmark (~20 min) | `.venv\Scripts\python -m src.pyspark_impl.spark_benchmark` | `outputs/pyspark/` |
| Plot inferences | `.venv\Scripts\python -m src.visualization.write_inferences` | `outputs/visualization_inferences.md` |
| Report (DOCX + PDF) | `.venv\Scripts\python -m src.report.build_report` | `reports/` |
| Notebooks | `.venv\Scripts\python -m src.utils.build_notebooks --execute` | `notebooks/` |
| Tests | `.venv\Scripts\python -m pytest tests -q` | 48 tests |

Every phase appends to `outputs/logs/project_execution.log`.

## Key parameters (justified in the report)

* Apriori: minimum support 0.02 (229 films), minimum confidence 0.60. Sensitivity grid 0.01-0.05 × 0.5-0.8.
* COFI: minimum support 0.02, the same as Apriori, plus a comparison over 0.005-0.10.
* Decision tree: target = Halliwell rating ≥ 2 stars. Entropy criterion, depth 6, min leaf 40, balanced weights
  and cost-complexity α (chosen by 5-fold CV), on a stratified 75/25 split.
* K-Means: 682 directors with ≥ 5 films, 19 standardised features, K = 3 (K 2-10 evaluated).
* Spark: `local[2]`, 2 GB driver, 4 shuffle partitions, Arrow enabled. Timings are the median of 3 runs on
  10/25/50/100 % of the data.

## Results at a glance

* 11,477 raw MAIN rows, 871 of them malformed, cleaned to 11,408 films. Every operation is logged in
  `outputs/cleaning_log.csv`.
* Apriori: 310 frequent itemsets and 226 rules, identical to mlxtend. COFI: 293 closed and 127 maximal itemsets.
  All five validation checks pass.
* Decision tree: test ROC-AUC about 0.74. Year and documented cast size dominate.
* K-Means: three director profiles (classical studio, modern colour, Hitchcock-TV suspense). Silhouette is about
  0.13, i.e. weak structure.
* PySpark gives equivalent results but is slower on this small dataset. The fixed overhead dominates; see
  `outputs/pyspark/`.

## Reports

`reports/ASBD_Movie_Analytics_Report.docx` (editable; student and institution fields are placeholders) and
`reports/ASBD_Movie_Analytics_Report.pdf` (exported by Word from the same DOCX).

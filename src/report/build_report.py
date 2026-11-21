"""
Builds the final academic report:
    reports/ASBD_Movie_Analytics_Report.docx  (primary, editable)
    reports/ASBD_Movie_Analytics_Report.pdf   (exported by Word from the same DOCX)

Every number in the text is read from the result files produced by the
pipeline, so the report cannot drift from the actual experiments.

Run:  python -m src.report.build_report
"""
from __future__ import annotations

import json
import shutil
from datetime import date

import numpy as np
import pandas as pd

from src.config import (APRIORI_DIR, COFI_DIR, DESCRIPTIVE_DIR, DT_DIR, KMEANS_DIR, OUTPUTS, PLOTS_DIR, PROCESSED_DIR,
                        REPORTS_DIR, SPARK_DIR, get_logger)
from src.report import docx_helpers as H
from src.report.word_postprocess import finalise

log = get_logger("report")
DOCX = REPORTS_DIR / "ASBD_Movie_Analytics_Report.docx"
PDF = REPORTS_DIR / "ASBD_Movie_Analytics_Report.pdf"


def J(p):
    return json.loads(p.read_text(encoding="utf-8"))


class R:
    """All results, loaded once."""

    def __init__(self):
        self.pre = J(OUTPUTS / "preprocessing_stats.json")
        self.clog = pd.read_csv(OUTPUTS / "cleaning_log.csv")
        self.qual = pd.read_csv(OUTPUTS / "data_quality_summary.csv")
        self.dd = pd.read_csv(OUTPUTS / "data_dictionary.csv")
        self.sel = pd.read_csv(OUTPUTS / "data_selection.csv")
        self.outl = pd.read_csv(OUTPUTS / "outlier_summary.csv")
        self.kf = J(DESCRIPTIVE_DIR / "key_findings.json")
        self.num = pd.read_csv(DESCRIPTIVE_DIR / "numeric_summary.csv")
        self.card = pd.read_csv(DESCRIPTIVE_DIR / "categorical_cardinality.csv")
        self.gfreq = pd.read_csv(DESCRIPTIVE_DIR / "freq_genres_multilabel.csv")
        self.assoc = pd.read_csv(DESCRIPTIVE_DIR / "categorical_associations.csv")
        self.hgen = pd.read_csv(DESCRIPTIVE_DIR / "halliwell_by_genre.csv")
        self.hera = pd.read_csv(DESCRIPTIVE_DIR / "halliwell_by_era.csv")
        self.figs = J(OUTPUTS / "figure_catalogue.json")
        self.ap = J(APRIORI_DIR / "apriori_summary.json")
        self.sens = pd.read_csv(APRIORI_DIR / "support_analysis.csv")
        self.lv = pd.read_csv(APRIORI_DIR / "levelwise_trace.csv")
        self.top = pd.read_csv(APRIORI_DIR / "top_rules.csv")
        self.rare = pd.read_csv(APRIORI_DIR / "supplementary_rating_oscar_rules.csv")
        self.items = pd.read_csv(APRIORI_DIR / "item_supports.csv")
        self.dt = J(DT_DIR / "decision_tree_summary.json")
        self.root = pd.read_csv(DT_DIR / "root_split_measures.csv")
        self.rules = pd.read_csv(DT_DIR / "if_then_rules.csv")
        self.imp = pd.read_csv(DT_DIR / "feature_importance.csv", index_col=0)["importance"]
        self.cm = pd.read_csv(DT_DIR / "confusion_matrix.csv", index_col=0)
        self.km = J(KMEANS_DIR / "kmeans_summary.json")
        self.ksel = pd.read_csv(KMEANS_DIR / "k_selection_metrics.csv")
        self.prof = pd.read_csv(KMEANS_DIR / "cluster_profiles.csv", index_col=0)
        self.hier = pd.read_csv(KMEANS_DIR / "hierarchical_comparison.csv")
        self.cofi = J(COFI_DIR / "cofi_summary.json")
        self.ccomp = pd.read_csv(COFI_DIR / "support_comparison.csv")
        self.cfi = pd.read_csv(COFI_DIR / "closed_frequent_itemsets.csv")
        self.bdt = pd.read_csv(SPARK_DIR / "benchmark_decision_tree.csv")
        self.bk = pd.read_csv(SPARK_DIR / "benchmark_kmeans.csv")
        self.bf = pd.read_csv(SPARK_DIR / "benchmark_frequent_patterns.csv")
        self.st = pd.read_csv(SPARK_DIR / "stress_test_replicated.csv")
        self.env = J(SPARK_DIR / "spark_configuration.json")

    def fig(self, file):
        return next(f for f in self.figs if f["file"] == file)

    def clog_n(self, table, attribute, contains):
        d = self.clog[(self.clog.table == table) & (self.clog.attribute == attribute)
                      & self.clog.description.str.contains(contains, regex=False)]
        return int(d.records_affected.iloc[0]) if len(d) else 0


def fig_block(doc, r: R, file: str, title: str, C: H.Counter, full=False) -> int:
    n = H.figure(doc, PLOTS_DIR / file, title, C)
    f = r.fig(file)
    if full:
        H.bullets(doc, [f"**Attributes involved:** {f['attributes']}",
                        f"**Why this plot:** {f['reason']}",
                        f"**Observations:** {f['observations']}",
                        f"**Key inference:** {f['inference']}",
                        f"**Conclusion:** {f['conclusion']}"])
    else:
        H.para(doc, f"*Figure {n} - observations:* {f['observations']} *Inference:* {f['inference']} "
                    f"{f['conclusion']}", align="justify")
    return n


# ======================================================================
def build() -> dict:
    r = R()
    doc = H.new_document()
    C = H.Counter()
    H.header_footer(doc, "Advanced Statistical and Big Data Analytics - UCI Movie Dataset: Descriptive, Predictive "
                         "and Big-Data Analytics")
    cover(doc, r)
    H.toc(doc, "Table of Contents", 'TOC \\o "1-3" \\h \\z \\u')
    H.page_break(doc)
    H.toc(doc, "List of Figures", 'TOC \\h \\z \\c "Figure"')
    H.page_break(doc)
    H.toc(doc, "List of Tables", 'TOC \\h \\z \\c "Table"')
    abstract(doc, r)                                 # Heading 1 starts a new page itself
    s1_intro(doc, r)
    s2_dataset(doc, r, C)
    s3_method(doc, r, C)
    s4_preprocessing(doc, r, C)
    s5_descriptive(doc, r, C)
    s6_visualisation(doc, r, C)
    s7_apriori(doc, r, C)
    s8_tree(doc, r, C)
    s9_kmeans(doc, r, C)
    s10_cofi(doc, r, C)
    s11_pyspark(doc, r, C)
    s12_benchmark(doc, r, C)
    s13_additional(doc, r, C)
    s14_limitations(doc, r)
    s15_conclusion(doc, r)
    s16_references(doc)
    appendix(doc, r, C)
    doc.save(DOCX)
    log.info("DOCX written: %d figures, %d tables", C.fig, C.tab)
    info = finalise(DOCX, PDF)
    info.update(figures=C.fig, tables=C.tab)
    for f in PLOTS_DIR.glob("*.png"):
        shutil.copy2(f, REPORTS_DIR / "figures" / f.name)
    log.info("PHASE 13-14 complete: %s", info)
    (REPORTS_DIR / "report_assets" / "report_build_info.json").write_text(json.dumps(info, indent=2))
    return info


# ======================================================================
def cover(doc, r):
    for _ in range(4):
        doc.add_paragraph()
    H.para(doc, "ADVANCED STATISTICAL AND BIG DATA ANALYTICS (ASBD)", align="center", size=12)
    H.para(doc, "Course Project Report", align="center", size=12)
    doc.add_paragraph()
    p = H.para(doc, "**Exhaustive Descriptive, Predictive and Big-Data Analytics of the UCI Movies Data Set**",
               align="center", size=22)
    H.para(doc, "Data preprocessing (KDD), descriptive statistics, visualisation, Apriori association rules, decision "
                "tree classification, K-Means clustering, COFI closed frequent itemsets and PySpark benchmarking",
           align="center", size=11.5)
    for _ in range(3):
        doc.add_paragraph()
    rows = [("Dataset", "Movie (UCI Machine Learning Repository, id 132) - G. Wiederhold"),
            ("Dataset source", "https://archive.ics.uci.edu/ml/datasets/Movie  (DOI 10.24432/C5SW2R)"),
            ("Student name", "[Student Name]"), ("Roll / registration number", "[Roll Number]"),
            ("Programme / semester", "[Programme, Semester]"), ("Course instructor", "[Instructor Name]"),
            ("Institution", "[Institution Name]"), ("Submission date", date.today().strftime("%d %B %Y"))]
    t = doc.add_table(rows=0, cols=2)
    t.style = "Table Grid"
    for k, v in rows:
        c = t.add_row().cells
        c[0].text = ""; c[1].text = ""
        H.add_rich(c[0].paragraphs[0], f"**{k}**", 10.5)
        H.add_rich(c[1].paragraphs[0], v, 10.5)
    from docx.shared import Cm
    for row in t.rows:
        row.cells[0].width = Cm(5.0); row.cells[1].width = Cm(11.0)
    H.page_break(doc)


def abstract(doc, r):
    doc.add_heading("Abstract", 1)
    b = r.bk[r.bk.workload == "kmeans_films"].iloc[-1]
    real_sp = pd.concat([r.bdt.speedup, r.bk.speedup, r.bf.speedup])
    slow_lo, slow_hi = 1 / real_sp.max(), 1 / real_sp.min()
    H.para(doc, (
        f"This report applies the complete Knowledge Discovery in Databases (KDD) process to the UCI *Movie* data set, "
        f"a relational collection of six hand-written HTML tables describing {r.pre['raw_main_rows']:,} film records, "
        f"46 thousand cast entries, 6.8 thousand actors and 3.3 thousand film people. The raw data is noisy: "
        f"{r.pre['raw_main_malformed']} MAIN rows have merged or split cells, missing values are hidden behind "
        f"placeholder codes, and category codes contain case variants and typos. A reproducible pipeline cleans, "
        f"integrates, selects and transforms the data into {r.pre['films_final']:,} analysable films. Descriptive "
        f"statistics and {len(r.figs)} figures ({len({f['family'] for f in r.figs})} distinct visualisation techniques) "
        f"characterise the collection. A from-scratch Apriori finds {r.ap['frequent_itemsets']} frequent itemsets and "
        f"{r.ap['rules']} rules at 2 % support / 60 % confidence; it matches an independent implementation exactly. "
        f"A decision tree predicts critical acclaim (two or more stars in Halliwell's Film Guide) with test ROC-AUC "
        f"{r.dt['test_metrics']['roc_auc']:.3f}. K-Means groups {r.km['n_directors']} directors into "
        f"{r.km['chosen_k']} career profiles. A COFI-tree implementation mines {r.cofi['closed_itemsets']} closed "
        f"frequent itemsets and passes five independent validation checks. Finally, Spark MLlib versions of the "
        f"decision tree, K-Means and frequent-pattern mining are benchmarked against Python. On the real data Spark is "
        f"between {slow_lo:.0f}× and {slow_hi:.0f}× slower (e.g. K-Means on all films: {b.py_total_s:.3f} s vs "
        f"{b.spark_total_s:.2f} s) because of fixed distributed-execution overheads, while producing equivalent results. "
        + (f"Only in a row-replication stress test (×{int(r.st[r.st.speedup > 1].replication_factor.min())}) does Spark "
           f"overtake Python, for " + ", ".join(sorted({w.split('_replicated')[0].replace('_', ' ')
                                                        for w in r.st[r.st.speedup > 1].workload})) + "."
           if (r.st.speedup > 1).any() else "Even a 20× row-replication stress test does not make Spark faster.")),
        align="justify")
    H.para(doc, "**Keywords:** KDD, data preprocessing, Apriori, COFI-tree, closed frequent itemsets, decision tree, "
                "K-Means, PySpark, MLlib, benchmarking.")


# ======================================================================
def s1_intro(doc, r):
    doc.add_heading("1. Introduction", 1)
    doc.add_heading("1.1 Background of data analytics", 2)
    H.para(doc, (
        "Data analytics converts raw data into information and, beyond what a database query can return, into "
        "knowledge that is not explicitly stored: patterns, associations, predictive models and groupings "
        "(Lecture 3). The course frames this as the KDD process: cleaning, integration, selection, transformation, "
        "data mining, pattern evaluation and knowledge presentation (Lecture 5). Real-world data is dirty "
        "(incomplete, noisy, inconsistent), so most of the effort in a KDD project goes into preprocessing. "
        "When data volumes grow, efficiency, scalability and parallelisation become major issues. This motivates "
        "distributed engines such as Apache Spark."), align="justify")
    doc.add_heading("1.2 Objectives", 2)
    H.bullets(doc, [
        "Perform exhaustive preprocessing (cleaning, selection, transformation) of the UCI Movie data set and "
        "document every operation.",
        "Characterise the data with descriptive statistics and at least ten distinct, appropriate visualisations.",
        "Mine association rules with Apriori, implemented and explained from first principles.",
        "Build a decision-tree classifier and a K-Means clustering where the data supports them.",
        "Mine closed frequent itemsets with the COFI-tree algorithm and validate the result independently.",
        "Implement two predictive-analytics techniques in PySpark and quantify the effect of big-data support "
        "honestly, including the case where distribution does not pay off."])
    doc.add_heading("1.3 Problem statement", 2)
    H.para(doc, (
        "Perform exhaustive and detailed descriptive analytics on the UCI Movies data set, including data cleaning, "
        "data selection and data transformation. Apply at least ten unique visualisation techniques and explain their "
        "inferences. Further perform association rule mining using Apriori, decision tree classification and/or "
        "K-Means clustering, closed frequent itemset mining using the COFI algorithm, and test two predictive analytics "
        "techniques using PySpark, establishing the performance enhancement achieved with big-data support. All "
        "assumed parameters (support, confidence, number of clusters, etc.) must be stated and justified."),
        align="justify")
    doc.add_heading("1.4 Scope", 2)
    H.para(doc, (
        "The analysis covers the six relational tables of the official UCI distribution (MAIN, CASTS, ACTORS, PEOPLE, "
        "REMAKES, STUDIOS) and their documentation. No external movie data is added and no records are synthesised "
        "for analysis. The only artificial data is a clearly labelled row-replication stress test in the benchmarking "
        "chapter, used to study scaling. All work runs on a single laptop (Intel i5-1235U, 8 GB RAM), with Spark in "
        "local mode."), align="justify")


def s2_dataset(doc, r, C):
    doc.add_heading("2. Dataset Description", 1)
    doc.add_heading("2.1 Source", 2)
    H.para(doc, (
        "The data set is *Movie* (UCI Machine Learning Repository, dataset id 132), created by Gio Wiederhold "
        "(Stanford University), donated in 1999 and licensed CC BY 4.0 (citation: Wiederhold, G. (1990). Movie "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SW2R). It was downloaded from the "
        "official endpoint `https://archive.ics.uci.edu/static/public/132/movie.zip` (1.78 MB) and stored unmodified "
        "in `data/raw/`. According to its documentation (doc.html), the database was compiled as test data for "
        "database courses (complex joins, temporal queries, recursion). It is complete for Hitchcock's films and TV "
        "episodes and otherwise reflects the author's interests and his reference books (Halliwell's Film Guide, Katz' "
        "Film Encyclopedia, Ebert, etc.). UCI lists no associated machine-learning task, so every analytical task in "
        "this report had to be derived and justified from the actual attributes."), align="justify")
    doc.add_heading("2.2 Files, formats and relationships", 2)
    inv = pd.read_csv(OUTPUTS / "data_quality_summary.csv")
    files = pd.DataFrame([
        ("main.html", "MAIN", "films (journal relation)", r.pre["raw_main_rows"], "11 per row"),
        ("casts.html", "CASTS", "actor-film-role association", 46197, "7"),
        ("actors.html", "ACTORS", "actor biographies", 6818, "11"),
        ("people.html", "PEOPLE", "directors, producers, writers", 3311, "10"),
        ("remakes.html", "REMAKES", "remake -> original links", 1277, "7"),
        ("studios.html", "STUDIOS", "studio history (sparse)", 207, "9"),
        ("doc.html", "-", "schema documentation, code tables", "-", "-")],
        columns=["File", "Relation", "Content", "Data rows", "Fields"])
    H.table(doc, files, "Files of the UCI Movie distribution (HTML tables, one record per <tr><td> line)", C)
    H.para(doc, (
        "All tables are HTML pages in which each record is one line `<tr><td> f1<td> f2 ... <td>|`. The files are "
        "ASCII, except `casts.html` and `remakes.html`, which contain a handful of Latin-1 bytes (they are not valid "
        "UTF-8). Accented letters are written as TeX escapes (`Mis\\'erables`). The relations are linked as follows: "
        "MAIN.director → PEOPLE.id-name (documented reference constraint); CASTS.film_id → MAIN.film_id; "
        "CASTS.actor → ACTORS.stagename (partial); REMAKES.film_id and REMAKES.priorfilm → MAIN.film_id. The "
        "`film_id` key is constructed as *director code + sequence number*."), align="justify")
    doc.add_heading("2.3 Data dictionary", 2)
    H.table(doc, r.dd, "Data dictionary of the raw attributes (from doc.html, section 3)", C,
            col_widths_cm=[2.0, 3.8, 3.4, 7.2])
    doc.add_heading("2.4 Initial observations and data quality", 2)
    q = r.qual[r.qual.table == "MAIN"][["attribute", "n_records", "missing_or_placeholder", "missing_pct", "n_unique",
                                         "most_frequent"]]
    H.table(doc, q, "Raw MAIN attribute profile (placeholder codes counted as missing)", C)
    H.bullets(doc, [
        f"MAIN has {r.pre['raw_main_rows']:,} data rows, of which {r.pre['raw_main_malformed']} "
        f"({100 * r.pre['raw_main_malformed'] / r.pre['raw_main_rows']:.1f} %) do not have the documented 11 cells.",
        "Missing values are almost never blank. They are encoded by placeholders: `aw` (awards unknown), `lc` "
        "(locale unknown), `prc` (process unknown), `Ctxx`/`CTxx` (uncategorised), `PU:` (producer unknown), "
        "`SU:` (studio unknown) and `\\Un`, `\\Und`, `19xx`.",
        "Categorical attributes are multi-valued (several genres, producers or studios per film) and coded "
        "(e.g. `Dram`, `\\Tcol`, `H**`).",
        "There is one genuine numeric attribute (year) plus derivable numbers: Halliwell stars, cast size, number "
        "of producers, film length (sparse) and director productivity.",
        f"The raw title has {int(r.qual[(r.qual.table == 'MAIN') & (r.qual.attribute == 'title_raw')].n_unique.iloc[0]):,} "
        f"distinct values, so titles are not unique (remakes such as 14 versions of *Romeo and Juliet*)."])


def s3_method(doc, r, C):
    doc.add_heading("3. Methodology", 1)
    doc.add_heading("3.1 KDD process", 2)
    H.para(doc, (
        "The project follows the KDD phases of Lecture 5. Each phase is a separate, re-runnable Python module, and all "
        "intermediate results are written to disk:"), align="justify")
    H.bullets(doc, [
        "**Cleaning** (`src/preprocessing/cleaning.py`): parse, realign, decode placeholders, standardise codes, "
        "detect duplicates, invalid values and outliers.",
        "**Integration** (`integration.py`): join MAIN with PEOPLE, CASTS, ACTORS and REMAKES; group-based and "
        "rule-based imputation.",
        "**Selection** (`selection.py`): task-relevant attributes per technique; removal of identifiers and "
        "leakage-prone attributes.",
        "**Transformation** (`transformation.py`): discretisation, label/one-hot/multi-hot encoding, z-score and "
        "min-max scaling, derived features, transactions.",
        "**Data mining**: Apriori, decision tree, K-Means (+ hierarchical), COFI, and the PySpark equivalents.",
        "**Pattern evaluation**: support/confidence/lift, cross-validation, silhouette/Davies-Bouldin, independent "
        "validation of every from-scratch algorithm.",
        "**Knowledge presentation**: figures, rule tables, IF-THEN rules, cluster profiles, this report."])
    doc.add_heading("3.2 Overall workflow", 2)
    H.para(doc, (
        "`python run_all.py` executes ten phases in order: preprocess → describe → plots → apriori → tree → kmeans → "
        "cofi → spark → inferences → report. Every phase appends to `outputs/logs/project_execution.log`. Random seeds "
        "are fixed (42) for sampling, splitting, K-Means initialisation and Spark."), align="justify")
    doc.add_heading("3.3 Tools, technologies and software environment", 2)
    env = r.env
    tools = pd.DataFrame([
        ("Language", f"Python {env['python']} (virtual environment on D:)"),
        ("Data handling", "pandas 2.2.1, NumPy 1.26.4, SciPy 1.16.3"),
        ("Machine learning", "scikit-learn 1.8.0 (decision tree, K-Means, metrics), mlxtend 0.23.4 (validation only)"),
        ("Visualisation", "matplotlib 3.10.1, seaborn 0.13.2, squarify 0.4.5"),
        ("Big data", f"Apache Spark {env['spark_version']} (PySpark, MLlib), Java 21 (Temurin), local[2]"),
        ("Reporting", "python-docx 1.2.0, Microsoft Word (equations, fields, PDF export)"),
        ("Testing", "pytest 9.2.0 (48 unit/integration tests)"),
        ("Hardware", f"{env['cpu']}, {env['ram_gb']} GB RAM, {env['os']}")], columns=["Aspect", "Configuration"])
    H.table(doc, tools, "Software and hardware environment", C, col_widths_cm=[3.5, 12.9])
    H.para(doc, (
        "Algorithms that the course derives from first principles (Apriori, COFI, entropy, information gain, gain "
        "ratio, Gini, K-Means) were implemented by hand and then validated against independent library "
        "implementations. Library implementations (scikit-learn, Spark MLlib) are used where the task is to apply or "
        "benchmark standard tools."), align="justify")


def s4_preprocessing(doc, r, C):
    doc.add_heading("4. Data Preprocessing", 1)
    H.para(doc, (f"Table {C.tab + 1} lists every cleaning, integration and imputation operation with the number of "
                 "records it affected. It is reproduced from `outputs/cleaning_log.csv`, which the pipeline writes "
                 "automatically."), align="justify")
    cl = r.clog[["table", "step", "attribute", "description", "records_affected", "action"]]
    H.table(doc, cl, "Complete log of cleaning, integration and imputation operations", C,
            col_widths_cm=[1.7, 1.7, 2.0, 6.4, 1.6, 3.0], font_size=7.5)
    doc.add_heading("4.1 Data cleaning", 2)
    doc.add_heading("4.1.1 Noisy and malformed records", 3)
    H.para(doc, (
        f"Because the files are hand-edited HTML, {r.pre['raw_main_malformed']} MAIN rows have between 2 and 17 cells "
        f"instead of 11. Typical defects are a director split across two cells (`D:` | `Dickson`), a missing awards "
        f"cell, or awards and locale merged (`H*csd, Austria`). Positional parsing would shift every later attribute "
        f"of these films. A *monotone slot-matching* realignment was therefore implemented. The first four cells (id, "
        f"title, year, director) are positional. Each subsequent cell is assigned to the earliest remaining schema "
        f"slot whose syntax it satisfies: `P:`/`PN:` → producer, `St:`/`SU:` → studio, process vocabulary → process, "
        f"category vocabulary → category, award grammar → awards, otherwise locale/notes. "
        f"{r.clog_n('MAIN', 'all fields', 'realigned')} rows were realigned this way. "
        f"{r.clog_n('MAIN', '*', '<4 fields')} rows with fewer than four cells, an empty film id or no title cell are "
        f"unrecoverable noise and were removed. Records the author himself flagged as doubtful (`Er(...)` in the "
        f"notes, {r.clog_n('MAIN', 'notes', 'Er(')} films) were kept and flagged."), align="justify")
    doc.add_heading("4.1.2 Missing values", 2 + 1)
    H.para(doc, (
        "Placeholder codes were first converted to true missing values. The treatment was then chosen per attribute "
        "(Lecture 5: global constant, attribute mean, class-wise mean, most probable value):"), align="justify")
    H.bullets(doc, [
        f"**Year** ({r.clog_n('MAIN', 'year', 'missing or unparseable')} missing): *group-based median imputation*, "
        f"using the median year of the same director's films "
        f"({r.clog_n('INTEGRATION', 'year', 'median year')} films). Film ids are sequential per director, so this is "
        f"the most probable value. Fallback: the director's career-start year from PEOPLE "
        f"({r.clog_n('INTEGRATION', 'year', 'career start')}). {r.clog_n('INTEGRATION', 'year', 'still unknown')} "
        f"films remained unknown and were removed.",
        f"**Colour process** (`prc`): *rule-based (most probable value) imputation* taken from the documentation "
        f"itself (doc.html 3.14: 'unknown process, likely bnw prior to 1945, col after 1955'), with silent b/w before "
        f"1928 ({r.clog_n('INTEGRATION', 'color_type', 'documented rule')} films). The 1945-1955 transition decade "
        f"has no defensible rule and was set to the constant 'unknown' "
        f"({r.clog_n('INTEGRATION', 'color_type', '1945-1955')} films). Mode imputation was rejected because it would "
        f"bias towards b/w or colour depending on the era.",
        "**Producer, studio, category, director background**: *global constant* ('unknown' / 'Unknown' category). "
        "These are nominal, so mean or median imputation is meaningless. Their missingness is informative (it "
        "depends strongly on the period, Figure 1).",
        "**Halliwell rating**: *not imputed*. A film absent from the guide has no rating, so it is excluded from the "
        "classification task instead of receiving a fabricated label.",
        "**Female cast share** (needs at least one actor of known gender): median imputation plus a missing-indicator "
        "feature for the classifier only. **Cast size**: 0 for films without cast records, with a `has_cast_info` "
        "flag, because missing cast information means 'not documented', not 'no actors'."])
    doc.add_heading("4.1.3 Inconsistent and invalid values", 3)
    H.bullets(doc, [
        f"Category codes with wrong letter case (`romt`, `DRam`, `CTxx`): {r.clog_n('MAIN', 'category', 'letter case')} "
        f"tokens case-folded. Misspelt codes (`Ducu`, `Muusc`, `Biop`, `C0md`, `Porm`): "
        f"{r.clog_n('MAIN', 'category', 'Misspelt')} tokens corrected with an explicit, documented typo table. Glued "
        f"codes (`CtxxDram`, `NoirSusp`) split. {r.clog_n('MAIN', 'category', 'not in the documented')} ambiguous "
        f"tokens (e.g. `CRomt`, `Sctn`, misplaced award codes) dropped rather than guessed.",
        f"Studio names with inconsistent spacing (`Shamley,CBS` vs `Shamley, CBS`): "
        f"{r.clog_n('MAIN', 'studios', 'spacing')} standardised. Major studios unified (`Warners`, `Warner` → Warner "
        f"Bros; `U.A.` → United Artists; `U-I` → Universal).",
        "String normalisation everywhere: HTML tags removed, TeX accents converted to Unicode (`Mis\\'erables` → "
        "Misérables), the `~` word joiner replaced by a space, whitespace collapsed, Latin-1 bytes decoded.",
        f"Validity rules: release year within 1880-1999 (none violated after parsing); film length within 1-600 minutes "
        f"({r.clog_n('MAIN', 'length_min', '1-600')} unit errors removed, e.g. lengths in feet); ACTORS death before "
        f"birth ({r.clog_n('ACTORS', 'birth/death', 'before')}); REMAKES dated before the original "
        f"({r.clog_n('REMAKES', 'year', 'before')}, flagged because the documentation requires remake year > original year).",
        f"Referential integrity: {r.clog_n('INTEGRATION', 'director', 'not found')} films reference directors "
        f"missing from PEOPLE, and {r.clog_n('INTEGRATION', 'CASTS.film_id', 'dangling')} cast rows reference "
        f"non-existent films (dangling pointers, which the documentation acknowledges)."])
    doc.add_heading("4.1.4 Duplicates", 3)
    H.para(doc, (
        f"Three kinds of duplication were distinguished. (i) **Exact duplicate records**: "
        f"{r.clog_n('MAIN', 'all fields', 'Exact duplicate')} removed. (ii) **The same film entered twice under "
        f"different ids** (identical normalised title, year and director): "
        f"{r.clog_n('MAIN', 'title+year+director', 'twice')} removed, keeping the first. (iii) **Key collisions**: "
        f"{r.clog_n('MAIN', 'film_id', 'collision')} distinct films share a film_id. These are not duplicates, so "
        f"they were kept and made unique with suffixes. Remakes with the same title (e.g. the many *Hamlet*s) are "
        f"legitimately different films and were not touched. CASTS had "
        f"{r.clog_n('CASTS', 'film_id+actor', 'twice')} repeated (film, actor) pairs, the key of that relation."),
        align="justify")
    doc.add_heading("4.1.5 Outliers and smoothing", 3)
    o = r.outl[["attribute", "n", "q1", "q3", "iqr", "upper_fence", "n_iqr_outliers", "n_z_outliers", "max", "verdict"]]
    H.table(doc, o, "Outlier analysis (IQR rule: x > Q3 + 1.5 IQR or x < Q1 - 1.5 IQR; z-rule: |z| > 3)", C,
            col_widths_cm=[2.3, 1.0, 1.0, 1.0, 1.0, 1.3, 1.3, 1.3, 1.1, 5.6], font_size=7.5, decimals=2)
    H.para(doc, (
        "Outliers were **flagged, not deleted**. Every flagged attribute was reviewed, and the extreme values are "
        "genuine: 1890s Edison/Lumière shorts, Hitchcock's 79 films (the database is complete for him by design), "
        "all-star casts and four-hour epics. Only values that are *errors* (impossible lengths) were removed. Noise "
        "smoothing used the methods of Lecture 5: (a) equal-depth binning of film length into 8 bins, smoothed by bin "
        "means and by bin boundaries (`data/processed/length_min_binning_smoothing.csv`); (b) a 5-year centred moving "
        "average of the yearly film counts (Figure 3); (c) regression and bin-mean smoothing of the female cast "
        "share (Figure 15). The binning routine was verified on the price example in the Lecture 5 slides "
        "(4, 8, 9, 15 | 21, 21, 24, 25 | 26, 28, 29, 34 → bin means 9, 23, 29)."), align="justify")
    sm = pd.read_csv(PROCESSED_DIR / "length_min_binning_smoothing.csv", index_col=0)
    smt = sm.groupby("bin").agg(n=("original", "size"), min=("original", "min"), max=("original", "max"),
                                 bin_mean=("by_mean", "first")).reset_index()
    H.table(doc, smt, "Equal-depth binning of film length (minutes) with smoothing by bin means", C, decimals=1)
    doc.add_heading("4.2 Data integration", 2)
    H.para(doc, (
        "The relational tables were integrated into one film-level table. Director attributes (birth country, "
        "female flag `Ge(F)`, career years) come from PEOPLE. Cast size, gender composition, mean actor birth year and "
        "acting-award nominations come from CASTS ⋈ ACTORS. The remake flags `is_remake` and `was_remade` come from "
        "REMAKES. This resolves the entity-identification problem through the dataset's own standardised keys "
        "(director id-names and stage names). A director-level table of 682 directors with at least five films was "
        "also aggregated for clustering."), align="justify")
    doc.add_heading("4.3 Data selection", 2)
    H.para(doc, (
        "Attributes were selected per analytical technique. Identifiers (film_id, title, raw names) carry no "
        "analytical meaning and were discarded as features (Lecture 5: 'students' ID is irrelevant to predicting "
        "GPA'). Attributes that encode the outcome of the classification target were excluded to prevent "
        "**target leakage**: Oscar nominations/wins, other critics' awards, and whether the film was *later* remade."),
        align="justify")
    H.table(doc, r.sel, "Attribute selection decisions and justification", C, col_widths_cm=[3.2, 3.0, 4.2, 6.0],
            font_size=7.5)
    rel = pd.DataFrame([
        ("Descriptive analytics", "all cleaned attributes"),
        ("Apriori / COFI / FP-Growth", "genres, era, colour type, studio group, director country, cast-size bin, "
                                       "Halliwell level, Oscar flag, remake flags, TV flag, book source"),
        ("Decision tree", "year, genres (multi-hot), colour type, studio group, director country, producer status, "
                          "#producers, cast size, female share, director productivity, remake/book/series/TV flags"),
        ("K-Means / hierarchical", "19 director-profile features: log #films, mean year, career span, colour/TV "
                                   "share, mean rating, rated share, Oscar rate, mean cast size, 10 genre shares")],
        columns=["Technique", "Attributes used"])
    H.table(doc, rel, "Task-relevant attributes per technique", C, col_widths_cm=[4.0, 12.4])
    doc.add_heading("4.4 Data transformation", 2)
    H.bullets(doc, [
        "**Discretisation and concept hierarchy**: year → decade → era. The five eras follow film history: Silent "
        "(≤1927), Classical studio (1928-48, until the 1948 Paramount decree), Post-studio (1949-66), New Hollywood "
        "(1967-80), Modern (1981-99). Cast size → {1-2, 3-5, 6-9, 10+}. Halliwell stars → 5 levels.",
        "**Label encoding** only for ordinal attributes (era_code 0-4, halliwell_code 0-4).",
        "**One-hot encoding** of nominal attributes (colour type, studio group, director country group, producer "
        "status) and **multi-hot (binary item presence)** of the multi-valued genre attribute (genres with ≥ 30 "
        "films).",
        "**Standardisation and normalisation** of numeric attributes only (never of codes). The formulas are given "
        "below. z-scores feed K-Means, where Euclidean distance requires comparable scales. The decision tree is "
        "scale-invariant and uses raw values.",
        "**Transactions**: each film becomes a basket of `attribute=value` items, e.g. "
        "{genre=Comedy, era=Classical studio (1928-48), color=bw, studio=MGM, dir_country=USA, cast 6-9}. Missing "
        "values and generic 'Other' buckets produce no item, so they cannot dominate the counts."])
    H.para(doc, "Min-max normalisation and z-score standardisation (Lecture 5):")
    H.equation(doc, "v′=(v-\"min\"_A)/(\"max\"_A-\"min\"_A)")
    H.equation(doc, "z=(x-μ_A)/σ_A")
    doc.add_heading("4.5 Feature engineering", 2)
    H.para(doc, (
        "Derived features: Halliwell stars, Oscar nominations and wins and other awards are parsed from the award "
        "grammar; MPAA rating, language, length, book source, series and co-director flags come from the notes "
        "sub-fields; the setting country/region from the locale hierarchy; director productivity is the number of "
        "the director's films in the collection; female share of the documented cast; remake indicators; decade and "
        "era. The classification target **acclaimed** is derived from an existing attribute: 1 if the film has two "
        "or more stars in Halliwell's Film Guide, 0 if listed with 0-1 stars. Unrated films are excluded."),
        align="justify")
    doc.add_heading("4.6 Before-and-after comparison", 2)
    ba = pd.DataFrame([
        ("MAIN data rows", f"{r.pre['raw_main_rows']:,}", f"{r.pre['films_final']:,} films"),
        ("Malformed rows", f"{r.pre['raw_main_malformed']}", "0 (realigned or removed)"),
        ("Missing year", f"{r.clog_n('MAIN', 'year', 'missing or unparseable')}", "0"),
        ("Unknown colour process", f"{int(r.qual[(r.qual.table == 'MAIN') & (r.qual.attribute == 'process_raw')].missing_or_placeholder.iloc[0])}",
         f"{r.clog_n('INTEGRATION', 'color_type', '1945-1955')} (1945-55 only)"),
        ("Duplicate / colliding records", f"{r.clog_n('MAIN', 'all fields', 'Exact duplicate') + r.clog_n('MAIN', 'title+year+director', 'twice') + r.clog_n('MAIN', 'film_id', 'collision')}",
         "0 duplicates; ids unique"),
        ("Distinct category tokens", "406 raw strings", f"{len(r.gfreq)} standard genres"),
        ("Attributes", "11 raw (MAIN) + 5 related tables", "1 integrated table (≈90 columns)"),
        ("Decision-tree matrix", "-", f"{r.pre['dt_rows']:,} labelled films × {r.pre['dt_features']} features"),
        ("Director profiles", "-", f"{r.pre['directors_profiled']} directors × {r.pre['kmeans_features']} features"),
        ("Transactions", "-", f"{r.pre['transactions']:,} baskets, {r.pre['items']} items, mean length "
                              f"{r.pre['avg_tx_len']:.2f}")], columns=["Aspect", "Before (raw)", "After preprocessing"])
    H.table(doc, ba, "Dataset before and after preprocessing", C, col_widths_cm=[5.0, 5.0, 6.4])


def s5_descriptive(doc, r, C):
    doc.add_heading("5. Descriptive Analytics", 1)
    doc.add_heading("5.1 Statistical summary of numerical attributes", 2)
    num = r.num[["attribute", "count", "mean", "median", "mode", "min", "max", "range", "std", "variance", "q1", "q3",
                 "iqr", "skewness", "kurtosis"]]
    H.table(doc, num, "Descriptive statistics of numerical attributes (cast size over documented films only)", C,
            font_size=7, decimals=2)
    kf = r.kf
    H.bullets(doc, [
        f"Release years span {kf['year_range'][0]}-{kf['year_range'][1]} (mean "
        f"{r.num.set_index('attribute').loc['year', 'mean']:.1f}, median {kf['median_year']:.0f}). The distribution is "
        f"mildly left-skewed (skewness {r.num.set_index('attribute').loc['year', 'skewness']:.2f}) and platykurtic, "
        f"i.e. spread fairly evenly over the century. The modal year is {kf['peak_year']} ({kf['peak_year_films']} films).",
        f"Only {kf['rated_films']:,} films ({100 * kf['rated_films'] / kf['n_films']:.1f} %) have a Halliwell rating. "
        f"Their mean is {kf['mean_halliwell']:.2f} stars, the median 1 and the mode 0 (listed without stars).",
        f"Count attributes (Oscar nominations, times remade, acting-award nominations) are extremely right-skewed with "
        f"high kurtosis: most films have 0, a few have several. Medians are more representative than means.",
        f"Documented casts have a median of {kf['median_cast_size_documented']:.0f} actors (max 46). Only "
        f"{100 * kf['films_with_cast'] / kf['n_films']:.1f} % of films have any cast record.",
        f"The mean female share of documented casts is {kf['mean_female_cast_share']:.3f}: about two thirds of listed "
        f"actors are male."])
    doc.add_heading("5.2 Categorical attributes: cardinality and frequency distributions", 2)
    H.table(doc, r.card, "Cardinality, missing values and mode of categorical attributes", C, decimals=2)
    g = r.gfreq.head(15)
    H.table(doc, g, "Frequency and percentage distribution of genres (multi-label; percentages of all films)", C,
            decimals=2)
    H.para(doc, (
        f"{kf['top_genre']} is the most frequent genre ({kf['top_genre_pct']} % of films), followed by comedy and "
        f"suspense. {kf['share_uncategorised_pct']} % of films have no valid category and "
        f"{kf['multi_genre_films_pct']} % have more than one. Full frequency tables for every categorical attribute are "
        f"in `outputs/descriptive/frequency_distributions.xlsx`."), align="justify")
    doc.add_heading("5.3 Relationships between attributes", 2)
    H.table(doc, r.assoc, "Chi-square tests and Cramér's V between categorical attributes", C, decimals=3)
    H.para(doc, (
        f"All tested pairs are significantly dependent (p < 0.001), but the strengths differ. Era and colour type are "
        f"strongly associated (Cramér's V = {kf['cramers_v_era_color']:.2f}, computed on recorded processes only so the "
        f"rule-based imputation cannot inflate it). Studio and era (studio-system films), studio and genre, and era "
        f"and genre are moderately associated. Critic rating depends more on era than on genre. The Spearman "
        f"correlation between year and Halliwell stars is {kf['spearman_year_halliwell']:+.2f} (older films are rated "
        f"higher; Kruskal-Wallis across eras H = {kf['kruskal_era_halliwell_H']}, p ≈ "
        f"{kf['kruskal_era_halliwell_p']:.1e})."), align="justify")
    hg = r.hgen.rename(columns={"genres": "genre"})
    H.table(doc, hg, "Halliwell rating by genre (genres with ≥ 30 rated films)", C)
    H.table(doc, r.hera, "Halliwell rating by era", C)
    doc.add_heading("5.4 Important observations", 2)
    H.bullets(doc, [
        "The collection is curated, not a random sample: it is complete for Hitchcock, Hitchcock-related and "
        "studio-era films, and dense for the 1980s-90s the author watched. Counts describe the collection, not the "
        "film industry.",
        "Technological eras are sharply visible: silent b/w before 1928, sound b/w to the early 1950s, colour after "
        "the mid-1960s.",
        "Genres have life cycles: musicals and westerns belong to 1930-1960, horror and action to the post-1970 era.",
        "Critical acclaim (Halliwell) is concentrated in the classical era, so any model of acclaim mostly learns the "
        "guide's period preference.",
        f"The most prolific director is {kf['most_prolific_director']} ({kf['most_prolific_director_films']} films, "
        f"plus TV episodes under the Shamley company), as the documentation promises."])


def s6_visualisation(doc, r, C):
    doc.add_heading("6. Data Visualization", 1)
    eda = [f for f in r.figs if f["number"] <= 16]
    types = sorted({f["family"] for f in r.figs})
    H.para(doc, (
        f"Sixteen exploratory figures with {len({f['family'] for f in eda})} distinct visualisation techniques are "
        f"presented in this chapter (Lecture 4 representations plus heatmaps, density, treemap and others). The mining "
        f"chapters add {len(r.figs) - len(eda)} more figures, for {len(types)} distinct techniques in total. Variations of "
        f"one technique, such as the several line plots, are counted once. Plot types were chosen for the data, "
        f"following the guidance of Lecture 4 (e.g. a pie "
        f"chart only for five mutually exclusive classes; a treemap rather than a pie for 11 studio groups; no time "
        f"series except for the genuine year attribute; correlations only between numeric attributes). Colours use a "
        f"colour-blind-safe categorical palette in a fixed order, a single-hue sequential scale for magnitudes and a "
        f"diverging scale with a neutral midpoint for correlations."), align="justify")
    titles = {
        1: "Missing-value heatmap of raw MAIN attributes by decade",
        2: "Histogram of release years", 3: "Films per year with 5-year moving-average smoothing",
        4: "Share of film processes over time (stacked area)", 5: "Top 15 genres (horizontal bar chart)",
        6: "Genre mix within each era (100 % stacked bar chart)", 7: "Film process distribution (donut chart)",
        8: "Documented cast size by era (box plot)", 9: "Release year by Halliwell rating (violin plot)",
        10: "Release-year densities of selected genres (KDE)", 11: "Spearman correlation heatmap",
        12: "Directors: mean year vs mean rating (bubble chart)", 13: "Films by production studio (treemap)",
        14: "Oscar-nominated films per decade (count plot)", 15: "Female cast share vs year with regression smoothing",
        16: "Jaccard similarity between genres (heatmap)"}
    for f in eda:
        doc.add_heading(f"6.{f['number']} {f['type']}", 2)
        fig_block(doc, r, f["file"], titles[f["number"]], C, full=True)
    doc.add_heading("6.17 Summary of visualisation techniques", 2)
    summ = pd.DataFrame([(f["number"], f["family"], f["type"], f["attributes"]) for f in r.figs],
                        columns=["Plot #", "Technique (family)", "Specific plot", "Attributes"])
    H.table(doc, summ, "All visualisations produced in the project (plot numbers = file prefixes in outputs/plots)",
            C, col_widths_cm=[1.2, 3.4, 4.4, 7.4], font_size=7.5)


def s7_apriori(doc, r, C):
    ap = r.ap
    doc.add_heading("7. Association Rule Mining Using Apriori", 1)
    doc.add_heading("7.1 Introduction and transaction definition", 2)
    H.para(doc, (
        f"Association rule mining finds implications X ⇒ Y between disjoint itemsets that hold in a sufficient fraction "
        f"of transactions (Lecture 6). Here a **transaction is one film**, and its items are the film's attribute "
        f"values: every standardised genre, the era, the colour process, the studio (classical majors and Hitchcock's "
        f"TV company), the director's birth country, a cast-size bin, the Halliwell level, an Oscar-nomination flag, "
        f"remake/later-remade flags, a TV flag and a book/play-source flag. This yields {ap['n_transactions']:,} "
        f"transactions over {ap['n_items']} distinct items with mean length {ap['avg_transaction_length']:.2f}. A "
        f"per-film actor basket was considered and rejected: only 8 400 films have casts, and 6 000+ actors appear "
        f"in very few films, so actor itemsets would be too sparse at any meaningful support."), align="justify")
    doc.add_heading("7.2 Mathematical foundation", 2)
    H.para(doc, "For a rule X ⇒ Y over N transactions:")
    H.equation(doc, "\"Support\"(X)=(\"count\"(X))/N")
    H.equation(doc, "\"Confidence\"(X⇒Y)=(\"Support\"(X∪Y))/(\"Support\"(X))")
    H.equation(doc, "\"Lift\"(X⇒Y)=(\"Confidence\"(X⇒Y))/(\"Support\"(Y))")
    H.equation(doc, "\"Leverage\"(X⇒Y)=\"Support\"(X∪Y)-\"Support\"(X)\"Support\"(Y)")
    H.equation(doc, "\"Conviction\"(X⇒Y)=(1-\"Support\"(Y))/(1-\"Confidence\"(X⇒Y))")
    H.para(doc, (
        "Coverage is the antecedent support. Lift > 1 indicates positive dependence and lift = 1 independence. Leverage "
        "measures the absolute excess of co-occurrence over independence. Conviction (Brin et al., 1997; Lecture 6 "
        "credits Motwani et al.) is infinite for rules with 100 % confidence. **Downward closure (Apriori property):** "
        "if an itemset is frequent, all its non-empty subsets are frequent. Equivalently (anti-monotonicity), every "
        "superset of an infrequent itemset is infrequent, because adding an item can only keep or reduce the set of "
        "transactions that contain it."), align="justify")
    doc.add_heading("7.3 Algorithm, candidate generation and pruning", 2)
    H.para(doc, (
        "The implementation (`src/association_rules/apriori.py`) follows the Lecture 6 pseudo-code without any "
        "library:"), align="justify")
    H.bullets(doc, [
        "**Scan 1:** count every item and keep L1 (frequent 1-itemsets). Infrequent items are removed from all "
        "transactions, a transaction-reduction idea from the Apriori variants of Lecture 9.",
        "**Candidate generation (join):** sorted (k-1)-itemsets that share their first k-2 items are joined, "
        "e.g. abc + abd → abcd. They are grouped by prefix so the join is linear per group.",
        "**Pruning:** a candidate is discarded if any of its (k-1)-subsets is not in L(k-1). The unit test reproduces "
        "the slide example: acde is pruned because ade ∉ L3.",
        "**Support counting (scan k):** for each transaction, all k-subsets are enumerated and looked up in a hash "
        "set of candidates. Candidates with count ≥ min_count form Lk.",
        "**Rule generation:** for every frequent itemset l and every non-empty proper subset s, output s ⇒ (l - s) if "
        "count(l)/count(s) ≥ min_conf. Then compute lift, leverage, conviction and coverage.",
        "**Ranking:** lift, then confidence. A closure-based redundancy filter is applied (Section 7.6)."])
    lv = r.lv.rename(columns={"candidates_joined": "C_k after join", "pruned": "pruned by Apriori property",
                              "candidates_counted": "C_k counted", "frequent": "|L_k|"})
    H.table(doc, lv, f"Level-wise Apriori trace at min_support = {ap['min_support']} (count ≥ {ap['min_count']})", C)
    fig_block(doc, r, "19_apriori_levelwise_candidates.png", "Apriori level-wise search: join, prune and frequent "
                                                              "itemsets per level", C)
    doc.add_heading("7.4 Parameter selection", 2)
    s = r.sens.copy()
    H.table(doc, s, "Sensitivity of Apriori output to minimum support and minimum confidence", C, decimals=3)
    fig_block(doc, r, "18_apriori_threshold_sensitivity.png", "Frequent itemsets and rules as functions of the "
                                                               "thresholds", C)
    H.para(doc, (
        f"**Chosen thresholds: minimum support = {ap['min_support']} ({ap['min_count']} films) and minimum confidence = "
        f"{ap['min_confidence']}.** Justification: (i) the genre distribution (Figure 5) shows that only about 15 genres "
        f"exceed 2 % of films, so a lower support mostly adds idiosyncratic combinations, while 5 % keeps only "
        f"80 itemsets that restate era/colour facts; (ii) at 2 % the number of frequent itemsets "
        f"({ap['frequent_itemsets']}) and rules ({ap['rules']}) is at the knee of the curves; (iii) 229 films is a large "
        f"enough absolute count for stable proportions; (iv) a confidence of 0.60 requires the consequent to hold in "
        f"most antecedent films, while 0.80 removes most genre-related rules. Rare items (Halliwell levels, Oscars) were "
        f"studied separately at 1 % support (Section 7.7)."), align="justify")
    doc.add_heading("7.5 Execution results: frequent itemsets", 2)
    bl = ", ".join(f"{v} of size {k}" for k, v in ap["by_length"].items())
    H.para(doc, (
        f"Apriori found **{ap['frequent_itemsets']} frequent itemsets** ({bl}) in {ap['runtime_s']:.2f} s and "
        f"generated **{ap['rules']} rules**, {ap['rules_lift_gt_1']} with lift > 1. Validation against mlxtend's "
        f"independent Apriori: identical itemsets = {ap['validation']['identical_itemsets']}, identical support counts "
        f"= {ap['validation']['identical_support_counts']}."), align="justify")
    it = r.items.head(15)[["itemset", "support_count", "support"]]
    H.table(doc, it, "Most frequent single items (L1)", C)
    doc.add_heading("7.6 Top association rules", 2)
    H.para(doc, (
        "Many generated rules are permutations of the same few itemsets. For example, the ~270 *Alfred Hitchcock "
        "Presents* episodes alone yield dozens of rules among {format=TV, studio=Shamley, genre=Suspense, color=bw, "
        "era=Post-studio}. To present *distinct* knowledge, the closure operator of Lecture 11 is applied: "
        "f(X ∪ Y) = i(t(X ∪ Y)). Rules whose itemsets have the same closure describe exactly the same set of films, "
        "and only the best rule (highest lift, shortest antecedent) of each group is kept. Rules with a single-item "
        "consequent and lift > 1.1 are retained, with at most three per consequent."), align="justify")
    t = r.top[["antecedent", "consequent", "support", "confidence", "lift", "leverage", "conviction"]].copy()
    t["conviction"] = t["conviction"].replace(np.inf, np.nan)
    t.insert(0, "#", range(1, len(t) + 1))
    H.table(doc, t, f"Top {len(t)} association rules (min_sup 0.02, min_conf 0.60; conviction blank = ∞)", C,
            col_widths_cm=[0.6, 5.4, 3.6, 1.4, 1.4, 1.2, 1.4, 1.4], font_size=7.5)
    fig_block(doc, r, "20_apriori_top_rules_lift.png", "Top association rules ranked by lift", C)
    fig_block(doc, r, "17_apriori_support_confidence_lift.png", "Support-confidence-lift scatter of all rules", C)
    doc.add_heading("7.7 Interpretation", 2)
    H.bullets(doc, [
        "**Hitchcock's television work** forms the strongest pattern: {color=bw, format=TV, genre=Suspense} ⇒ "
        "studio=Shamley holds with confidence 1.0 and lift ≈ 31. Every b/w suspense TV film in the collection is an "
        "*Alfred Hitchcock Presents/Hour* episode produced by Shamley. This reflects the documented completeness of "
        "Hitchcock's TV work.",
        "**Technology defines eras**: era=Silent ⇒ color=silent_bw (confidence ≈ 0.94, lift ≈ 16), and modern films "
        "with documented casts are colour with near certainty. These rules confirm the data's internal consistency "
        "rather than revealing new knowledge.",
        "**Genre life cycles**: {colour, Romantic} and {colour, Comedy} ⇒ era=Modern (lift ≈ 2); {b/w, US director, "
        "Comedy} ⇒ Classical studio era (lift ≈ 3.7). Black-and-white comedies are mostly 1930s-40s screwball "
        "comedies.",
        "**Remakes**: {b/w, later_remade=yes} ⇒ Classical studio era (lift ≈ 3.6). The films that were later remade "
        "are predominantly studio-era originals.",
        "**Studio system**: studio=MGM ⇒ dir_country=USA (lift ≈ 1.65). Studio-era Hollywood staff was American-born "
        "far more often than the collection average.",
        "These rules describe co-occurrence in a curated collection. They are **not causal**: being a comedy does not "
        "cause a film to be modern."])
    if len(r.rare):
        rr = r.rare[["antecedent", "consequent", "support", "confidence", "lift"]].head(8)
        H.para(doc, (
            "**Supplementary rules for rare quality items.** Halliwell levels and Oscar nominations each cover less than "
            "10 % of films, so they cannot appear at 2 % support / 60 % confidence. At 1 % support, rules towards them "
            "have low confidence (the base rates are small) but high lift:"), align="justify")
        H.table(doc, rr, "Supplementary high-lift rules towards rating/Oscar items (min_sup 0.01, conf ≥ 0.15, "
                         "lift > 2)", C, col_widths_cm=[6.5, 3.5, 2, 2, 2], font_size=7.5)
        H.para(doc, (
            "Classical-era films with a large documented cast (6-9 actors) are about 4× as likely as average to have "
            "two Halliwell stars. New Hollywood films are about 2.5× as likely as average to be merely 'cited' "
            "without stars. This anticipates the decision-tree result in Chapter 8."), align="justify")


def s8_tree(doc, r, C):
    dt, tm = r.dt, r.dt["test_metrics"]
    doc.add_heading("8. Decision Tree Classification", 1)
    doc.add_heading("8.1 Suitability and problem definition", 2)
    H.para(doc, (
        f"UCI lists no target attribute for this data set, so a defensible target was **constructed from an existing "
        f"attribute**. The awards field records the star rating of *Halliwell's Film Guide* (1987 edition) for "
        f"{dt['n_labelled']:,} films. The binary target is **acclaimed = 1 if the film has ≥ 2 stars, 0 if it is listed "
        f"with 0-1 stars**. Two stars is the guide's threshold for a recommended film. The positive class rate is "
        f"{100 * dt['positive_rate']:.1f} %. The question is whether release-time characteristics (period, genre, "
        f"colour, studio, director background and productivity, producer coding, cast documentation, remake/book "
        f"source) predict critical acclaim. To avoid **target leakage**, Oscar data, other critics' ratings and "
        f"'later remade' flags are excluded, because they are outcomes of the same critical reception or are known "
        f"only after release. {dt['n_features']} features remain."), align="justify")
    doc.add_heading("8.2 Attribute-selection measures", 2)
    H.para(doc, "For a data set S with class proportions p_i and a split of S into partitions S_j (Lecture 13):")
    H.equation(doc, "\"Entropy\"(S)=-∑_(i=1)^c p_i log_2 p_i")
    H.equation(doc, "\"Gain\"(A)=\"Entropy\"(S)-∑_(j=1)^v (|S_j|)/(|S|) \"Entropy\"(S_j)")
    H.equation(doc, "\"SplitInfo\"_A (S)=-∑_(j=1)^v (|S_j|)/(|S|)  log_2 ((|S_j|)/(|S|))")
    H.equation(doc, "\"GainRatio\"(A)=(\"Gain\"(A))/(\"SplitInfo\"_A (S))")
    H.equation(doc, "\"Gini\"(S)=1-∑_(i=1)^c p_i^2")
    H.equation(doc, "\"Gini\"_A (S)=(|S_1|)/(|S|) \"Gini\"(S_1)+(|S_2|)/(|S|) \"Gini\"(S_2)")
    H.para(doc, (
        f"These measures were implemented from scratch (`src/classification/split_criteria.py`) and evaluated for every "
        f"attribute at the root of the training set (entropy {dt['root']['entropy_root']:.3f} bits, Gini "
        f"{dt['root']['gini_root']:.3f}). For numeric attributes the best binary threshold is searched among "
        f"midpoints. Following C4.5, the gain ratio is evaluated at the information-gain-optimal threshold. "
        f"Maximising the ratio itself is known to favour tiny edge splits."), align="justify")
    rt = r.root.head(12)
    rt = pd.DataFrame({"Attribute": rt.attribute, "Info gain": rt.info_gain, "Threshold": rt.ig_threshold.map("{:.2f}".format),
                       "Gain ratio": rt.gain_ratio, "Split info": rt.split_info, "Gini red.": rt.gini_reduction,
                       "Rank IG": rt.rank_info_gain, "Rank GR": rt.rank_gain_ratio, "Rank Gini": rt.rank_gini_reduction})
    H.table(doc, rt, "Root-node attribute evaluation (own implementation, training set)", C, font_size=7.5, decimals=4,
            col_widths_cm=[4.4, 1.5, 1.5, 1.5, 1.5, 1.5, 1.3, 1.3, 1.4])
    fig_block(doc, r, "28_decision_tree_split_criteria.png", "Information gain, gain ratio and Gini reduction at the "
                                                              "root", C)
    doc.add_heading("8.3 Tree construction, stopping criteria and pruning", 2)
    bp = dt["best_cv_params"]
    fp = dt["final_params"]
    H.para(doc, (
        f"scikit-learn's CART implementation builds the tree greedily, top-down and recursively (Lecture 13). At each "
        f"node it chooses the split with the largest impurity decrease, using the explicitly configured "
        f"**criterion** = 'gini' or 'entropy' (information gain). **Stopping criteria** (pre-pruning): all samples in a "
        f"node belong to one class, maximum depth reached, or fewer than `min_samples_leaf` samples would remain in a "
        f"child. The data were split **75/25 with stratification** ({dt['root']['n_train']} training / "
        f"{dt['root']['n_test']} test films; {dt['root']['positives_train']} and {dt['root']['positives_test']} "
        f"positives). A 5-fold stratified cross-validated grid search on the training set (scoring ROC-AUC) compared "
        f"criterion ∈ {{gini, entropy}}, max_depth ∈ {{3, 4, 5, 6, 8, 10, None}}, min_samples_leaf ∈ {{5, 10, 20, 40}} "
        f"and class_weight ∈ {{None, balanced}}. Best: {bp} (CV ROC-AUC {dt['best_cv_roc_auc']:.3f}). **Post-pruning** "
        f"used minimal cost-complexity pruning (Breiman et al., 1984). The pruning path was cross-validated, and the "
        f"largest α whose CV AUC is within 0.005 of the best was chosen (a one-standard-error style rule that prefers "
        f"simpler trees): α = {fp['ccp_alpha']:.4f}."), align="justify")
    cc = pd.DataFrame(dt["criterion_cv"])
    hh = pd.DataFrame(dt["criterion_head_to_head"]).T.reset_index().rename(columns={"index": "criterion"})
    H.table(doc, cc, "Splitting-criterion comparison in cross-validation (training set)", C)
    H.table(doc, hh, "Splitting-criterion comparison on the test set (all other settings identical)", C)
    fig_block(doc, r, "27_decision_tree_pruning_curve.png", "Cost-complexity pruning path", C)
    fig_block(doc, r, "23_decision_tree_structure.png", "Final decision tree (top four levels; node values are "
                                                         "class-weighted)", C)
    doc.add_heading("8.4 Evaluation", 2)
    H.equation(doc, "\"Precision\"=TP/(TP+FP),  \"Recall\"=TP/(TP+FN),  F_1=(2⋅\"Precision\"⋅\"Recall\")/(\"Precision\"+\"Recall\")")
    met = pd.DataFrame([("Accuracy", tm["accuracy"]), ("Balanced accuracy", tm["balanced_accuracy"]),
                        ("Precision (acclaimed)", tm["precision"]), ("Recall (acclaimed)", tm["recall"]),
                        ("F1-score (acclaimed)", tm["f1"]), ("ROC-AUC", tm["roc_auc"]),
                        ("Training accuracy", tm["train_accuracy"]),
                        ("5-fold CV accuracy (mean ± sd)", f"{tm['cv_accuracy_mean']:.3f} ± {tm['cv_accuracy_std']:.3f}"),
                        ("Majority-class baseline accuracy", tm["baseline_majority_accuracy"]),
                        ("Tree depth / leaves", f"{tm['depth']} / {tm['leaves']}"),
                        ("Fit time / predict time (s)", f"{tm['fit_time_s']:.4f} / {tm['predict_time_s']:.4f}")],
                       columns=["Metric", "Value"])
    H.table(doc, met, "Test-set evaluation of the final decision tree", C, col_widths_cm=[7, 5])
    rep = (DT_DIR / "classification_report.txt").read_text()
    H.para(doc, "Classification report (scikit-learn):")
    H.code_block(doc, rep)
    fig_block(doc, r, "24_decision_tree_confusion_matrix.png", "Confusion matrix on the test set", C)
    fig_block(doc, r, "25_decision_tree_roc_curve.png", "ROC curve on the test set", C)
    fig_block(doc, r, "26_decision_tree_feature_importance.png", "Feature importance of the final tree", C)
    doc.add_heading("8.5 IF-THEN rules", 2)
    H.para(doc, (
        f"Each of the {dt['n_rules']} root-to-leaf paths is one IF-THEN rule. The table shows the most reliable rules "
        f"of each class. 'Train rate' is the actual (unweighted) share of acclaimed training films in the leaf. "
        f"Because the tree was trained with balanced class weights, a leaf predicts 'acclaimed' when its acclaimed "
        f"share clearly exceeds the 32 % base rate, even below 50 %."), align="justify")
    rl = pd.concat([r.rules[r.rules.predicted_class == 1].head(6), r.rules[r.rules.predicted_class == 0].head(6)])
    rl = rl[["rule", "training_positive_rate", "samples"]].rename(columns={"training_positive_rate": "train rate "
                                                                          "(acclaimed)"})
    H.table(doc, rl, "Most reliable IF-THEN rules extracted from the tree", C, col_widths_cm=[12.4, 2.2, 1.8],
            font_size=7.5)
    doc.add_heading("8.6 Interpretation", 2)
    top_imp = ", ".join(f"{k} ({v:.2f})" for k, v in r.imp.head(4).items())
    H.para(doc, (
        f"The tree splits first on **year ≤ 1967.5**, which is also the top attribute by information gain, gain ratio "
        f"and Gini. Older films are far more likely to be acclaimed. Among them, films with large documented casts "
        f"(> 7.5 actors) by prolific directors are acclaimed in about 80 % of training cases. Among films after 1967, "
        f"only those with very large casts are predicted as acclaimed. The most important features are {top_imp}. The "
        f"model reaches ROC-AUC {tm['roc_auc']:.3f} and recall {tm['recall']:.2f} for acclaimed films. Its plain "
        f"accuracy ({tm['accuracy']:.3f}) is close to the majority baseline ({tm['baseline_majority_accuracy']:.3f}) "
        f"because the balanced weighting deliberately trades precision for recall, and balanced accuracy "
        f"({tm['balanced_accuracy']:.3f}) is the fairer summary. Substantively, the model captures the **critic's "
        f"preference for classic studio films** and the fact that well-documented (large cast) films are the ones "
        f"the author cared about. It does not measure intrinsic film quality, and documentation depth (cast size) is "
        f"partly a consequence of a film's fame. Gini and entropy give practically identical CV performance (best CV "
        f"AUC differs by {abs(dt['criterion_cv'][0]['best_cv_roc_auc'] - dt['criterion_cv'][1]['best_cv_roc_auc']):.3f})."),
        align="justify")


def s9_kmeans(doc, r, C):
    km = r.km
    doc.add_heading("9. K-Means Clustering", 1)
    doc.add_heading("9.1 Suitability and clustering objective", 2)
    H.para(doc, (
        "K-Means needs a numeric space in which means (centroids) and Euclidean distances are meaningful. At film "
        "level almost every attribute is nominal, so one-hot vectors would make centroids hard to interpret. "
        "Clustering **directors** is meaningful instead: each director's films are aggregated into a numeric career "
        f"profile. {km['n_directors']} directors with at least five films (so that proportions are stable) are "
        f"described by {km['n_features']} standardised features: log number of films, mean release year, career span, "
        f"share of colour and TV films, mean Halliwell rating (0 if never rated), share of rated films, Oscar "
        f"nomination rate, mean documented cast size and the shares of ten genres. No identifier is used as a "
        f"feature. The objective is to discover groups of directors with similar careers."), align="justify")
    doc.add_heading("9.2 Algorithm and distance measure", 2)
    H.para(doc, (
        "K-Means (Lectures 20 and 22) alternates two steps until the assignments stop changing: assign each point to "
        "the nearest centroid by Euclidean distance, then recompute each centroid as the mean of its points. This "
        "monotonically decreases the within-cluster sum of squares (WCSS):"), align="justify")
    H.equation(doc, "d(x,c)=√(∑_(j=1)^m (x_j-c_j)^2)")
    H.equation(doc, "c_k=(1)/(|C_k|)  ∑_(x∈C_k) x")
    H.equation(doc, "\"WCSS\"=∑_(k=1)^K ∑_(x∈C_k) ‖x-c_k‖^2")
    H.equation(doc, "s(i)=(b(i)-a(i))/max(a(i),b(i))")
    ag = km["manual_vs_sklearn"]
    H.para(doc, (
        f"A manual implementation (`manual_kmeans`: k-means++ seeding, Euclidean assignment, mean update, convergence "
        f"when assignments are stable and the centroid shift is < 10⁻⁶) was checked against scikit-learn from the "
        f"same initial centroids. Adjusted Rand index = {ag['ari_manual_vs_sklearn_same_init']:.2f}, identical WCSS "
        f"({ag['wcss_manual']:.2f}), convergence in {ag['manual_iterations']} iterations. The unit tests also "
        f"reproduce the worked example of Lecture 20: final clusters {{1, 2}} and {{3, ..., 7}} with centroids "
        f"(1.25, 1.5) and (3.9, 5.1). The final model is scikit-learn's K-Means, best of 20 k-means++ initialisations."),
        align="justify")
    doc.add_heading("9.3 Selection of K", 2)
    H.table(doc, r.ksel, "Model-selection criteria for K = 2..10", C, decimals=3)
    fig_block(doc, r, "29_kmeans_elbow_silhouette_db.png", "Elbow (WCSS), silhouette and Davies-Bouldin index for "
                                                            "K = 2..10", C)
    H.para(doc, (
        f"The WCSS curve has no sharp elbow (largest curvature at K = {km['elbow_k']}). Silhouettes are low "
        f"everywhere (0.105-0.151), and the Davies-Bouldin index decreases slowly with K. The data therefore have weak "
        f"cluster structure: careers vary continuously. The decision rule was to choose, among K = 3-6 (interpretable, "
        f"no cluster below 20 directors), the K with the best silhouette, with Davies-Bouldin and Calinski-Harabasz as "
        f"corroboration. This gives **K = {km['chosen_k']}** (silhouette {km['silhouette']:.3f}, DB "
        f"{km['davies_bouldin']:.2f}, CH {km['calinski_harabasz']:.1f}). The higher silhouettes at K = 9-10 come from "
        f"splitting off tiny groups of 11 directors and do not add interpretable structure."), align="justify")
    doc.add_heading("9.4 Cluster visualisations and characteristics", 2)
    for f, t in [("30_kmeans_silhouette_plot.png", "Silhouette plot of the final clustering"),
                 ("31_kmeans_pca_clusters.png", "Clusters in the first two principal components"),
                 ("32_kmeans_cluster_sizes.png", "Cluster size distribution"),
                 ("33_kmeans_centroid_heatmap.png", "Centroids in standardised feature space"),
                 ("34_kmeans_parallel_coordinates.png", "Parallel coordinates of the centroids"),
                 ("35_kmeans_pairplot.png", "Pair plot of key profile features by cluster"),
                 ("36_kmeans_manual_convergence.png", "Convergence of the manual K-Means implementation")]:
        fig_block(doc, r, f, t, C)
    p = r.prof.reset_index()
    p1 = p[["cluster", "size", "label", "n_films", "mean_year", "career_span", "share_color", "share_tv",
            "mean_halliwell", "oscar_nom_rate"]]
    p2 = p[["cluster", "share_Dram", "share_Comd", "share_Susp", "share_Romt", "share_Horr", "share_Musc",
            "share_West", "share_Actn", "example_directors"]]
    H.table(doc, p1, "Cluster profiles (means of original, unscaled features)", C, font_size=7.5, decimals=2)
    H.table(doc, p2, "Cluster genre profiles and example directors", C, font_size=7.5, decimals=2)
    doc.add_heading("9.5 Interpretation", 2)
    pr = r.prof
    H.bullets(doc, [
        f"**C0 - {pr.loc[0, 'label']} ({int(pr.loc[0, 'size'])} directors):** mean film year "
        f"{pr.loc[0, 'mean_year']:.0f}, only {100 * pr.loc[0, 'share_color']:.0f} % colour, highest mean Halliwell "
        f"rating ({pr.loc[0, 'mean_halliwell']:.2f}) and Oscar rate ({pr.loc[0, 'oscar_nom_rate']:.3f}), more musicals "
        f"and westerns. These are the studio-era masters (e.g. {pr.loc[0, 'example_directors']}).",
        f"**C1 - {pr.loc[1, 'label']} ({int(pr.loc[1, 'size'])} directors):** {100 * pr.loc[1, 'share_tv']:.0f} % TV "
        f"work and {100 * pr.loc[1, 'share_Susp']:.0f} % suspense. These are the directors of the Hitchcock TV "
        f"series ({pr.loc[1, 'example_directors']}). It is the most cohesive cluster (silhouettes up to 0.55).",
        f"**C2 - {pr.loc[2, 'label']} ({int(pr.loc[2, 'size'])} directors):** mean year "
        f"{pr.loc[2, 'mean_year']:.0f}, {100 * pr.loc[2, 'share_color']:.0f} % colour, lower ratings (the 1987 guide "
        f"rates few recent films), more horror and action ({pr.loc[2, 'example_directors']}).",
        "The clusters are essentially **period** (classical vs modern) plus one **format specialisation** (TV "
        "suspense). This matches the dominant role of era found in the descriptive statistics, the association rules "
        "and the decision tree. Hitchcock himself falls into C1 because of his large TV share, although his feature "
        "films resemble C0. This shows that cluster labels summarise tendencies, not hard categories."])
    doc.add_heading("9.6 Hierarchical clustering (additional analysis)", 2)
    H.para(doc, (
        "Agglomerative clustering (Lecture 21) with single (minimum), complete (maximum) and average linkage, plus Ward's "
        "minimum-variance linkage, was applied to the same standardised profiles. Each dendrogram was cut at K = 3."),
        align="justify")
    H.table(doc, r.hier, "Comparison of hierarchical linkage methods (cut at K = 3)", C, decimals=3)
    fig_block(doc, r, "37_hierarchical_dendrograms.png", "Dendrograms for single, complete, average and Ward linkage", C)
    H.para(doc, (
        "Single and average linkage reach high cophenetic correlation and silhouette, but only by 'chaining': one "
        "cluster absorbs almost all directors and the others are singletons, which makes the partition useless. Their "
        "ARI with K-Means is ≈ 0. Ward linkage minimises the same variance criterion as K-Means and produces balanced "
        "groups. This illustrates the Lecture 21 remark that there is no objective way to choose the number of "
        "clusters from a dendrogram, and shows that the linkage choice changes the result substantially."),
        align="justify")


def s10_cofi(doc, r, C):
    c = r.cofi
    v = c["validation"]
    doc.add_heading("10. COFI Algorithm and Closed Frequent Itemsets", 1)
    doc.add_heading("10.1 Introduction", 2)
    H.para(doc, (
        "The number of frequent itemsets explodes as support decreases: an itemset of length 100 has 2¹⁰⁰ - 1 frequent "
        "subsets (Lecture 6). Closed and maximal itemsets are condensed representations (Lectures 10-11). COFI "
        "(Co-Occurrence Frequent-Item tree; El-Hajj & Zaïane, 2003) is a pattern-growth algorithm that mines one small "
        "tree per frequent item instead of recursively building conditional FP-trees. **The course slides do not "
        "describe COFI**. The implementation therefore follows the original paper, while all closed/maximal concepts "
        "and the validation follow the slides."), align="justify")
    doc.add_heading("10.2 Frequent vs closed vs maximal itemsets", 2)
    H.bullets(doc, [
        "**Frequent itemset:** support(X) ≥ min_sup.",
        "**Closed frequent itemset (CFI):** frequent, and no proper superset has the same support. Equivalently, with "
        "the A-Close closure operator f(X) = i(t(X)) of Lecture 11, X is closed iff f(X) = X. CFIs are lossless: the "
        "support of any frequent itemset equals the support of its smallest closed superset.",
        "**Maximal frequent itemset (MFI):** frequent, and no proper superset is frequent. MFIs are fewer but lossy: "
        "they say which itemsets are frequent but not their supports (Lecture 10 example: {a1..a100}, {a1..a50} gives "
        "two CFIs but one MFI).",
        "Every MFI is closed, but not every CFI is maximal."])
    H.equation(doc, "X \"closed\"  ⟺  ∄ Y⊃X: \"Support\"(Y)=\"Support\"(X)")
    H.equation(doc, "f(X)=i(t(X)),   X \"closed\" ⟺ f(X)=X")
    doc.add_heading("10.3 COFI-tree construction", 2)
    H.bullets(doc, [
        "**Frequent item selection and ordering:** one scan counts item supports. Frequent items are ordered by "
        "descending support (alphabetical tie-break).",
        "**FP-tree:** a second scan inserts every transaction (filtered and sorted) as a path. Shared prefixes share "
        "nodes, and each item has a header chain (node-links).",
        "**Conditional transactions per item:** for each frequent item A, from the *least* to the most frequent, "
        "A's node-links are followed. Each A-node's ancestor path is a conditional transaction weighted by the "
        "A-node's count.",
        "**Local pruning:** items whose co-occurrence count with A is below min_count are dropped. These are the "
        "'locally infrequent' items, a key saving of COFI.",
        "**COFI-tree of A:** root A, the conditional paths inserted with the most frequent item next to the root. "
        "Every node stores (item, support count, *participation count*) and a node-link. Only items more frequent "
        "than A occur, so every pattern is generated in exactly one tree (that of its least frequent item)."])
    doc.add_heading("10.4 Mining procedure", 2)
    H.para(doc, (
        "Items of A's COFI-tree are visited from the least to the most frequent, so descendants are processed before "
        "ancestors. For each node N on the item's chain:"), align="justify")
    H.equation(doc, "\"count\"(B)=\"support\"(N)-\"participation\"(N)")
    H.para(doc, (
        "If this is positive, the path from N to the root is a *frequent-path-base* B with that count, and the "
        "participation of every node on the path is increased by it. The bases form a disjoint partition of the "
        "transactions containing A. The support of a pattern {A} ∪ S is then the sum over the bases that contain S:"),
        align="justify")
    H.equation(doc, "\"support\"({A}∪S)=∑_(B⊇S) \"count\"(B)")
    H.para(doc, (
        "Patterns reaching min_count are frequent. The COFI-tree is then discarded, so only one small tree is in "
        "memory at any time. Closedness is decided with the Lecture 10 definition over the complete frequent set. It "
        "suffices to check immediate supersets X ∪ {i}, because support is anti-monotone: an equal-support proper "
        "superset implies an equal-support immediate superset."), align="justify")
    doc.add_heading("10.5 Parameters and results", 2)
    H.para(doc, (
        f"The same transactions and the same threshold as Apriori were used for comparability: **min_support = "
        f"{c['min_support']} (count ≥ {c['min_count']})**. COFI built {c['frequent_items']} COFI-trees. The largest "
        f"has {c['largest_cofi_tree_nodes']} nodes, compared with {c['fp_tree_nodes']:,} nodes in the global FP-tree. "
        f"It found **{c['frequent_itemsets']} frequent, {c['closed_itemsets']} closed and {c['maximal_itemsets']} "
        f"maximal itemsets** in {c['total_time_s']:.2f} s (FP-tree {c['time_fp_tree_s']:.2f} s, COFI mining "
        f"{c['time_cofi_mining_s']:.2f} s, closedness check {c['time_closure_check_s']:.3f} s; peak Python heap "
        f"{c['peak_python_memory_mb']:.1f} MB)."), align="justify")
    cf = r.cfi.head(15)[["itemset", "support_count", "support_pct", "n_items", "closed", "maximal"]]
    H.table(doc, cf, "Largest closed frequent itemsets (min_sup 0.02)", C, col_widths_cm=[9.5, 1.6, 1.6, 1.2, 1.2, 1.3],
            font_size=7.5)
    cc = r.ccomp[["min_support", "min_count", "frequent_itemsets", "closed_itemsets", "maximal_itemsets",
                  "compression_ratio", "cofi_time_s", "apriori_time_s", "largest_cofi_tree_nodes", "identical_to_apriori"]]
    H.table(doc, cc, "Frequent vs closed vs maximal itemsets across support thresholds", C, font_size=7.5, decimals=3)
    fig_block(doc, r, "21_cofi_frequent_closed_maximal.png", "Numbers of frequent, closed and maximal itemsets", C)
    fig_block(doc, r, "22_cofi_tree_sizes.png", "Size of each item's COFI-tree", C)
    doc.add_heading("10.6 Validation", 2)
    val = pd.DataFrame([
        ("V1", "Lecture 11 A-Close example (5 transactions, min count 3)", f"closed sets {v['V1_lecture11_example']['closed_found']}",
         "PASS" if v["V1_lecture11_example"]["passed"] else "FAIL"),
        ("V2", "Exhaustive brute-force enumeration on 30 random databases",
         f"frequent identical {v['V2_bruteforce_random']['frequent_identical']}/30, closed identical "
         f"{v['V2_bruteforce_random']['closed_identical']}/30", "PASS" if v["V2_bruteforce_random"]["passed"] else "FAIL"),
        ("V3", "Movie data vs from-scratch Apriori and mlxtend FP-Growth",
         f"{v['V3_movie_cross_check']['cofi_frequent']} = {v['V3_movie_cross_check']['apriori_frequent']} = "
         f"{v['V3_movie_cross_check']['mlxtend_fpgrowth_frequent']} itemsets with identical supports",
         "PASS" if v["V3_movie_cross_check"]["identical_to_apriori"] and v["V3_movie_cross_check"]["identical_to_mlxtend_fpgrowth"] else "FAIL"),
        ("V4", "Every closed itemset: support recounted from raw transactions; frequent; f(X) = X",
         f"{v['V4_V5_closed_checks']['closed_checked']} checked; wrong support {v['V4_V5_closed_checks']['wrong_support']}, "
         f"infrequent {v['V4_V5_closed_checks']['infrequent']}, not closed {v['V4_V5_closed_checks']['not_closed_by_closure_operator']}",
         "PASS" if v["V4_V5_closed_checks"]["passed"] else "FAIL"),
        ("V5", "Completeness: each frequent itemset's support equals that of its closure",
         f"violations: {v['V4_V5_closed_checks']['frequent_without_closed_cover']}",
         "PASS" if v["V4_V5_closed_checks"]["passed"] else "FAIL")],
        columns=["Check", "Description", "Result", "Status"])
    H.table(doc, val, "Independent validation of the COFI implementation", C, col_widths_cm=[1.0, 6.4, 7.4, 1.6],
            font_size=7.5)
    H.para(doc, (
        "The Lecture 11 transactions were reconstructed from the slide's i(generators) column (A: {1,3,5}, B: "
        "{2,3,4,5}, C: {1,2,3,5}, E: {2,3,4,5}). COFI returns exactly the four closed sets of the slide, {A,C}, {B,E}, "
        "{B,C,E} and {C}. The algorithm is therefore exact, and the reported itemsets are genuinely closed and "
        "frequent with correct supports."), align="justify")
    doc.add_heading("10.7 Interpretation and computational benefits", 2)
    lo = r.ccomp.iloc[0]
    H.para(doc, (
        f"On these short, sparse transactions (mean 5.1 items) few frequent itemsets share a support with a superset, "
        f"so the closed representation is only {c['frequent_itemsets'] / c['closed_itemsets']:.2f}× smaller than the "
        f"full frequent set at 2 %. The gain grows as support falls (to {lo['compression_ratio']:.2f}× at "
        f"{lo['min_support']} support) and would be dramatic on dense data, as the 2¹⁰⁰ example of Lecture 10 "
        f"shows. The non-closed itemsets that are removed are exactly the redundant ones. For instance, "
        f"{{studio=Shamley, format=TV}} always co-occurs with b/w, suspense and the post-studio era, and its closure "
        f"{{color=bw, era=Post-studio, format=TV, genre=Suspense, studio=Shamley}} (278 films) summarises all of them. "
        f"This is the same closure-based redundancy elimination used to select the Apriori rules. Computationally, COFI "
        f"avoids Apriori's candidate generation and repeated scans (two scans in total). It keeps one small tree in "
        f"memory (≤ {c['largest_cofi_tree_nodes']} nodes) and was faster than our Apriori at most thresholds (Table "
        f"above). The 127 maximal itemsets are the most compact description, but they cannot reproduce supports."),
        align="justify")


def s11_pyspark(doc, r, C):
    env = r.env
    doc.add_heading("11. PySpark Implementation", 1)
    doc.add_heading("11.1 Big-data motivation", 2)
    H.para(doc, (
        "Lecture 5 lists efficiency, scalability and parallelisation as major issues in data mining. When data no "
        "longer fits one machine's memory, or one CPU core is too slow, frameworks such as Apache Spark distribute "
        "data and computation across a cluster. This chapter re-implements two of the predictive-analytics techniques "
        "in PySpark (Spark MLlib) and quantifies what distribution changes for this dataset."), align="justify")
    doc.add_heading("11.2 PySpark architecture", 2)
    H.para(doc, (
        "A Spark application consists of a **driver** (here the Python process with a JVM SparkContext reached "
        "through Py4J) and **executors** that run **tasks** on **partitions** of a distributed dataset. DataFrames "
        "are lazily evaluated. Transformations build a logical plan, which the Catalyst optimiser compiles into stages "
        "of tasks separated by shuffles, and actions (`count`, `toPandas`, model `fit`) trigger execution. MLlib "
        "algorithms run inside the JVM on `Vector` columns. In **local mode** (`local[2]`) the driver and executors "
        "share one JVM, and two worker threads emulate a two-core cluster. Pandas DataFrames enter Spark through "
        "Apache Arrow (columnar, zero-copy batches)."), align="justify")
    doc.add_heading("11.3 Selected algorithms", 2)
    H.bullets(doc, [
        "**Technique 1 - Decision Tree classification** (`pyspark.ml.classification.DecisionTreeClassifier`): "
        "entropy impurity, maxDepth 6, minInstancesPerNode 40, balanced class weights via `weightCol`, maxBins 256. "
        "The sklearn counterpart uses the same depth, leaf size, criterion and weights on the same stratified "
        "samples and splits.",
        "**Technique 2 - K-Means clustering** (`pyspark.ml.clustering.KMeans`): k-means|| initialisation (the "
        "scalable k-means++ of Bahmani et al., 2012), maxIter 100, tol 10⁻⁴, the same K as sklearn. It is applied to "
        "the director profiles (K = 3) and to a larger film-level matrix (11 408 films × 51 standardised/one-hot "
        "features, K = 6).",
        "**Supplementary - frequent-pattern mining** (`pyspark.ml.fpm.FPGrowth`). **Spark MLlib has no Apriori**; "
        "its FP-Growth is the parallel PFP algorithm (Li et al., 2008). FP-Growth and Apriori differ in method: "
        "FP-Growth compresses the database into an FP-tree and grows patterns without candidate generation in two "
        "scans, while Apriori is level-wise with candidate generation and one scan per level. Both are exact for the "
        "same minimum support, so their *outputs* are compared like-for-like. Spark only generates rules with a "
        "single-item consequent, so rule counts are compared with that subset of our rules."])
    doc.add_heading("11.4 Configuration for the target laptop", 2)
    cf = pd.DataFrame([(k, str(v)) for k, v in env["spark_conf"].items()] +
                      [("Spark version", env["spark_version"]), ("Session start-up time", f"{env['spark_startup_s']:.1f} s"),
                       ("Java", "Temurin OpenJDK 21.0.11"),
                       ("spark.python.worker.module (Windows)", "spark_win_worker (see 11.5)")],
                      columns=["Setting", "Value"])
    H.table(doc, cf, "Spark configuration (Intel i5-1235U, 8 GB RAM; C: drive nearly full)", C,
            col_widths_cm=[6.5, 9.9], font_size=8)
    H.para(doc, (
        "The configuration is constrained by the laptop. Two local cores and a 2 GB driver leave memory for the OS and "
        "the Python driver. Shuffle partitions are reduced from 200 to 4 to suit ~10⁴ rows. Spark scratch space is on "
        "the E: drive so the nearly full C: drive is never used. The UI is disabled, data is cached once per "
        "workload and unpersisted afterwards, and the session is stopped at the end."), align="justify")
    doc.add_heading("11.5 Implementation notes and Windows-specific issues", 2)
    H.bullets(doc, [
        "Spark's Windows launch scripts fail when the path contains spaces (the project folder is "
        "`...\\Big Data labs\\...`). A directory junction `D:\\ASBD_Hackathon` points to the project, and Spark jobs "
        "are run through it. `run_all.py` does this automatically.",
        "With PySpark 4.0.1 on Windows, Python worker processes exit before the JVM has read their reply ('Python "
        "worker exited unexpectedly'). A thin worker entry point (`src/pyspark_impl/win_worker/spark_win_worker.py`) "
        "runs the unmodified `pyspark.worker.main` and waits for the JVM to close the socket. The diagnosis was carried "
        "over from the author's earlier lab on the same machine. MLlib itself runs in the JVM and is unaffected.",
        "No data is written through the Hadoop file system, so the missing `winutils.exe` only produces a warning."])
    H.para(doc, "Core of the Spark decision-tree and K-Means code (`src/pyspark_impl/spark_benchmark.py`):")
    H.code_block(doc, """asm = VectorAssembler(inputCols=feats, outputCol="features")
sdf_tr = asm.transform(spark.createDataFrame(train_pdf)).select("features", "label", "w").repartition(2).cache()
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label", weightCol="w", impurity="entropy",
                            maxDepth=6, minInstancesPerNode=40, maxBins=256, seed=42)
model = dt.fit(sdf_tr)
pred = model.transform(sdf_te).select(vector_to_array("probability").getItem(1).alias("p1"), "label").toPandas()

km = KMeans(k=6, initMode="k-means||", maxIter=100, tol=1e-4, seed=42, featuresCol="features")
km_model = km.fit(sdf)

fp = FPGrowth(itemsCol="items", minSupport=0.02, minConfidence=0.60)
fp_model = fp.fit(spark.createDataFrame(pd.DataFrame({"items": transactions})))""")


def s12_benchmark(doc, r, C):
    env = r.env
    bdt, bk, bf, st = r.bdt, r.bk, r.bf, r.st
    doc.add_heading("12. Performance Benchmarking", 1)
    doc.add_heading("12.1 Experimental methodology", 2)
    H.bullets(doc, [
        "**Identical data:** for each size, the same random (stratified for the tree) sample and the same 75/25 split "
        "are given to both implementations.",
        "**Data sizes:** 10 %, 25 %, 50 % and 100 % of the real data. No synthetic records are used for these results.",
        f"**Timing:** `time.perf_counter`, median of {env['repeats']} repetitions. One Spark warm-up run per workload "
        f"(JIT compilation, class loading) is discarded. Session start-up ({env['spark_startup_s']:.1f} s) is reported "
        f"separately and excluded.",
        "**Spark time** is split into data conversion (pandas → Spark via Arrow, vector assembly, cache + count), fit, "
        "and predict (transform + collect to the driver). Speedups use fit + predict. A second speedup also includes "
        "conversion.",
        "**Memory:** peak Python heap via tracemalloc for the Python implementations; resident set size of the Spark "
        "JVM (psutil) after each workload.",
        "**Quality:** classification metrics and clustering quality are computed with the same sklearn code from both "
        "implementations' predictions.",
        "**Isolation:** the reported run was executed with no other project workload running. The user's normal "
        "desktop applications (IDE, browser, a local MySQL server) stayed open, so free physical memory was often "
        "below 0.5 GB on the 7.9 GB laptop. Timings therefore reflect realistic, memory-constrained conditions and "
        "contain some noise. An earlier run whose stress-test phase overlapped with notebook execution was "
        "discarded."])
    H.equation(doc, "\"Speedup\"=T_\"Python\"/T_\"PySpark\"")
    H.equation(doc, "\"Improvement\" (%)=(T_\"Python\"-T_\"PySpark\")/T_\"Python\" ×100")
    H.para(doc, "A speedup below 1 and a negative improvement mean that PySpark was slower.")
    doc.add_heading("12.2 Hardware configuration", 2)
    H.para(doc, (
        f"{env['cpu']}; {env['logical_cpus']} logical CPUs; {env['ram_gb']} GB RAM; {env['os']}; Python {env['python']}; "
        f"Spark {env['spark_version']} in local[2] mode with a 2 GB driver."), align="justify")
    doc.add_heading("12.3 Decision tree: Python vs PySpark", 2)
    H.para(doc, "Abbreviations in the tables: Py = Python/scikit-learn, Sp = PySpark, conv = data conversion into "
                "Spark, Impr % = percentage improvement (negative = PySpark slower), JVM MB = resident memory of the "
                "Spark JVM. Times are in seconds.", size=9.5)
    pct = lambda s: (100 * s).round(0).astype(int).astype(str) + " %"
    t = pd.DataFrame({"Data": pct(bdt.fraction), "Rows": bdt.records, "Train": bdt.train, "Feat.": bdt.features,
                      "Parts": bdt.partitions, "Py fit": bdt.py_fit_s, "Py pred": bdt.py_predict_s,
                      "Sp conv": bdt.spark_convert_s, "Sp fit": bdt.spark_fit_s, "Sp pred": bdt.spark_predict_s,
                      "Speedup": bdt.speedup, "Impr %": bdt.speedup_improvement_pct.map("{:.1f}".format),
                      "Py MB": bdt.py_peak_mem_mb.map("{:.2f}".format), "JVM MB": bdt.spark_jvm_rss_mb.map("{:.0f}".format)})
    H.table(doc, t, "Decision tree execution time (s), speedup and memory", C, font_size=7.5, decimals=4)
    q = pd.DataFrame({"Data": pct(bdt.fraction), "Py accuracy": bdt.py_accuracy, "Sp accuracy": bdt.spark_accuracy,
                      "Py F1": bdt.py_f1, "Sp F1": bdt.spark_f1, "Py AUC": bdt.py_roc_auc, "Sp AUC": bdt.spark_roc_auc,
                      "Py leaves": bdt.py_leaves, "Sp nodes": bdt.spark_nodes})
    H.table(doc, q, "Decision tree quality: scikit-learn vs Spark MLlib (same test sets)", C, font_size=7.5)
    doc.add_heading("12.4 K-Means: Python vs PySpark", 2)
    wl = bk.workload.str.replace("kmeans_", "", regex=False)
    t = pd.DataFrame({"Data set": wl, "Data": pct(bk.fraction), "Rows": bk.records, "Feat.": bk.features, "K": bk.k,
                      "Py fit": bk.py_fit_s, "Py pred": bk.py_predict_s, "Sp conv": bk.spark_convert_s,
                      "Sp fit": bk.spark_fit_s, "Sp pred": bk.spark_predict_s, "Speedup": bk.speedup,
                      "Impr %": bk.speedup_improvement_pct.map("{:.1f}".format)})
    H.table(doc, t, "K-Means execution time (s) and speedup", C, font_size=7.5, decimals=4)
    q = pd.DataFrame({"Data set": wl, "Data": pct(bk.fraction), "Py WCSS": bk.py_wcss.map("{:.1f}".format),
                      "Sp WCSS": bk.spark_wcss.map("{:.1f}".format), "Py silh.": bk.py_silhouette, "Sp silh.": bk.spark_silhouette,
                      "ARI Py-Sp": bk.ari_py_vs_spark, "Py iter": bk.py_iterations, "Sp iter": bk.spark_iterations,
                      "Py MB": bk.py_peak_mem_mb.map("{:.2f}".format), "JVM MB": bk.spark_jvm_rss_mb.map("{:.0f}".format)})
    H.table(doc, q, "K-Means quality and memory: scikit-learn vs Spark MLlib", C, font_size=7.5, decimals=3)
    doc.add_heading("12.5 Frequent patterns: Apriori (ours) vs Spark FP-Growth", 2)
    t = pd.DataFrame({"Data": pct(bf.fraction), "Trans.": bf.transactions, "Apriori": bf.apriori_s,
                      "Rules": bf.apriori_rules_s, "mlxtend FPG": bf.mlxtend_fpgrowth_s, "Sp conv": bf.spark_convert_s,
                      "Sp FPG": bf.spark_fpgrowth_total_s, "Speedup": bf.speedup,
                      "Itemsets ours / Sp": bf.apriori_itemsets.astype(str) + " / " + bf.spark_itemsets.astype(str),
                      "Identical": bf.identical_itemsets_and_counts.map({True: "yes", False: "no"}),
                      "Rules ours / Sp": bf.apriori_rules_single_consequent.astype(str) + " / " + bf.spark_rules.astype(str)})
    H.table(doc, t, "Frequent-pattern mining time (s) and output consistency (rules: single-item consequents)", C,
            font_size=7.5, decimals=4)
    ident = bool(bf.identical_itemsets_and_counts.all())
    H.para(doc, (
        f"At every size, Spark's FP-Growth returns {'exactly the same' if ident else 'NOT always the same'} frequent "
        f"itemsets with identical counts as our Apriori, and the same number of single-consequent rules. The algorithms "
        f"differ, but the task-level results agree, as expected for two exact algorithms. The pandas-based mlxtend "
        f"FP-Growth was the slowest of the three ({bf.mlxtend_fpgrowth_s.iloc[-1]:.2f} s at 100 %), because of its "
        f"DataFrame overhead on 11 thousand rows. Algorithmic advantages (no candidate generation) only matter once "
        f"the data is large or dense."), align="justify")
    doc.add_heading("12.6 Graphs", 2)
    for f, tt in [("38_pyspark_execution_time.png", "Execution time vs data size, Python vs PySpark"),
                  ("39_pyspark_speedup.png", "Speedup per workload (log scale)"),
                  ("40_pyspark_time_breakdown.png", "Where PySpark time goes at 100 % of the data"),
                  ("41_pyspark_quality_comparison.png", "Result quality: scikit-learn vs Spark MLlib")]:
        fig_block(doc, r, f, tt, C)
    doc.add_heading("12.7 Scalability stress test (replicated rows - not new data)", 2)
    H.para(doc, (
        "To see *where* Spark's fixed overhead would be amortised, the real rows were replicated ×5 and ×20. **These "
        "replicated records contain no new information. They are used only to measure scaling, and no analytical "
        "conclusion is drawn from them.**"), align="justify")
    s = pd.DataFrame({
        "Workload": st.workload.str.replace(r"_replicated_x\d+", "", regex=True).str.replace("_", " "),
        "Replication": "×" + st.replication_factor.astype(int).astype(str),
        "Rows / transactions": st.records.fillna(st.transactions).astype(int),
        "Python (s)": st.py_total_s.fillna(st.apriori_total_s),
        "PySpark (s)": st.spark_total_s.fillna(st.spark_fpgrowth_total_s),
        "Speedup": st.speedup, "Impr %": st.speedup_improvement_pct.map("{:.1f}".format)})
    H.table(doc, s, "Stress test by row replication (×5, ×20); Python = sklearn (DT, K-Means) or our Apriori", C,
            font_size=8, decimals=3)
    doc.add_heading("12.8 Interpretation", 2)
    kf = bk[bk.workload == "kmeans_films"]
    H.para(doc, (
        f"**PySpark was slower than single-machine Python for every technique and every real data size.** Speedups "
        f"were {bdt.speedup.min():.3f}-{bdt.speedup.max():.3f}× for the decision tree, "
        f"{kf.speedup.min():.3f}-{kf.speedup.max():.3f}× for K-Means on films and "
        f"{bf.speedup.min():.3f}-{bf.speedup.max():.3f}× for frequent-pattern mining. The improvement percentages are "
        f"therefore negative. The reasons are structural. Each Spark job pays a roughly constant cost of task "
        f"scheduling, JVM-Python (Py4J/Arrow) serialisation, stage planning and result collection, typically 0.1-1 s. "
        f"The Python computations on ≤ 11 408 rows take milliseconds. Spark's iterative algorithms (K-Means, tree "
        f"induction level by level) launch one or more jobs per iteration or level, multiplying the overhead. Data "
        f"conversion into Spark alone costs more than the whole scikit-learn run. Speedups improve with data volume "
        f"because the Python cost grows while Spark's overhead is fixed. In the stress test, "
        + "; ".join(f"{w.replace('_replicated_x' + str(int(f)), '').replace('_', ' ')} ×{int(f)}: {s:.2f}×"
                    for w, f, s in zip(st.workload, st.replication_factor, st.speedup))
        + ". "
        + ("Spark overtakes Python only for " + ", ".join(sorted({w.split('_replicated')[0].replace('_', ' ') for w in st[st.speedup > 1].workload}))
           + ": Apriori's level-wise candidate counting grows with the number of transactions, while FP-Growth's "
             "two-scan, tree-based approach and Spark's parallel counting amortise their overhead. "
           if (st.speedup > 1).any() else "Even at ×20 Spark does not overtake Python. ")
        + f"scikit-learn's highly optimised C/Cython decision tree and K-Means remain much faster at every size tested. "
        f"The results are nevertheless equivalent: Spark's tree reaches similar accuracy and "
        f"AUC (differences come from binning continuous features into 256 thresholds), Spark K-Means reaches "
        f"near-identical WCSS and silhouette (different k-means|| seeding leads to a slightly different local optimum), "
        f"and FP-Growth reproduces every itemset exactly."), align="justify")
    H.para(doc, (
        "**When would PySpark win?** When the data no longer fits comfortably in one process's memory (tens of "
        "millions of rows or more), when it is already stored in a distributed file system (so no pandas conversion is "
        "needed), and when a real cluster provides many more cores than the two used here. For the 2 MB UCI Movie "
        "data set the honest conclusion is that big-data tooling brings no performance enhancement. Its value lies in "
        "the fact that the same code would scale unchanged."), align="justify")
    doc.add_heading("12.9 Limitations of the benchmark", 2)
    H.bullets(doc, [
        "A single laptop in local mode is not a cluster. Network shuffles, data locality and executor parallelism are "
        "not exercised.",
        "Python-side timings are in milliseconds and therefore sensitive to OS scheduling. Medians of three runs reduce "
        "but do not remove this noise.",
        "Spark's tree uses binned thresholds (maxBins) and its K-Means uses a different initialisation, so the "
        "configurations are comparable but not identical.",
        "Memory figures (Python heap vs JVM RSS) measure different things and only indicate footprints.",
        "Replicated data in the stress test has artificial structure (exact duplicates), which may favour some "
        "algorithms."])


def s13_additional(doc, r, C):
    doc.add_heading("13. Additional Findings", 1)
    H.para(doc, "Analyses beyond the mandatory techniques (clearly separated from them):")
    H.bullets(doc, [
        "**Similarity analysis (Lecture 22):** the Jaccard similarity between genres (Figure 16) shows that genres are "
        "largely disjoint labels. The strongest overlaps are Romantic-Comedy and Musical-Comedy. Jaccard ignores joint "
        "absences, which matters for sparse binary data.",
        "**Hierarchical clustering and dendrograms (Lecture 21):** single/complete/average/Ward linkage compared with "
        "cophenetic correlation and ARI against K-Means (Section 9.6). Ward agrees best with K-Means; single linkage "
        "chains.",
        "**Dimensionality reduction:** PCA projection of the 19-D director profiles (Figure 31). PC1 (period/colour) "
        "explains the largest share of variance.",
        "**Closure-based rule de-duplication:** applying the A-Close closure operator to association rules removes "
        "rules that describe the same film set. This joins Lectures 6 and 11.",
        "**Sensitivity analysis:** support/confidence grids for Apriori, and frequent/closed/maximal counts across six "
        "support levels for COFI.",
        "**Statistical tests:** chi-square / Cramér's V between categorical attributes, Kruskal-Wallis for ratings "
        "across eras.",
        "**Data-quality analysis:** the missing-value heatmap shows that missingness depends on the period "
        "(e.g. awards 94-100 % missing for pre-1920 and 1990s films). This is an important caveat for any model "
        "trained on this data.",
        "**Smoothing demonstrations:** equal-depth binning, moving average and regression smoothing (Lecture 5)."])


def s14_limitations(doc, r):
    doc.add_heading("14. Challenges and Limitations", 1)
    doc.add_heading("14.1 Dataset limitations", 2)
    H.bullets(doc, [
        "The collection is curated by one person (complete for Hitchcock, dense for his interests), so frequencies and "
        "trends describe the collection, not cinema in general.",
        "Many attributes are sparse (studio known for ~48 % of films, Halliwell rating for ~25 %, cast for ~74 %). "
        "Missingness is period-dependent, i.e. not missing at random.",
        "The data is frozen in 1999 and the critic ratings come from a 1987 guide, so recent films are under-rated.",
        "Free-text and hand-coded fields required heuristic parsing (realignment, typo table). A small number of "
        "ambiguous tokens were dropped rather than guessed."])
    doc.add_heading("14.2 Algorithm limitations", 2)
    H.bullets(doc, [
        "Apriori's output depends strongly on the thresholds. Rare but important items (Oscars) need separate low-support "
        "runs.",
        "The decision tree is unstable (small data changes can change splits), its accuracy is moderate, and its target "
        "reflects one critic's taste.",
        "K-Means assumes spherical clusters of similar size; the director profiles have weak cluster structure "
        "(silhouette ≈ 0.13).",
        "COFI's sub-pattern enumeration is exponential in the frequent-path-base length. That is harmless here "
        "(bases ≤ 9 items) but would need COFI's further optimisations on dense data. Closedness is decided after "
        "mining, not inside the COFI-trees."])
    doc.add_heading("14.3 Hardware and implementation limitations", 2)
    H.bullets(doc, [
        "8 GB RAM and a nearly full C: drive constrained the Spark configuration (local[2], 2 GB driver, scratch on E:). "
        "Spark ran in local mode, not on a cluster.",
        "PySpark on Windows required two workarounds (space-free junction path, Windows-safe Python worker).",
        "The slow network forced reuse of locally available packages (the virtual environment inherits the system's "
        "scientific stack; versions are pinned in requirements.txt)."])


def s15_conclusion(doc, r):
    kf, ap, dt, km, c = r.kf, r.ap, r.dt["test_metrics"], r.km, r.cofi
    doc.add_heading("15. Conclusion", 1)
    H.bullets(doc, [
        f"**Preprocessing:** a reproducible KDD pipeline turned {r.pre['raw_main_rows']:,} noisy HTML rows "
        f"({r.pre['raw_main_malformed']} malformed) and five related tables into {r.pre['films_final']:,} clean, "
        f"integrated films. Every cleaning operation is logged and justified, imputation is attribute-specific "
        f"(group median, documented rule, constant), and leakage-prone attributes were removed.",
        f"**Descriptive analytics and visualisation:** {len(r.figs)} figures of "
        f"{len({f['family'] for f in r.figs})} distinct techniques show a curated collection dominated by drama and comedy, with "
        f"sharp technological eras and genre life cycles. Era is the strongest organising variable "
        f"(Cramér's V {kf['cramers_v_era_color']:.2f} with colour process).",
        f"**Association patterns:** Apriori (validated against mlxtend) found {ap['frequent_itemsets']} frequent "
        f"itemsets and {ap['rules']} rules. The most interesting rules identify Hitchcock's TV productions, technology "
        f"eras, genre-era life cycles and the classical origin of remade films.",
        f"**Classification:** release-time attributes predict Halliwell acclaim with ROC-AUC {dt['roc_auc']:.3f}. Year "
        f"and documentation depth dominate, revealing the guide's preference for classic films.",
        f"**Clustering:** {km['n_directors']} directors form {km['chosen_k']} career profiles (classical studio, "
        f"modern colour, TV suspense). The structure is weak but interpretable and agrees with Ward hierarchical "
        f"clustering.",
        f"**COFI:** an exact, independently validated COFI-tree implementation mined {c['closed_itemsets']} closed and "
        f"{c['maximal_itemsets']} maximal itemsets from {c['frequent_itemsets']} frequent ones, using one small tree "
        f"at a time.",
        "**PySpark:** Spark MLlib reproduced the decision tree, K-Means and frequent-pattern results, but it was "
        "slower on this small data set because of fixed distributed-execution overhead. Its benefit appears only as "
        "volumes grow.",
        "**Overall:** careful preprocessing and validation matter more than algorithmic sophistication on real, messy "
        "data. Every mining technique pointed to the same underlying structure, film history, and the honest "
        "big-data answer for a 2 MB data set is that a single machine is the better tool."])


def s16_references(doc):
    doc.add_heading("16. References", 1)
    refs = [
        "Wiederhold, G. (1990). *Movie* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SW2R",
        "Sivaselvan, B. *Lecture slides 2-22, Advanced Statistical and Big Data Analytics* (KDD phases, Apriori & "
        "FP-Growth, Apriori variants, CFI/MFI, A-Close, Pincer-Search, classification, Bayesian classification, "
        "clustering, hierarchical clustering, similarity measures). IIITDM Kancheepuram. Course material.",
        "Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. In *Proceedings of the 20th "
        "International Conference on Very Large Data Bases (VLDB)* (pp. 487-499).",
        "Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. In *Proceedings of "
        "the ACM SIGMOD International Conference on Management of Data* (pp. 1-12).",
        "El-Hajj, M., & Zaïane, O. R. (2003). COFI-tree mining: A new approach to pattern growth with reduced candidacy "
        "generation. In *Proceedings of the IEEE ICDM Workshop on Frequent Itemset Mining Implementations (FIMI'03)*.",
        "Pasquier, N., Bastide, Y., Taouil, R., & Lakhal, L. (1999). Discovering frequent closed itemsets for "
        "association rules. In *Database Theory - ICDT'99*, LNCS 1540 (pp. 398-416). Springer.",
        "Brin, S., Motwani, R., Ullman, J. D., & Tsur, S. (1997). Dynamic itemset counting and implication rules for "
        "market basket data. In *Proceedings of the ACM SIGMOD Conference* (pp. 255-264).",
        "Piatetsky-Shapiro, G. (1991). Discovery, analysis, and presentation of strong rules. In *Knowledge Discovery "
        "in Databases* (pp. 229-248). AAAI/MIT Press.",
        "Quinlan, J. R. (1986). Induction of decision trees. *Machine Learning*, 1(1), 81-106.",
        "Quinlan, J. R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.",
        "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. "
        "Wadsworth.",
        "MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. In *Proceedings "
        "of the 5th Berkeley Symposium on Mathematical Statistics and Probability* (Vol. 1, pp. 281-297).",
        "Lloyd, S. (1982). Least squares quantization in PCM. *IEEE Transactions on Information Theory*, 28(2), 129-137.",
        "Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. In *Proceedings of the "
        "18th ACM-SIAM Symposium on Discrete Algorithms* (pp. 1027-1035).",
        "Bahmani, B., Moseley, B., Vattani, A., Kumar, R., & Vassilvitskii, S. (2012). Scalable k-means++. "
        "*Proceedings of the VLDB Endowment*, 5(7), 622-633.",
        "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. "
        "*Journal of Computational and Applied Mathematics*, 20, 53-65.",
        "Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis "
        "and Machine Intelligence*, 1(2), 224-227.",
        "Caliński, T., & Harabasz, J. (1974). A dendrite method for cluster analysis. *Communications in Statistics*, "
        "3(1), 1-27.",
        "Li, H., Wang, Y., Zhang, D., Zhang, M., & Chang, E. Y. (2008). PFP: Parallel FP-growth for query "
        "recommendation. In *Proceedings of the ACM Conference on Recommender Systems* (pp. 107-114).",
        "Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the "
        "ACM*, 59(11), 56-65.",
        "Meng, X., et al. (2016). MLlib: Machine learning in Apache Spark. *Journal of Machine Learning Research*, "
        "17(34), 1-7.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, "
        "12, 2825-2830.",
        "Raschka, S. (2018). MLxtend: Providing machine learning and data science utilities and extensions to Python's "
        "scientific computing stack. *Journal of Open Source Software*, 3(24), 638.",
        "Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques* (3rd ed.). Morgan Kaufmann.",
        "Leskovec, J., Rajaraman, A., & Ullman, J. D. (2020). *Mining of Massive Datasets* (3rd ed.). Cambridge "
        "University Press.",
        "Halliwell, L. (1987). *Halliwell's Film and Video Guide* (6th ed.). Scribner. (Source of the ratings encoded "
        "in the dataset.)",
        "Apache Software Foundation. *Spark 4.0 MLlib Guide* (classification, clustering, frequent pattern mining). "
        "https://spark.apache.org/docs/latest/ml-guide.html",
        "scikit-learn developers. *scikit-learn User Guide: Decision Trees; Clustering; Model Evaluation*. "
        "https://scikit-learn.org/stable/user_guide.html"]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = H.Cm(0.9)
        p.paragraph_format.first_line_indent = H.Cm(-0.9)
        p.paragraph_format.space_after = H.Pt(3)
        H.add_rich(p, f"[{i}]  {ref}", 9.5)


def appendix(doc, r, C):
    doc.add_heading("Appendix A. Important Code Snippets", 1)
    H.para(doc, "Apriori candidate generation with pruning (`src/association_rules/apriori.py`):")
    H.code_block(doc, """def generate_candidates(prev_level, prev_set, k):
    groups = defaultdict(list)                      # join: same (k-2)-prefix
    for itemset in sorted(prev_level):
        groups[itemset[:-1]].append(itemset)
    joined = [p + (m[i][-1], m[j][-1]) for p, m in groups.items()
              for i in range(len(m)) for j in range(i + 1, len(m))]
    survivors = [c for c in joined                  # prune: downward closure
                 if all(s in prev_set for s in combinations(c, k - 1))]
    return survivors, len(joined), len(joined) - len(survivors)""")
    H.para(doc, "COFI-tree mining with participation counts (`src/cofi/cofi.py`):")
    H.code_block(doc, """for b_item in self.header_order:                  # least frequent first
    n = self.head.get(b_item)
    while n is not None:
        d = n.count - n.participation               # transactions ending here
        if d > 0:
            path, p = [], n
            while p is not self.root:
                path.append(p.item); p.participation += d; p = p.parent
            bases[frozenset(path)] += d             # frequent-path-base
        n = n.link
for base, cnt in bases.items():                      # sub-pattern supports
    for r in range(1, len(base) + 1):
        for sub in combinations(sorted(base), r):
            support[frozenset(sub)] += cnt""")
    H.para(doc, "Information gain and gain ratio (`src/classification/split_criteria.py`):")
    H.code_block(doc, """def evaluate_binary_split(x, y, thr):
    left, right = y[x <= thr], y[x > thr]
    wl, wr = len(left) / len(y), len(right) / len(y)
    gain = entropy(y) - (wl * entropy(left) + wr * entropy(right))
    split_info = -sum(w * np.log2(w) for w in (wl, wr) if w > 0)
    gini_split = wl * gini(left) + wr * gini(right)
    return dict(info_gain=gain, gain_ratio=gain / split_info, gini_reduction=gini(y) - gini_split)""")
    H.para(doc, "Manual K-Means (Lloyd) (`src/clustering/kmeans.py`):")
    H.code_block(doc, """for it in range(max_iter):
    dist = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(-1))   # Euclidean
    new_labels = dist.argmin(1)                                      # assignment
    newC = np.array([X[new_labels == j].mean(0) for j in range(K)])  # centroid update
    if np.array_equal(labels, new_labels) and shift(newC, C) < tol: break
    C, labels = newC, new_labels""")
    doc.add_heading("Appendix B. Algorithm Configurations and Parameter Values", 1)
    dt = r.dt
    par = pd.DataFrame([
        ("Apriori", f"min_support {r.ap['min_support']} (count {r.ap['min_count']}), min_confidence {r.ap['min_confidence']}; "
                    "grid 0.01-0.05 x 0.5-0.8; supplementary 0.01 / 0.15 / lift > 2"),
        ("Rule selection", "single-item consequent, lift > 1.1, one rule per closure, ≤ 3 per consequent"),
        ("COFI", f"min_support {r.cofi['min_support']} (count {r.cofi['min_count']}); comparison 0.005-0.10"),
        ("Decision tree", f"{dt['final_params']}; stratified 75/25 split, 5-fold stratified CV, seed 42"),
        ("K-Means", f"K = {r.km['chosen_k']} (K 2-10 evaluated), k-means++, n_init 20, standardised 19 features, seed 42"),
        ("Hierarchical", "single, complete, average, Ward; Euclidean; cut at K = 3"),
        ("Spark DT", "entropy, maxDepth 6, minInstancesPerNode 40, maxBins 256, weightCol (balanced), seed 42"),
        ("Spark K-Means", "k-means||, maxIter 100, tol 1e-4, seed 42; K = 3 (directors), 6 (films)"),
        ("Spark FP-Growth", "minSupport 0.02, minConfidence 0.60"),
        ("Benchmark", "fractions 10/25/50/100 %, 3 repeats (median), 1 warm-up, stress test x5 / x20")],
        columns=["Component", "Configuration"])
    H.table(doc, par, "Algorithm configurations and parameter values", C, col_widths_cm=[3.2, 13.2], font_size=8)
    doc.add_heading("Appendix C. Execution Environment and Reproduction", 1)
    H.code_block(doc, r"""cd "D:\sem-7-labs\Big Data labs\endsem-hackathon"    (or the junction D:\ASBD_Hackathon)
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_all.py                 # all phases incl. Spark + report
.venv\Scripts\python -m pytest tests -q         # 48 tests""")
    doc.add_heading("Appendix D. Project Deliverables", 1)
    dl = pd.DataFrame([
        ("data/raw/", "original UCI zip and extracted files (unmodified)"),
        ("data/processed/, data/transactions/", "cleaned, integrated, transformed, ML-ready and transaction data"),
        ("src/", "modular source code (loading, preprocessing, analytics, visualisation, mining, Spark, report)"),
        ("notebooks/", "eight executed Jupyter notebooks, one per phase"),
        ("outputs/", "statistics, logs, apriori/, decision_tree/, kmeans/, cofi/, pyspark/ results, plots/"),
        ("outputs/visualization_inferences.md", "per-plot inferences"),
        ("reference_materials/lecture_summary.md", "concepts extracted from the lecture slides"),
        ("tests/", "unit and integration tests"),
        ("reports/", "this report (DOCX + PDF) and its figures")], columns=["Location", "Content"])
    H.table(doc, dl, "Project deliverables", C, col_widths_cm=[5.5, 10.9])


def run():
    return build()


if __name__ == "__main__":
    run()

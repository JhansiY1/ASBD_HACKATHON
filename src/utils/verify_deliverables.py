"""
Final verification of all deliverables (PHASE 15).

Checks that every required file exists and is non-empty, that the DOCX opens
with python-docx and the PDF with PyMuPDF, counts figures/tables/pages and
checks that every generated plot is embedded in the report.
Writes outputs/logs/final_verification.txt.

Run:  python -m src.utils.verify_deliverables
"""
from __future__ import annotations

import json

from src.config import OUTPUTS, PROJECT_ROOT, REPORTS_DIR, get_logger

log = get_logger("verify")

REQUIRED = [
    "data/raw/movie.zip", "data/raw/movie/data/main.html", "data/processed/films_cleaned_integrated.csv",
    "data/processed/films_transformed.csv", "data/processed/dt_ml_matrix.csv",
    "data/processed/director_profiles.csv", "data/processed/kmeans_director_matrix_z.csv",
    "data/transactions/film_transactions.csv", "data/transactions/film_transactions_binary.csv",
    "reference_materials/lecture_summary.md",
    "outputs/dataset_overview.txt", "outputs/data_dictionary.csv", "outputs/data_quality_summary.csv",
    "outputs/cleaning_log.csv", "outputs/preprocessing_report.md", "outputs/visualization_inferences.md",
    "outputs/apriori/frequent_itemsets.csv", "outputs/apriori/association_rules.csv", "outputs/apriori/top_rules.csv",
    "outputs/apriori/support_analysis.csv", "outputs/decision_tree/confusion_matrix.csv",
    "outputs/decision_tree/if_then_rules.csv", "outputs/kmeans/cluster_profiles.csv",
    "outputs/cofi/closed_frequent_itemsets.csv", "outputs/cofi/cofi_execution_summary.txt",
    "outputs/cofi/support_comparison.csv", "outputs/pyspark/benchmark_decision_tree.csv",
    "outputs/pyspark/benchmark_kmeans.csv", "outputs/pyspark/benchmark_frequent_patterns.csv",
    "outputs/pyspark/performance_comparison_summary.csv", "outputs/pyspark/spark_configuration.json",
    "outputs/pyspark/stress_test_replicated.csv", "outputs/logs/project_execution.log",
    "reports/ASBD_Movie_Analytics_Report.docx", "reports/ASBD_Movie_Analytics_Report.pdf",
    "README.md", "requirements.txt", "environment.yml", "CLAUDE.md",
    "tests/test_preprocessing.py", "tests/test_apriori.py", "tests/test_cofi.py", "tests/test_algorithms.py",
] + [f"notebooks/{n}" for n in ("01_dataset_exploration.ipynb", "02_preprocessing.ipynb",
                                 "03_descriptive_analytics.ipynb", "04_apriori.ipynb", "05_decision_tree.ipynb",
                                 "06_kmeans.ipynb", "07_cofi.ipynb", "08_pyspark_benchmarking.ipynb")]


def run() -> dict:
    lines, ok = [], True
    for rel in REQUIRED:
        p = PROJECT_ROOT / rel
        good = p.exists() and p.stat().st_size > 0
        ok &= good
        lines.append(f"[{'OK' if good else 'MISSING'}] {rel} ({p.stat().st_size if p.exists() else 0} bytes)")
    # notebooks executed?
    import nbformat
    for nb in sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb")):
        n = nbformat.read(nb, as_version=4)
        executed = sum(1 for c in n.cells if c.cell_type == "code" and c.get("outputs"))
        errors = sum(1 for c in n.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error")
        lines.append(f"[{'OK' if errors == 0 and executed else 'CHECK'}] {nb.name}: {executed} code cells with output, {errors} errors")
        ok &= errors == 0
    # DOCX
    from docx import Document
    d = Document(REPORTS_DIR / "ASBD_Movie_Analytics_Report.docx")
    n_img = len(d.inline_shapes)
    n_tab = len(d.tables)
    n_eq = len(d.element.body.findall(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"))
    headings = [p.text for p in d.paragraphs if p.style.name == "Heading 1"]
    lines.append(f"DOCX opens: {len(d.paragraphs)} paragraphs, {n_img} images, {n_tab} tables, {n_eq} native equations, "
                 f"{len(headings)} chapters")
    # PDF
    import fitz
    pdf = fitz.open(REPORTS_DIR / "ASBD_Movie_Analytics_Report.pdf")
    text = "".join(pg.get_text() for pg in pdf)
    lines.append(f"PDF opens: {pdf.page_count} pages, {len(pdf.get_toc())} bookmarks, {len(text):,} characters")
    # every plot embedded?
    cat = json.loads((OUTPUTS / "figure_catalogue.json").read_text())
    plots = sorted((OUTPUTS / "plots").glob("*.png"))
    eda_fam = {c["family"] for c in cat if c["number"] <= 16}
    lines.append(f"Plots generated: {len(plots)}; catalogue entries: {len(cat)}; distinct techniques: "
                 f"{len({c['family'] for c in cat})} overall, {len(eda_fam)} among EDA plots (>= 10 required: "
                 f"{'OK' if len(eda_fam) >= 10 else 'FAIL'}); images in report: {n_img}")
    ok &= len(eda_fam) >= 10
    ok &= n_img >= len(plots)
    lines.append(f"OVERALL: {'PASS' if ok else 'CHECK ITEMS ABOVE'}")
    out = "\n".join(lines)
    (OUTPUTS / "logs" / "final_verification.txt").write_text(out, encoding="utf-8")
    print(out)
    log.info("PHASE 15 verification: %s", "PASS" if ok else "CHECK")
    return dict(ok=ok)


if __name__ == "__main__":
    run()

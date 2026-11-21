"""
Run the complete ASBD Movie Analytics pipeline end to end.

    python run_all.py              # every phase
    python run_all.py --skip-spark # everything except the (slow) PySpark benchmark
    python run_all.py --only report

Phases (KDD order):
  preprocess   -> src.preprocessing.pipeline          (cleaning, integration, selection, transformation)
  describe     -> src.analytics.descriptive           (descriptive statistics)
  plots        -> src.visualization.eda_plots         (16 exploratory visualisations)
  apriori      -> src.association_rules.run_apriori   (Apriori from scratch + rules)
  tree         -> src.classification.decision_tree    (decision tree)
  kmeans       -> src.clustering.kmeans               (K-Means + hierarchical)
  cofi         -> src.cofi.run_cofi                   (COFI closed itemsets + validation)
  spark        -> src.pyspark_impl.spark_benchmark    (PySpark implementations + benchmark)
  inferences   -> src.visualization.write_inferences  (outputs/visualization_inferences.md)
  report       -> src.report.build_report             (DOCX + PDF report)

Spark's Windows launch scripts break on paths containing spaces.  When the
project lives in such a path, the Spark phase is executed through the
space-free junction D:\\ASBD_Hackathon (created with
`mklink /J D:\\ASBD_Hackathon "<project path>"`) if it exists.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PHASES = [("preprocess", "src.preprocessing.pipeline"), ("describe", "src.analytics.descriptive"),
          ("plots", "src.visualization.eda_plots"), ("apriori", "src.association_rules.run_apriori"),
          ("tree", "src.classification.decision_tree"), ("kmeans", "src.clustering.kmeans"),
          ("cofi", "src.cofi.run_cofi"), ("spark", "src.pyspark_impl.spark_benchmark"),
          ("inferences", "src.visualization.write_inferences"), ("report", "src.report.build_report")]
JUNCTION = Path(r"D:\ASBD_Hackathon")


def run_spark_phase() -> None:
    if " " in str(ROOT) and (JUNCTION / ".venv" / "Scripts" / "python.exe").exists():
        py = JUNCTION / ".venv" / "Scripts" / "python.exe"
        print(f"[spark] project path contains spaces -> running through {JUNCTION}")
        subprocess.run([str(py), "-m", "src.pyspark_impl.spark_benchmark"], cwd=JUNCTION, check=True)
    else:
        importlib.import_module("src.pyspark_impl.spark_benchmark").run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-spark", action="store_true")
    ap.add_argument("--only", nargs="*", help="run only these phases")
    args = ap.parse_args()
    from src.config import get_logger
    log = get_logger("run_all")
    for name, module in PHASES:
        if args.only and name not in args.only:
            continue
        if name == "spark" and args.skip_spark:
            log.info("skipping spark phase (--skip-spark)")
            continue
        t0 = time.perf_counter()
        log.info("=== phase %s started", name)
        if name == "spark":
            run_spark_phase()
        else:
            importlib.import_module(module).run()
        log.info("=== phase %s finished in %.1fs", name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()

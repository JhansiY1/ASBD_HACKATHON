"""
Central configuration for the ASBD Movie Analytics project.

All paths are resolved relative to the project root so that the pipeline is
reproducible from any working directory.  Random seeds and algorithm
parameters that are shared by several modules are defined here once.
"""
from __future__ import annotations

import logging
import os

# joblib/loky tries to count physical cores with the deprecated 'wmic'
# command (absent on recent Windows); fixing the count avoids the warning.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "movie" / "data"
RAW_DOC_DIR = PROJECT_ROOT / "data" / "raw" / "movie"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRANSACTIONS_DIR = PROJECT_ROOT / "data" / "transactions"
OUTPUTS = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS / "plots"
APRIORI_DIR = OUTPUTS / "apriori"
DT_DIR = OUTPUTS / "decision_tree"
KMEANS_DIR = OUTPUTS / "kmeans"
COFI_DIR = OUTPUTS / "cofi"
SPARK_DIR = OUTPUTS / "pyspark"
LOG_DIR = OUTPUTS / "logs"
DESCRIPTIVE_DIR = OUTPUTS / "descriptive"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (PROCESSED_DIR, TRANSACTIONS_DIR, PLOTS_DIR, APRIORI_DIR, DT_DIR,
           KMEANS_DIR, COFI_DIR, SPARK_DIR, LOG_DIR, DESCRIPTIVE_DIR,
           REPORTS_DIR / "figures", REPORTS_DIR / "report_assets"):
    _d.mkdir(parents=True, exist_ok=True)

# Spark scratch space is kept on E: (large free drive) when available, so the
# nearly full C: drive is never used for shuffle / temp files.
SPARK_LOCAL_DIR = Path("E:/spark_tmp") if Path("E:/").exists() else PROJECT_ROOT / "spark_tmp"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Selected algorithm parameters (justified in the report / sensitivity runs)
# --------------------------------------------------------------------------
APRIORI_MIN_SUPPORT = 0.02      # 2 % of the transactions
APRIORI_MIN_CONFIDENCE = 0.60   # 60 %
COFI_MIN_SUPPORT = 0.02         # identical threshold so results are comparable
KMEANS_K_RANGE = range(2, 11)


def get_logger(name: str = "asbd") -> logging.Logger:
    """Return a logger that writes both to the console and to the project log.

    The project execution log (outputs/logs/project_execution.log) is the
    single audit trail requested by the assignment.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "project_execution.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

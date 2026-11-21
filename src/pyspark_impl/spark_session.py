"""
Spark session factory tuned for the target laptop
(Intel i5-1235U, 8 GB RAM, Windows 11, C: almost full).

Configuration choices
---------------------
master              local[2]   two worker threads: leaves CPU/RAM for the OS and
                               the Python driver on an 8 GB machine
driver memory       2g         local mode runs executors inside the driver JVM
shuffle partitions  4          default 200 is far too many for ~10^4 rows
spark.local.dir     E:/spark_tmp  shuffle/spill files kept off the full C: drive
UI                  disabled   saves memory and a port
Arrow               enabled    faster pandas <-> Spark conversion
PYSPARK_PYTHON      the venv interpreter (driver and workers use the same Python)

The package is named ``pyspark_impl`` (not ``pyspark``) so that it can never
shadow the real pyspark library on the import path.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from src.config import SPARK_LOCAL_DIR, get_logger

log = get_logger("spark")

SPARK_CONF = {
    "spark.master": "local[2]",
    "spark.driver.memory": "2g",
    "spark.sql.shuffle.partitions": "4",
    "spark.default.parallelism": "2",
    "spark.ui.enabled": "false",
    "spark.sql.execution.arrow.pyspark.enabled": "true",
    "spark.local.dir": str(SPARK_LOCAL_DIR).replace("\\", "/"),
    "spark.driver.extraJavaOptions": f"-Djava.io.tmpdir={str(SPARK_LOCAL_DIR).replace(chr(92), '/')}",
    "spark.sql.adaptive.enabled": "true",
}


def build_spark(app_name: str = "ASBD-Movie-Analytics"):
    """Start the local SparkSession.

    Windows specifics (see README):
      * the project must be run from a path WITHOUT spaces (Spark's launch
        scripts split on them) - use the D:\\ASBD_Hackathon junction;
      * Python workers use the Windows-safe entry point in win_worker/.
    """
    if " " in sys.executable:
        log.warning("Python path contains spaces (%s); Spark launch scripts may fail. "
                    "Run from D:\\ASBD_Hackathon instead.", sys.executable)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    worker_dir = str(Path(__file__).resolve().parent / "win_worker")
    os.environ["PYTHONPATH"] = worker_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    SPARK_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    from pyspark.sql import SparkSession
    t0 = time.perf_counter()
    b = SparkSession.builder.appName(app_name)
    for k, v in SPARK_CONF.items():
        b = b.config(k, v)
    if os.name == "nt":
        b = b.config("spark.python.worker.module", "spark_win_worker").config("spark.python.worker.reuse", "true")
    spark = b.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    startup = time.perf_counter() - t0
    log.info("Spark %s session started in %.2fs (%s)", spark.version, startup, SPARK_CONF["spark.master"])
    return spark, startup


@contextmanager
def spark_session(app_name: str = "ASBD-Movie-Analytics"):
    spark, startup = build_spark(app_name)
    try:
        yield spark, startup
    finally:
        spark.stop()
        log.info("Spark session stopped")


def jvm_rss_mb() -> float:
    """Resident memory of the Spark JVM (child process of this Python driver)."""
    try:
        import psutil
        me = psutil.Process()
        return sum(c.memory_info().rss for c in me.children(recursive=True)
                   if "java" in c.name().lower()) / 1e6
    except Exception:
        return float("nan")


def python_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return float("nan")

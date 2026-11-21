"""
Windows-safe entry point for PySpark Python workers.

Problem on this machine (Windows 11, PySpark 4.0.1, Python 3.12, Java 21):
every Python task fails with "Python worker exited unexpectedly (crashed)".
The worker completes the task and writes its reply, but the process exits
before the JVM has read it; the socket is torn down and the JVM sees EOF.
(Diagnosis and workaround adapted from the author's earlier lab-12 project on
the same laptop.)

This module performs what `python -m pyspark.worker` does - connect back to
the JVM, authenticate, send the PID, run `pyspark.worker.main` - and then
waits on the socket instead of exiting immediately: a further task on the
same connection is processed (worker reuse); EOF or an idle timeout ends the
process.  Algorithms are unaffected; only process shutdown changes.

It is activated only on Windows through `spark.python.worker.module`
(see spark_session.py).  MLlib algorithms run inside the JVM and do not need
Python workers at all; this module only matters for Python-side RDD work.
"""
import os
import socket

from pyspark.util import local_connect_and_auth
from pyspark.worker import main, write_int

IDLE_TIMEOUT_S = 60

if __name__ == "__main__":
    conn_info = os.environ.get("PYTHON_WORKER_FACTORY_SOCK_PATH",
                               int(os.environ.get("PYTHON_WORKER_FACTORY_PORT", -1)))
    secret = os.environ.get("PYTHON_WORKER_FACTORY_SECRET")
    sock_file, sock = local_connect_and_auth(conn_info, secret)
    write_int(os.getpid(), sock_file)
    sock_file.flush()
    while True:
        try:
            main(sock_file, sock_file)
            sock_file.flush()
        except (ConnectionError, BrokenPipeError, OSError):
            break
        try:
            sock.settimeout(IDLE_TIMEOUT_S)
            nxt = sock.recv(1, socket.MSG_PEEK)
        except (socket.timeout, OSError):
            break
        finally:
            try:
                sock.settimeout(None)
            except OSError:
                pass
        if not nxt:
            break

from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timed():
    payload = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield payload
    finally:
        end = time.perf_counter()
        payload["elapsed_ms"] = (end - start) * 1000.0

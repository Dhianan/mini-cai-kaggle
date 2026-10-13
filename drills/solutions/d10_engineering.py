"""Drill 10 (45 min): general engineering, the non-ML coding round.

Three small problems in the style of a 45-minute screen. Talk through complexity
and failure modes out loud.
"""
import json
import math
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable


class LRUCache:
    """get/put in O(1). Capacity >= 1. get returns None when missing."""

    def __init__(self, capacity: int):
        self.cap = capacity
        self.d: OrderedDict = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.d:
                return None
            self.d.move_to_end(key)
            return self.d[key]

    def put(self, key, value) -> None:
        with self.lock:
            self.d[key] = value
            self.d.move_to_end(key)
            if len(self.d) > self.cap:
                self.d.popitem(last=False)


def first_bad_step(lines: Iterable[str], key: str = "loss", spike: float = 10.0) -> int | None:
    """Scan a metrics.jsonl stream. Return the step of the first row whose `key` is
    NaN/inf, or more than `spike` times the median of all earlier values of `key`.
    Skip malformed lines and rows without `key`. None if nothing is wrong."""
    seen: list[float] = []
    for line in lines:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        v = row.get(key)
        if v is None:
            continue
        v = float(v)
        if not math.isfinite(v):
            return row.get("step")
        if seen:
            med = sorted(seen)[len(seen) // 2]
            if med > 0 and v > spike * med:
                return row.get("step")
        seen.append(v)
    return None


def parallel_map(fn: Callable[[Any], Any], items: list, workers: int = 4, retries: int = 2) -> list:
    """Apply fn to every item with a thread pool, preserving order. Retry each item up
    to `retries` extra times on exception; if it still fails, put the exception
    object in that slot instead of raising."""
    def run(x):
        for attempt in range(retries + 1):
            try:
                return fn(x)
            except Exception as e:
                if attempt == retries:
                    return e
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run, items))

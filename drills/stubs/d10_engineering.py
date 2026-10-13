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
        raise NotImplementedError()

    def get(self, key):
        raise NotImplementedError()

    def put(self, key, value) -> None:
        raise NotImplementedError()

def first_bad_step(lines: Iterable[str], key: str='loss', spike: float=10.0) -> int | None:
    """Scan a metrics.jsonl stream. Return the step of the first row whose `key` is
NaN/inf, or more than `spike` times the median of all earlier values of `key`.
Skip malformed lines and rows without `key`. None if nothing is wrong."""
    raise NotImplementedError()

def parallel_map(fn: Callable[[Any], Any], items: list, workers: int=4, retries: int=2) -> list:
    """Apply fn to every item with a thread pool, preserving order. Retry each item up
to `retries` extra times on exception; if it still fails, put the exception
object in that slot instead of raising."""
    raise NotImplementedError()

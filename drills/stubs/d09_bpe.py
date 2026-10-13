"""Drill 09 (45 min): byte-pair encoding from scratch (character level is fine).

Interview follow-ups: why does tokenization matter for arithmetic and code? Why
byte-level instead of character-level? How do special tokens bypass merges?
"""
from collections import Counter

def train_bpe(text: str, n_merges: int) -> list[tuple[str, str]]:
    """Learn merges over whitespace-separated words. Each word starts as a list of
characters. Each round, merge the most frequent adjacent pair (ties: the pair
that sorts first). Stop early if no pair occurs at least twice."""
    raise NotImplementedError()

def _merge(sym: tuple, pair: tuple[str, str]) -> tuple:
    raise NotImplementedError()

def encode(text: str, merges: list[tuple[str, str]]) -> list[list[str]]:
    """Apply merges in learned order to each word. Returns a token list per word."""
    raise NotImplementedError()

def decode(tokens: list[list[str]]) -> str:
    raise NotImplementedError()

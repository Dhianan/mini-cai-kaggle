"""Drill 09 (45 min): byte-pair encoding from scratch (character level is fine).

Interview follow-ups: why does tokenization matter for arithmetic and code? Why
byte-level instead of character-level? How do special tokens bypass merges?
"""
from collections import Counter


def train_bpe(text: str, n_merges: int) -> list[tuple[str, str]]:
    """Learn merges over whitespace-separated words. Each word starts as a list of
    characters. Each round, merge the most frequent adjacent pair (ties: the pair
    that sorts first). Stop early if no pair occurs at least twice."""
    words = Counter(text.split())
    vocab = {tuple(w): c for w, c in words.items()}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best_count = max(pairs.values())
        if best_count < 2:
            break
        best = min(p for p, c in pairs.items() if c == best_count)
        merges.append(best)
        vocab = {_merge(sym, best): c for sym, c in vocab.items()}
    return merges


def _merge(sym: tuple, pair: tuple[str, str]) -> tuple:
    out, i = [], 0
    while i < len(sym):
        if i + 1 < len(sym) and (sym[i], sym[i + 1]) == pair:
            out.append(sym[i] + sym[i + 1])
            i += 2
        else:
            out.append(sym[i])
            i += 1
    return tuple(out)


def encode(text: str, merges: list[tuple[str, str]]) -> list[list[str]]:
    """Apply merges in learned order to each word. Returns a token list per word."""
    out = []
    for w in text.split():
        sym = tuple(w)
        for m in merges:
            sym = _merge(sym, m)
        out.append(list(sym))
    return out


def decode(tokens: list[list[str]]) -> str:
    return " ".join("".join(t) for t in tokens)

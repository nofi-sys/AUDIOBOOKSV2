"""Helpers for coarse Spanish phonetic normalization and similarity."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from rapidfuzz.distance import Levenshtein


def strip_accents(text: str) -> str:
    """Remove diacritics while keeping base characters."""
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def phonetic_normalize(token: str) -> str:
    """
    Rough phonetic normalization for Spanish.
    Collapses common spelling variants: h silent, qu/c/z to k/s, v->b, ll/y->i, ch->x.
    """
    t = strip_accents(token.lower())
    t = re.sub(r"[^a-z0-9ñü]+", "", t)
    t = t.replace("ch", "x")
    t = re.sub(r"ll", "i", t)
    t = t.replace("y", "i")
    t = re.sub(r"qu", "k", t)
    t = re.sub(r"c(?=[ei])", "s", t)
    t = t.replace("c", "k")
    t = t.replace("z", "s")
    t = t.replace("v", "b")
    t = re.sub(r"g(?=[ei])", "h", t)
    t = t.replace("j", "h")
    t = t.replace("h", "")
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


def phonetic_sequence(tokens: Iterable[str]) -> str:
    """Collapse a list of tokens into a single phonetic string."""
    parts = [phonetic_normalize(tok) for tok in tokens if tok]
    return "".join(parts)


def phonetic_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    """Return a 0..1 similarity score based on phonetic edit distance."""
    seq_a = phonetic_sequence(tokens_a)
    seq_b = phonetic_sequence(tokens_b)
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    dist = Levenshtein.distance(seq_a, seq_b)
    max_len = max(len(seq_a), len(seq_b))
    return 1.0 - (dist / max_len)

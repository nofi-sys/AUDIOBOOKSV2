"""Pre-alignment glossary pass to flag likely ASR mistakes using phonetic similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from phonetic_utils import phonetic_similarity


DEFAULT_RATIO_THRESHOLD = 0.4  # max normalized edit distance; similarity >= 0.6
MAX_GLOSSARY_TOKENS = 4


def load_glossary(path: str | Path) -> List[dict]:
    data = Path(path).read_text(encoding="utf-8")
    return json.loads(data)


def _ensure_schema(conn) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(asr_tokens)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for col_def in (
        ("suspected_of", "TEXT"),
        ("suspicion_score", "REAL"),
        ("corrected_raw", "TEXT"),
    ):
        if col_def[0] not in existing_cols:
            cur.execute(f"ALTER TABLE asr_tokens ADD COLUMN {col_def[0]} {col_def[1]}")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS asr_suspicions (
            suspicion_id INTEGER PRIMARY KEY,
            asr_start_idx INTEGER NOT NULL,
            asr_end_idx INTEGER NOT NULL,
            candidate_ref TEXT NOT NULL,
            source TEXT NOT NULL,
            score REAL NOT NULL,
            status TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _lengths_to_try(base_len: int) -> Tuple[int, ...]:
    options = {base_len}
    if base_len > 1:
        options.add(base_len - 1)
    if base_len < MAX_GLOSSARY_TOKENS:
        options.add(base_len + 1)
    return tuple(sorted(o for o in options if 1 <= o <= MAX_GLOSSARY_TOKENS))


def _best_single_updates(
    candidates: Dict[int, Tuple[float, str]], idx: int, score: float, candidate_ref: str
) -> None:
    current = candidates.get(idx)
    if current is None or score > current[0]:
        candidates[idx] = (score, candidate_ref)


def mark_asr_suspicions(
    asr_tokens_norm: Sequence[str],
    glossary: Sequence[dict],
    conn,
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> List[dict]:
    """
    Populate asr_suspicions and tag asr_tokens with single-word suspects.
    """
    _ensure_schema(conn)
    min_similarity = 1.0 - ratio_threshold
    suspicions: Dict[Tuple[int, int, str], float] = {}
    best_single: Dict[int, Tuple[float, str]] = {}

    for entry in glossary:
        g_tokens_norm = [t for t in entry.get("tokens_norm", []) if t]
        if not g_tokens_norm or len(g_tokens_norm) > MAX_GLOSSARY_TOKENS:
            continue
        g_tokens_raw = entry.get("tokens_raw") or entry.get("tokens_norm") or []
        candidate_ref = " ".join(g_tokens_raw)
        base_len = len(g_tokens_norm)
        for span_len in _lengths_to_try(base_len):
            for start in range(0, len(asr_tokens_norm) - span_len + 1):
                span = asr_tokens_norm[start : start + span_len]
                sim = phonetic_similarity(g_tokens_norm, span)
                if sim < min_similarity:
                    continue
                key = (start, start + span_len, candidate_ref)
                if sim > suspicions.get(key, 0.0):
                    suspicions[key] = sim
                if span_len == 1 and base_len == 1:
                    _best_single_updates(best_single, start, sim, candidate_ref)

    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO asr_suspicions(asr_start_idx, asr_end_idx, candidate_ref, source, score, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (s, e, cand, "glossary_prepass", score, "raw")
            for (s, e, cand), score in suspicions.items()
        ],
    )
    for idx, (score, cand) in best_single.items():
        cur.execute(
            "UPDATE asr_tokens SET suspected_of = ?, suspicion_score = ? WHERE idx = ?",
            (cand, float(score), idx),
        )
    conn.commit()
    return [
        {
            "asr_start_idx": s,
            "asr_end_idx": e,
            "candidate_ref": cand,
            "score": score,
            "source": "glossary_prepass",
        }
        for (s, e, cand), score in suspicions.items()
    ]


def mark_asr_suspicions_from_db(
    db_path: str | Path,
    glossary_path: str | Path,
    *,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
) -> List[dict]:
    """Helper to run the scan directly on a SQLite DB produced by the aligner."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    asr_tokens = [row[0] for row in cur.execute("SELECT token_norm FROM asr_tokens ORDER BY idx")]
    glossary = load_glossary(glossary_path)
    try:
        return mark_asr_suspicions(asr_tokens, glossary, conn, ratio_threshold=ratio_threshold)
    finally:
        conn.close()

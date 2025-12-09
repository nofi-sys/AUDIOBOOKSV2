"""Post-alignment processing for ASR suspicions: confirmation, clusters, WER tweaks."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from phonetic_utils import phonetic_similarity
from text_utils import normalize
from alignment import WARN_WER


DEFAULT_CLUSTER_RATIO = 0.4
MAX_CLUSTER_OPS = 6
CONFIRM_WINDOW = 5
SIMPLE_CORRECTION_MIN_SCORE = 0.7


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return bool(row)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(asr_tokens)")
    cols = {row[1] for row in cur.fetchall()}
    for col_def in (
        ("suspected_of", "TEXT"),
        ("suspicion_score", "REAL"),
        ("corrected_raw", "TEXT"),
    ):
        if col_def[0] not in cols:
            cur.execute(f"ALTER TABLE asr_tokens ADD COLUMN {col_def[0]} {col_def[1]}")
    if not _table_exists(cur, "asr_suspicions"):
        cur.execute(
            """
            CREATE TABLE asr_suspicions (
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


def _load_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}


def _load_tokens(cur: sqlite3.Cursor, table: str) -> Tuple[List[str], List[str]]:
    cols = _load_columns(cur, table)
    norm_col = "token_norm" if "token_norm" in cols else ("norm" if "norm" in cols else "token")
    raw_col = "token_raw" if "token_raw" in cols else ("token" if "token" in cols else norm_col)
    norm = [row[0] for row in cur.execute(f"SELECT {norm_col} FROM {table} ORDER BY idx")]
    raw = [row[0] for row in cur.execute(f"SELECT {raw_col} FROM {table} ORDER BY idx")]
    return norm, raw


def _load_word_alignment(cur: sqlite3.Cursor) -> List[Tuple[int | None, int | None, str]]:
    ops: List[Tuple[int | None, int | None, str]] = []
    if _table_exists(cur, "word_alignment"):
        rows = cur.execute("SELECT ref_idx, asr_idx, op FROM word_alignment ORDER BY rowid").fetchall()
        for ref_idx, asr_idx, op in rows:
            ops.append((ref_idx, asr_idx, op))
        return ops
    if _table_exists(cur, "alignments_word"):
        rows = cur.execute(
            "SELECT ref_idx, asr_idx, tipo FROM alignments_word ORDER BY row_id, ref_idx, asr_idx"
        ).fetchall()
        for ref_idx, asr_idx, op in rows:
            ri = None if ref_idx is None or ref_idx == -1 else ref_idx
            ai = None if asr_idx is None or asr_idx == -1 else asr_idx
            ops.append((ri, ai, op))
    return ops


def _normalize_tokens(text: str) -> List[str]:
    return normalize(text, strip_punct=True).split()


def confirm_suspicions_with_alignment(conn: sqlite3.Connection, window: int = CONFIRM_WINDOW) -> Dict[str, int]:
    """Confirm glossary-based suspicions using word alignment to locate the ref window."""
    _ensure_schema(conn)
    cur = conn.cursor()
    ref_norm, _ = _load_tokens(cur, "ref_tokens")
    asr_norm, _ = _load_tokens(cur, "asr_tokens")
    word_ops = _load_word_alignment(cur)

    pending = cur.execute(
        "SELECT suspicion_id, asr_start_idx, asr_end_idx, candidate_ref, score "
        "FROM asr_suspicions WHERE source = 'glossary_prepass' AND status = 'raw'"
    ).fetchall()

    stats = {"confirmed": 0, "rejected": 0}
    updates: List[Tuple[str, float, int]] = []
    for sid, start, end, cand, score in pending:
        cand_norm = _normalize_tokens(cand)
        if not cand_norm:
            stats["rejected"] += 1
            updates.append(("rejected", score, sid))
            continue
        ref_hits = [ri for (ri, aj, _op) in word_ops if aj is not None and start <= aj < end and ri is not None]
        if ref_hits:
            ref_min = min(ref_hits)
            ref_max = max(ref_hits)
        else:
            approx = int(round((start / max(1, len(asr_norm))) * len(ref_norm))) if asr_norm else 0
            ref_min = ref_max = approx
        lo = max(0, ref_min - window)
        hi = min(len(ref_norm), ref_max + window)
        found = False
        for i in range(lo, max(lo, hi - len(cand_norm) + 1)):
            if ref_norm[i:i + len(cand_norm)] == cand_norm:
                found = True
                break
        if found:
            stats["confirmed"] += 1
            updates.append(("confirmed", min(1.0, score * 1.1), sid))
        else:
            stats["rejected"] += 1
            updates.append(("rejected", score, sid))

    if updates:
        cur.executemany("UPDATE asr_suspicions SET status = ?, score = ? WHERE suspicion_id = ?", updates)
        conn.commit()
    return stats


def _clusters(word_ops: Sequence[Tuple[int | None, int | None, str]]) -> List[List[Tuple[int | None, int | None, str]]]:
    clusters: List[List[Tuple[int | None, int | None, str]]] = []
    current: List[Tuple[int | None, int | None, str]] = []
    for op in word_ops:
        if op[2] == "match":
            if current:
                clusters.append(current)
                current = []
            continue
        current.append(op)
    if current:
        clusters.append(current)
    return clusters


def _pattern_ok(n_ref: int, n_asr: int) -> bool:
    if n_ref == 1 and 1 <= n_asr <= 3:
        return True
    if n_asr == 1 and 1 <= n_ref <= 3:
        return True
    if n_ref == n_asr == 1:
        return True
    return False


def detect_cluster_suspicions(
    conn: sqlite3.Connection,
    *,
    ratio_threshold: float = DEFAULT_CLUSTER_RATIO,
    max_ops: int = MAX_CLUSTER_OPS,
) -> int:
    """Create new suspicions from non-match clusters in word alignment."""
    _ensure_schema(conn)
    cur = conn.cursor()
    ref_norm, ref_raw = _load_tokens(cur, "ref_tokens")
    asr_norm, _ = _load_tokens(cur, "asr_tokens")
    word_ops = _load_word_alignment(cur)

    existing = {
        (row[0], row[1], row[2])
        for row in cur.execute("SELECT asr_start_idx, asr_end_idx, candidate_ref FROM asr_suspicions")
    }

    min_similarity = 1.0 - ratio_threshold
    inserted = 0
    for cluster in _clusters(word_ops):
        if len(cluster) > max_ops:
            continue
        has_del = any(op[2] == "del" for op in cluster)
        has_ins_sub = any(op[2] in ("ins", "sub") for op in cluster)
        if not (has_del and has_ins_sub):
            continue
        ref_idxs = [ri for (ri, _aj, _op) in cluster if ri is not None]
        asr_idxs = [aj for (_ri, aj, _op) in cluster if aj is not None]
        if not ref_idxs or not asr_idxs:
            continue
        ref_tokens = [ref_norm[ri] for ri in ref_idxs if 0 <= ri < len(ref_norm)]
        asr_tokens = [asr_norm[aj] for aj in asr_idxs if 0 <= aj < len(asr_norm)]
        if not _pattern_ok(len(ref_tokens), len(asr_tokens)):
            continue
        sim = phonetic_similarity(ref_tokens, asr_tokens)
        if sim < min_similarity:
            continue
        start = min(asr_idxs)
        end = max(asr_idxs) + 1
        candidate_ref = " ".join(ref_raw[ri] for ri in ref_idxs if 0 <= ri < len(ref_raw))
        key = (start, end, candidate_ref)
        if key in existing:
            continue
        cur.execute(
            "INSERT INTO asr_suspicions(asr_start_idx, asr_end_idx, candidate_ref, source, score, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (start, end, candidate_ref, "cluster_postalign", float(sim), "confirmed"),
        )
        existing.add(key)
        inserted += 1
    if inserted:
        conn.commit()
    return inserted


def apply_simple_corrections(conn: sqlite3.Connection, min_score: float = SIMPLE_CORRECTION_MIN_SCORE) -> int:
    """Fill corrected_raw for high-confidence one-to-one suspicions."""
    _ensure_schema(conn)
    cur = conn.cursor()
    confirmed = cur.execute(
        "SELECT asr_start_idx, asr_end_idx, candidate_ref, score FROM asr_suspicions WHERE status = 'confirmed'"
    ).fetchall()
    updates: List[Tuple[str, int]] = []
    for start, end, cand, score in confirmed:
        if (end - start) != 1 or score < min_score:
            continue
        cand_norm = _normalize_tokens(cand)
        if len(cand_norm) != 1:
            continue
        updates.append((cand, start))
    if updates:
        cur.executemany("UPDATE asr_tokens SET corrected_raw = ? WHERE idx = ?", updates)
        conn.commit()
    return len(updates)


def _op_in_span(op: Tuple[int | None, int | None, str], rs: int, re: int, hs: int, he: int) -> bool:
    ref_idx, asr_idx, _ = op
    ref_ok = ref_idx is None or (rs <= ref_idx < re)
    asr_ok = asr_idx is None or (hs <= asr_idx < he)
    return ref_ok and asr_ok


def recompute_wers_with_suspicions(conn: sqlite3.Connection) -> None:
    """Recompute WER/flags treating confirmed suspicion blocks as single errors."""
    _ensure_schema(conn)
    cur = conn.cursor()
    word_ops = _load_word_alignment(cur)

    cluster_ranges: Dict[Tuple[int, int], int] = {}
    for cluster in _clusters(word_ops):
        asr_idxs = [aj for (_ri, aj, _op) in cluster if aj is not None]
        if not asr_idxs:
            continue
        key = (min(asr_idxs), max(asr_idxs) + 1)
        cluster_ranges[key] = len([op for op in cluster if op[2] != "match"])

    susp_blocks = [
        (s, e) for s, e in cur.execute(
            "SELECT asr_start_idx, asr_end_idx FROM asr_suspicions WHERE status = 'confirmed'"
        ).fetchall()
    ]

    rows = cur.execute(
        "SELECT row_id, ref_start, ref_end, asr_start, asr_end FROM paragraph_rows ORDER BY row_id"
    ).fetchall()
    for row_id, rs, re, hs, he in rows:
        ops_row = [op for op in word_ops if _op_in_span(op, rs, re, hs, he)]
        errors = sum(1 for _ri, _aj, op in ops_row if op != "match")
        for s, e in susp_blocks:
            if not (hs <= e and s < he):
                continue
            compress = cluster_ranges.get((s, e))
            if compress is None:
                block_ops = [op for op in ops_row if op[1] is not None and s <= op[1] < e and op[2] != "match"]
                compress = len(block_ops)
            if compress and compress > 1:
                errors -= (compress - 1)
        errors = max(0, errors)
        wer = min(100.0, 100.0 * errors / max(1, re - rs))
        if wer == 0.0:
            flag = "OK"
        elif wer <= WARN_WER * 100.0:
            flag = "WARN"
        elif wer < 100.0:
            flag = "BAD"
        else:
            flag = "BAD"
        cur.execute("UPDATE paragraph_rows SET wer = ?, flag = ? WHERE row_id = ?", (wer, flag, row_id))
        cur.execute("UPDATE paragraphs SET wer = ?, flag = ? WHERE id = ?", (wer, flag, row_id))
    conn.commit()


def run_postalign_pipeline(db_path: str | Path) -> None:
    """Convenience wrapper to confirm suspicions, add clusters, apply corrections, and refresh WER."""
    conn = sqlite3.connect(str(db_path))
    try:
        confirm_suspicions_with_alignment(conn)
        detect_cluster_suspicions(conn)
        apply_simple_corrections(conn)
        recompute_wers_with_suspicions(conn)
    finally:
        conn.close()

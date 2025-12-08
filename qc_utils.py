"""Utility functions for QC metadata management.

Canonical row contract for QC JSON files:
    [ID, Check, OK, AI, WER, tc, Original, ASR]

- JSON (.qc.json) should always use these 8 columns in this order.
- The GUI inserts a "Score" column in-memory; if a saved row contains it (9 cols),
  it is moved to the extras tail so the first 8 columns keep the contract.
- Legacy rows with WER/tc at the end and no text are reinterpreted to the contract.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import List, Sequence

from rapidfuzz.distance import Levenshtein

from text_utils import normalize


def canonical_row(row: Sequence) -> list:
    """Return a row normalized to the 8-column contract, preserving any extras."""

    def _to_int(val):
        try:
            return int(val)
        except Exception:
            return val

    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return val

    def _is_numberish(val) -> bool:
        try:
            float(val)
            return True
        except Exception:
            return False

    def _has_letters(val) -> bool:
        return bool(re.search(r"[A-Za-z]", str(val)))

    r = list(row)
    extras: list = []

    # Score-present layout from GUI: id, check, ok, ai, score, wer, tc, original, asr
    if len(r) >= 9:
        score = r[4]
        id_, check, ok, ai, _score, wer, tc, original, asr = r[:9]
        extras = [score] + r[9:]
        base = [id_, check, ok, ai, wer, tc, original, asr]
    elif len(r) == 8:
        base = list(r[:8])
        looks_legacy = (
            _is_numberish(base[6])
            and _is_numberish(base[7])
            and not _has_letters(base[6])
            and not _has_letters(base[7])
            and (str(base[4]).strip() == "" and str(base[5]).strip() == "")
        )
        if looks_legacy:
            base = [base[0], base[1], base[2], base[3], base[6], base[7], "", ""]
    elif len(r) == 7:
        base = [r[0], r[1], r[2], "", r[3], r[4], r[5], r[6]]
    elif len(r) == 6:
        base = [r[0], r[1], "", "", r[2], r[3], r[4], r[5]]
    else:
        padded = list(r) + [""] * max(0, 8 - len(r))
        base = padded[:8]

    canonical = [
        _to_int(base[0]),
        "" if base[1] is None else str(base[1]),
        "" if base[2] is None else str(base[2]),
        "" if base[3] is None else str(base[3]),
        _to_float(base[4]),
        _to_float(base[5]) if _is_numberish(base[5]) else ("" if base[5] is None else str(base[5])),
        "" if base[6] is None else str(base[6]),
        "" if base[7] is None else str(base[7]),
    ] + list(extras)

    return canonical


def merge_qc_metadata(old_rows: List[List], new_rows: List[List]) -> List[List]:
    """Merge Check/OK/AI from old rows into new rows, keeping the canonical order."""

    merged: List[List] = []

    old_rows = [canonical_row(r) for r in old_rows]
    new_rows = [canonical_row(r) for r in new_rows]
    old_norm = [normalize(str(r[6])) for r in old_rows]

    for new in new_rows:
        base = list(new)
        n_norm = normalize(str(new[6]))
        best = None
        best_sim = 0.8
        for o_norm, o_row in zip(old_norm, old_rows):
            sim = Levenshtein.normalized_similarity(n_norm, o_norm)
            if sim > best_sim:
                best_sim = sim
                best = o_row
        if best is not None:
            base[1] = best[1]
            base[2] = best[2]
            base[3] = best[3]
            if len(best) > len(base):
                base.extend(best[len(base):])
        merged.append(base)
    return merged


def log_correction_metadata(
    json_path: str,
    row_id: str,
    original_asr: str,
    proposed_asr: str,
    verdict: str,
) -> None:
    """Logs the details of a supervised AI correction attempt."""
    if not json_path:
        return
    log_file = Path(json_path).with_suffix(".metadata.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] - Fila ID: {row_id}\n"
        f"  Veredicto: {verdict.upper()}\n"
        f"  ASR Original : {original_asr}\n"
        f"  ASR Propuesto: {proposed_asr}\n"
        f"--------------------------------------------------\n"
    )
    try:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error al escribir en el log de metadatos: {e}")

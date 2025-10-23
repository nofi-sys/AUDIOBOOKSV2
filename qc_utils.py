"""Utility functions for QC metadata management."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import List

from rapidfuzz.distance import Levenshtein

from text_utils import normalize


def merge_qc_metadata(old_rows: List[List], new_rows: List[List]) -> List[List]:
    """Merge QC columns from ``old_rows`` into ``new_rows``.

    Rows are matched based on the normalized ``Original`` text. If a row in
    ``new_rows`` is more than 80% similar to one in ``old_rows`` (Levenshtein
    normalized similarity), the values of the ``✓``, ``OK`` and ``AI`` columns
    are copied from the old row. Any extra columns beyond ``ASR`` are also
    preserved if present.
    """

    merged: List[List] = []

    old_norm = [normalize(str(r[-2])) for r in old_rows]

    has_score = any(len(r) >= 9 for r in old_rows)

    for new in new_rows:
        if has_score:
            base = [
                new[0],
                new[1],
                "",
                "",
                "",
                new[2],
                new[3],
                new[4],
                new[5],
            ]
        else:
            base = [new[0], new[1], "", "", new[2], new[3], new[4], new[5]]
        n_norm = normalize(str(new[-2]))
        best = None
        best_sim = 0.8
        for o_norm, o_row in zip(old_norm, old_rows):
            sim = Levenshtein.normalized_similarity(n_norm, o_norm)
            if sim > best_sim:
                best_sim = sim
                best = o_row
        if best is not None:
            if len(best) > 1:
                base[1] = best[1]
            if len(best) > 2:
                base[2] = best[2]
            if len(best) > 3:
                base[3] = best[3]
            if has_score and len(best) > 4:
                base[4] = best[4]
            extra_idx = 9 if has_score else 8
            if len(best) > extra_idx:
                base.extend(best[extra_idx:])
        merged.append(base)
    return merged



def canonical_row(r: list[str | float]) -> list:
    """Normalize incoming rows to the standard 8-column layout."""
    if len(r) >= 8:
        id_, flag, ok, ai, wer, tc, original, asr, *extra = r
        return [id_, flag, ok, ai, wer, tc, original, asr] + list(extra)
    if len(r) == 7:
        if isinstance(r[2], (int, float)):
            base = canonical_row(r[:6])
            return base + r[6:]
        id_, flag, ok, wer, tc, original, asr = r
        return [id_, flag, ok, '', wer, tc, original, asr]
    if len(r) == 6:
        id_, flag, wer, tc, original, asr = r
        return [id_, flag, '', '', wer, tc, original, asr]
    padded = list(r) + [''] * max(0, 6 - len(r))
    return canonical_row(padded[:6])


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
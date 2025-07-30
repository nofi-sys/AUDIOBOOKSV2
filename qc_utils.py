"""Utility functions for QC metadata management."""

from __future__ import annotations

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



def canonical_row(row: List) -> List:
    """Return row in standard QC order.

    The canonical format is either ``[ID, ✓, OK, AI, WER, tc, Original, ASR]``
    or ``[ID, ✓, OK, AI, Score, WER, tc, Original, ASR]`` when a Score column is
    present. Input rows may omit some of these columns. Missing fields are filled
    with empty strings so that the output always has either 8 or 9 elements.
    """
    if len(row) >= 9:
        return row
    if len(row) == 8:
        return row
    if len(row) == 7:
        # common case from alignment with extra "takes" list at the end
        if isinstance(row[6], list):
            # [ID, ✓, WER, tc, Original, ASR, takes]
            return [row[0], row[1], "", "", row[2], row[3], row[4], row[5], row[6]]
        # [ID, ✓, OK, WER, tc, Original, ASR]
        return [row[0], row[1], row[2], "", row[3], row[4], row[5], row[6]]
    if len(row) == 6:
        # [ID, ✓, WER, tc, Original, ASR]
        return [row[0], row[1], "", "", row[2], row[3], row[4], row[5]]
    return row

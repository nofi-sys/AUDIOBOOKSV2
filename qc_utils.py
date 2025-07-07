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

    for new in new_rows:
        base = [new[0], new[1], "", "", "", new[2], new[3], new[4], new[5]]
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
            if len(best) > 4:
                base[4] = best[4]
            if len(best) > 9:
                base.extend(best[9:])
        merged.append(base)
    return merged

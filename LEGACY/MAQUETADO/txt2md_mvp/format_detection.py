"""Utilities to detect formatting conventions in plain-text sources."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FORMAT_DIR = Path(__file__).resolve().parent / "format_rules"


@lru_cache(maxsize=1)
def _load_format_catalogue() -> List[Dict[str, object]]:
    path = FORMAT_DIR / "formats.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data


def list_known_formats() -> List[Dict[str, object]]:
    """Return the catalogue of formats with metadata."""
    return _load_format_catalogue().copy()


def detect_format_from_text(text: str, *, max_chars: int = 200_000) -> Tuple[Optional[str], Dict[str, int]]:
    """Return (format_id, counts) for the text snippet.

    counts maps format_id -> total marker occurrences. The selected format is
    the one with the highest non-zero count (ties prefer the first in catalogue).
    """
    snippet = text[:max_chars]
    catalogue = _load_format_catalogue()
    best_id: Optional[str] = None
    best_score = 0
    scores: Dict[str, int] = {}
    for entry in catalogue:
        format_id = str(entry.get("id"))
        markers = entry.get("markers", [])
        total = 0
        for marker in markers:
            pattern = marker.get("pattern")
            if not pattern:
                continue
            try:
                regex = re.compile(pattern, flags=re.MULTILINE)
            except re.error:
                continue
            matches = regex.findall(snippet)
            total += len(matches)
        scores[format_id] = total
        if total > best_score:
            best_score = total
            best_id = format_id
    if best_score == 0:
        return None, scores
    return best_id, scores


def detect_format_from_file(path: str, *, max_chars: int = 200_000) -> Tuple[Optional[str], Dict[str, int]]:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return detect_format_from_text(text, max_chars=max_chars)

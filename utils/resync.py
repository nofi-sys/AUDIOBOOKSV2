from __future__ import annotations

"""Synchronize QC JSON rows using a word-timed CSV."""

from pathlib import Path
from typing import Callable, List, Tuple
import json
import re
import unicodedata
from difflib import SequenceMatcher

_tok_re = re.compile(r"\w+['-]?\w*")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9'\-\s]", " ", text).strip()


def tokenize(text: str) -> List[str]:
    return _tok_re.findall(_norm(text))


def load_words_csv(path: Path) -> Tuple[List[str], List[float]]:
    """Read ``path`` returning tokens and timestamps."""
    words: List[str] = []
    tcs: List[float] = []
    with path.open("r", encoding="utf8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                t_str, w_raw = line.split(";", 1)
            elif "," in line:
                t_str, w_raw = line.split(",", 1)
            else:
                parts = re.split(r"\s+", line, 1)
                if len(parts) < 2:
                    continue
                t_str, w_raw = parts
            try:
                t = float(t_str.replace(",", "."))
            except ValueError:
                continue
            for tok in tokenize(w_raw):
                words.append(tok)
                tcs.append(round(t, 2))
    return words, tcs


def find_anchors(csv_words: List[str], json_words: List[str]) -> List[Tuple[int, int, int]]:
    """Return matching n-gram anchors between CSV and JSON words."""
    anchors: List[Tuple[int, int, int]] = []
    used_csv: set[int] = set()
    used_json: set[int] = set()
    for n in (5, 4, 3, 2):
        csv_map = {" ".join(csv_words[i : i + n]): i for i in range(len(csv_words) - n + 1)}
        j = 0
        while j <= len(json_words) - n:
            if any((j + k) in used_json for k in range(n)):
                j += 1
                continue
            key = " ".join(json_words[j : j + n])
            i = csv_map.get(key)
            if i is not None and not any((i + k) in used_csv for k in range(n)):
                anchors.append((j, i, n))
                for k in range(n):
                    used_json.add(j + k)
                    used_csv.add(i + k)
                j += n
                continue
            j += 1
    return sorted(anchors, key=lambda a: a[0])


def _align_chunk(j_words: List[str], c_words: List[str]) -> List[int]:
    sm = SequenceMatcher(None, j_words, c_words, autojunk=False)
    mapping = [-1] * len(j_words)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                mapping[i1 + k] = j1 + k
    last = -1
    prev_idx = -1
    for idx, val in enumerate(mapping):
        if val != -1:
            if last == -1:
                for k in range(idx):
                    mapping[k] = val
            else:
                step = (val - last) / (idx - prev_idx)
                for k in range(prev_idx + 1, idx):
                    mapping[k] = round(last + step * (k - prev_idx))
            prev_idx, last = idx, val
    if mapping and mapping[-1] == -1:
        last_idx = max((i for i, v in enumerate(mapping) if v != -1), default=None)
        if last_idx is not None:
            for k in range(last_idx + 1, len(mapping)):
                mapping[k] = mapping[last_idx]
    return mapping


def resync_rows(
    rows: List[List],
    csv_words: List[str],
    csv_tcs: List[float],
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> List[List]:
    """Update rows assigning ``tc`` from ``csv_tcs``."""

    if log_cb is None:
        log_cb = lambda *_: None
    if progress_cb is None:
        progress_cb = lambda *_: None

    j_tokens: List[str] = []
    tok2row: List[int] = []
    for ridx, row in enumerate(rows):
        toks = tokenize(str(row[-1]))
        j_tokens.extend(toks)
        tok2row.extend([ridx] * len(toks))

    anchors = find_anchors(csv_words, j_tokens)
    log_cb(f"Encontradas {len(anchors)} anclas")

    mapping = [-1] * len(j_tokens)
    for jidx, cidx, n in anchors:
        for k in range(n):
            mapping[jidx + k] = cidx + k

    prev_j = prev_c = 0
    anchors.append((len(j_tokens), len(csv_words), 0))
    for j, c, n in anchors:
        j_chunk = j_tokens[prev_j:j]
        c_chunk = csv_words[prev_c:c]
        if j_chunk and c_chunk:
            local_map = _align_chunk(j_chunk, c_chunk)
            for off, cm in enumerate(local_map):
                mapping[prev_j + off] = prev_c + cm if cm != -1 else prev_c
        prev_j, prev_c = j + n, c + n

    row_tc: List[float | None] = [None] * len(rows)
    for jidx, cidx in enumerate(mapping):
        ridx = tok2row[jidx]
        if row_tc[ridx] is None and cidx != -1:
            row_tc[ridx] = csv_tcs[cidx]

    last_tc = 0.0
    for i in range(len(row_tc)):
        if row_tc[i] is None:
            row_tc[i] = last_tc
        else:
            last_tc = row_tc[i]

    for i, row in enumerate(rows):
        if len(row) > 5:
            row[5] = f"{row_tc[i]:.2f}"
        else:
            row.append(f"{row_tc[i]:.2f}")
        if i % 10 == 0:
            progress_cb(i / len(rows))
    return rows


def resync_file(json_path: str | Path, csv_path: str | Path) -> List[List]:
    """Return rows from ``json_path`` with updated ``tc`` using ``csv_path``."""

    rows = json.loads(Path(json_path).read_text(encoding="utf8"))
    csv_words, csv_tcs = load_words_csv(Path(csv_path))
    resync_rows(rows, csv_words, csv_tcs)
    return rows

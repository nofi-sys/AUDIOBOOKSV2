from __future__ import annotations
"""Merged alignment routines for QC application (2025‑07‑fix).

✅ Esta versión corrige el *desfase de columnas* que observaste —el start‑time
quedaba en la celda ASR y el tc mostraba 0⁠s— y entrega los mismos seis campos
que esperaba la GUI original:

    [ID, flag, WER%, tc, ASR, Original]

* **tc** ahora es una **cadena time‑code** «HH:MM:SS.d» (una décima) igual
  que antes.
* El ASR y el Original vuelven a su orden histórico, de modo que
  `_row_from_alignment` ya no los invierte.
* El resto del algoritmo (DTW, refinamiento, repetición, word‑level) se mantiene.
"""

from typing import Callable, List, Tuple
import json
import re
from rapidfuzz.distance import Levenshtein
from text_utils import (
    normalize,
    token_equal,
    STOP,
    STOP_WEIGHT,
    find_anchor_trigrams,
)

# ───────────────────────────────────────── debug ───────────────────────────────────
DEBUG_LOGGER: Callable[[str], None] = lambda m: None

def set_debug_logger(logger: Callable[[str], None]) -> None:  # pragma: no cover
    global DEBUG_LOGGER
    DEBUG_LOGGER = logger

def _d(msg: str) -> None:  # internal helper
    DEBUG_LOGGER(msg)

# ───────────────────────────────────── parámetros ──────────────────────────────────
COARSE_W = 40      # banda inicial DTW
WARN_WER = 0.08    # umbral ⚠️
GAMMA_TIME = 0.3   # peso de la penalización temporal
WORDS_PER_SEC = 3.0

# ───────────────────────────── helpers de formato time‑code ────────────────────────

def _sec_to_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:05.1f}"

# ──────────────────────────────  núcleo DTW (idéntico)  ────────────────────────────

def _dtw_band(a: List[str], b: List[str], w: int) -> List[Tuple[int, int]]:
    n, m = len(a), len(b)
    W = max(w, abs(n - m))
    BIG = 1e9
    D = {(-1, -1): (0.0, None)}
    back: dict[tuple[int, int], tuple[int, int] | None] = {}
    for i in range(n):
        lo, hi = max(0, i - W), min(m - 1, i + W)
        for j in range(lo, hi + 1):
            best_cost, best_prev = BIG, None
            for di, dj, mv in ((-1, 0, 1), (0, -1, 1), (-1, -1, 0)):
                prev = (i + di, j + dj)
                if prev in D:
                    c = D[prev][0] + mv
                    if c < best_cost:
                        best_cost, best_prev = c, prev
            match = 0.0 if token_equal(a[i], b[j]) else (STOP_WEIGHT if a[i] in STOP or b[j] in STOP else 1.0)
            pos = GAMMA_TIME * abs((i / n) - (j / m))
            D[(i, j)] = (best_cost + match + pos, best_prev)
            back[(i, j)] = best_prev
    if (n - 1, m - 1) not in back:
        raise RuntimeError("DTW window too narrow")
    path: list[tuple[int, int]] = []
    i, j = n - 1, m - 1
    while (i, j) != (-1, -1):
        path.append((i, j))
        prev = back[(i, j)]
        if prev is None:
            break
        i, j = prev
    return path[::-1]


def _fallback_pairs(r: List[str], h: List[str]) -> List[Tuple[int, int]]:
    pairs, j_last = [], 0
    for i, tok in enumerate(r):
        for j in range(j_last, len(h)):
            if token_equal(tok, h[j]):
                pairs.append((i, j))
                j_last = j + 1
                break
    return pairs


def _safe_dtw(a: List[str], b: List[str], w: int = COARSE_W) -> List[Tuple[int, int]]:
    band, max_band = w, max(len(a), len(b)) * 2
    while band <= max_band:
        try:
            _d(f"DTW band {band}")
            return _dtw_band(a, b, band)
        except RuntimeError:
            band *= 2
    _d("DTW fallback")
    return _fallback_pairs(a, b)

# ────────────────────────────── utilidades varias ────────────────────────────────

def _sentence_spans(tok: list[str]) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for sent in re.split(r"(?<=[\.\?\!])\s+", " ".join(tok)):
        w = sent.split()
        if w:
            spans.append((pos, pos + len(w)))
            pos += len(w)
    return spans


def _flag_wer(ref_t: list[str], hyp_t: list[str]) -> tuple[str, float]:
    if not hyp_t:
        return "❌", 100.0
    wer = Levenshtein.normalized_distance(ref_t, hyp_t)
    base = Levenshtein.normalized_distance([w.strip(".,;!") for w in ref_t], [w.strip(".,;!") for w in hyp_t])
    if base <= 0.05:
        return "✅", wer * 100
    thr = 0.20 if len(ref_t) < 5 else WARN_WER
    flag = "✅" if wer <= thr else ("⚠️" if wer <= 0.20 else "❌")
    return flag, wer * 100

# ─────────────────────────────  ALIGN build_rows  ───────────────────────────────

def build_rows(ref: str, hyp: str) -> list[list]:
    """Sentence‑level alignment. Returns rows: [ID, flag, WER, tc, ASR, Original]"""
    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = normalize(hyp, strip_punct=False).split()
    anchor = find_anchor_trigrams(ref_tok, hyp_tok)

    # DTW secciones
    pairs, seg = [], [(-1, -1)] + anchor + [(len(ref_tok) - 1, len(hyp_tok) - 1)]
    for (pi, pj), (ni, nj) in zip(seg[:-1], seg[1:]):
        if ni > pi + 1 and nj > pj + 1:
            sr, sh = ref_tok[pi + 1:ni], hyp_tok[pj + 1:nj]
            if sr and sh:
                ref_idx = [i for i, t in enumerate(sr) if t not in STOP]
                hyp_idx = [j for j, t in enumerate(sh) if t not in STOP]
                sub = _safe_dtw([sr[i] for i in ref_idx], [sh[j] for j in hyp_idx])
                pairs += [(pi + 1 + ref_idx[ri], pj + 1 + hyp_idx[hj]) for ri, hj in sub]
        if 0 <= ni < len(ref_tok) and 0 <= nj < len(hyp_tok):
            pairs.append((ni, nj))
    pairs.sort()

    # dedup
    used_h, map_h = {}, [-1] * len(ref_tok)
    for ri, hj in pairs:
        if hj not in used_h or abs(ri - used_h[hj]) > 1:
            used_h[hj] = ri
            map_h[ri] = hj

    rows, consumed, t_start = [], set(), 0.0
    for s, e in _sentence_spans(ref_tok):
        ref_seg = ref_tok[s:e]
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            hs, he = min(idx), max(idx) + 1
            consumed.update(range(hs, he))
            asr = " ".join(hyp_tok[hs:he])
        else:
            asr = ""
        flag, werp = _flag_wer(ref_seg, asr.split())
        dur = len(asr.split()) / WORDS_PER_SEC
        rows.append([len(rows), flag, round(werp, 1), _sec_to_tc(t_start), asr, " ".join(ref_seg)])
        t_start += dur

    # tokens sobrantes
    extra_idx = [i for i in range(len(hyp_tok)) if i not in consumed]
    if extra_idx:
        extra = " ".join(hyp_tok[min(extra_idx):])
        rows.append([len(rows), "❌", 100.0, _sec_to_tc(t_start), extra, ""])

    return rows

# ────────────────────────── build_rows_wordlevel (tc real) ───────────────────────

def build_rows_wordlevel(ref: str, asr_word_json: str) -> list[list]:
    data = json.loads(asr_word_json)
    words = [
        {
            "norm": normalize(w.get("word", w.get("text", "")), strip_punct=False),
            "start": float(w.get("start", seg.get("start", 0.0))),
            "end": float(w.get("end", seg.get("end", 0.0))),
        }
        for seg in data.get("segments", data) for w in seg.get("words", [])
    ]
    hyp_tok, ref_tok = [w["norm"] for w in words], normalize(ref, strip_punct=False).split()
    pairs = _safe_dtw(ref_tok, hyp_tok)
    map_h = [-1] * len(ref_tok)
    for i, j in pairs:
        if map_h[i] == -1:
            map_h[i] = j

    rows, consumed = [], set()
    for s, e in _sentence_spans(ref_tok):
        ref_seg = ref_tok[s:e]
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            hs, he = min(idx), max(idx) + 1
            consumed.update(range(hs, he))
            asr = " ".join(hyp_tok[hs:he])
            tc = _sec_to_tc(words[hs]["start"])
        else:
            asr, tc = "", _sec_to_tc(0.0)
        flag, werp = _flag_wer(ref_seg, asr.split())
        rows.append([len(rows), flag, round(werp, 1), tc, asr, " ".join(ref_seg)])

    leftover = [i for i in range(len(hyp_tok)) if i not in consumed]
    if leftover:
        rows.append([len(rows), "❌", 100.0, _sec_to_tc(words[min(leftover)]["start"]), " ".join(hyp_tok[min(leftover):]), ""])
    return rows

from __future__ import annotations
"""
Alignment routines para la QC-app (versión 2025-08-01).

▪️ Devuelve SIEMPRE filas con **seis columnas** en este orden
    [ID, flag, WER, tc, Original, ASR]

    • `flag`  →  ✅ / ⚠️ / ❌
    • `WER`   →  Porcentaje (float, 0-100)
    • `tc`    →  time-code «HH:MM:SS.d» (una décima)
    • `Original`  →  Texto de referencia
    • `ASR`       →  Texto transcrito

Las dos variantes expuestas son:

    build_rows(ref: str, hyp: str)
        – Alinea por tokens (no requiere time-codes).
          El `tc` se calcula por acumulación (≈ 3 palabras/segundo).

    build_rows_wordlevel(ref: str, asr_word_json: str)
        – Usa un JSON de palabras con start/end; el `tc`
          toma el `start` real del primer token alineado.

Constante pública:
    COL_ORDER = ("ID", "flag", "WER", "tc", "Original", "ASR")
"""

# ───────────────────────── imports & const ──────────────────────────
from typing import Callable, List, Tuple
import json
import re
import os
from rapidfuzz.distance import Levenshtein
from text_utils import (
    normalize,
    token_equal,
    STOP,
    STOP_WEIGHT,
    find_anchor_trigrams,
)

from rectifier import rectify_rows, RectifyReport

__all__ = ["COL_ORDER", "build_rows", "build_rows_wordlevel"]

COL_ORDER = ("ID", "flag", "WER", "tc", "Original", "ASR")

COARSE_W     = 40      # banda inicial DTW
WARN_WER     = 0.08    # umbral para ⚠️
GAMMA_TIME   = 0.3     # penalización temporal (proporción)
WORDS_PER_SEC = 3.0    # usado en build_rows (sin time-codes)
_DEFUZZ = 1

# ─────────────────────── debug facilito ─────────────────────────────
DEBUG_LOGGER: Callable[[str], None] = lambda m: None


def set_debug_logger(logger: Callable[[str], None]) -> None:  # pragma: no cover
    global DEBUG_LOGGER
    DEBUG_LOGGER = logger

def _d(msg: str) -> None:
    DEBUG_LOGGER(msg)

def _log_rectify_report(report: RectifyReport) -> None:
    parts: list[str] = [f"anchors={len(report.anchors)}"]
    if report.total_moves:
        parts.append(f"moves={report.total_moves}")
    if report.empty_rows:
        parts.append(f"empty={len(report.empty_rows)}")
    if report.anomalies:
        parts.append(f"issues={len(report.anomalies)}")
    if report.notes:
        parts.extend(report.notes)
    summary = ', '.join(parts) if parts else 'clean'
    _d(f"R2-report: {summary}")

# ───────────────────── helpers de time-code ─────────────────────────
def _sec_to_tc(sec: float) -> float:
    """Convert seconds to a compact float with two decimals for storage."""
    try:
        return round(float(sec), 2)
    except Exception:
        return 0.0

def _similar(a: str, b: str) -> bool:
    # reutiliza el import existente
    return a == b or Levenshtein.distance(a, b) <= _DEFUZZ

def _join(tok, i, n) -> str:
    return " ".join(tok[i:i+n])
def _choose_run(idx: list[int], ref_len: int, max_gap: int = 12,
                max_ratio: float = 2.5, extra: int = 12) -> tuple[int, int]:
    """
    Elige la corrida contigua de índices (brechas ≤ max_gap) y
    limita su ancho a (ref_len * max_ratio + extra).
    Devuelve (hs, he) tipo slice.
    """
    if not idx:
        return (0, 0)
    idx = sorted(idx)
    runs: list[list[int]] = []
    cur = [idx[0]]
    for a, b in zip(idx, idx[1:]):
        if b - a <= max_gap:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    # run preferida: más larga; si empata, la de menor span
    runs.sort(key=lambda r: (-(len(r)), (r[-1] - r[0])))
    hs, he = runs[0][0], runs[0][-1] + 1

    limit = int(ref_len * max_ratio) + extra
    if (he - hs) > limit:
        mid = runs[0][len(runs[0]) // 2]
        hs = max(hs, mid - limit // 2)
        he = hs + limit
    return hs, he

def _find_anchors_fuzzy(ref_tok: list[str], hyp_tok: list[str]) -> list[tuple[int,int]]:
    anchors: list[tuple[int,int,int]] = []
    used_r: set[int] = set(); used_h: set[int] = set()

    for n in (5,4,3,2):
        ref_map = { _join(ref_tok, i, n): i for i in range(len(ref_tok)-n+1) }
        j = 0
        while j <= len(hyp_tok) - n:
            if any((j+k) in used_h for k in range(n)):
                j += 1; continue
            key = _join(hyp_tok, j, n)
            i = ref_map.get(key)
            if i is None and n == 2:
                w0, w1 = hyp_tok[j:j+2]
                for i2 in range(len(ref_tok)-1):
                    if _similar(w0, ref_tok[i2]) and _similar(w1, ref_tok[i2+1]):
                        i = i2; break
            if i is not None and not any((i+k) in used_r for k in range(n)):
                anchors.append((i, j, n))
                for k in range(n): used_r.add(i+k); used_h.add(j+k)
                j += n; continue
            j += 1

    # respaldo unigram disperso
    for j in range(0, len(hyp_tok), 20):
        if j in used_h: continue
        w = hyp_tok[j]
        try:
            i = next(ix for ix,x in enumerate(ref_tok) if _similar(w, x) and ix not in used_r)
        except StopIteration:
            continue
        anchors.append((i, j, 1))
        used_r.add(i); used_h.add(j)

    anchors.sort(key=lambda x: x[0])
    # devolvemos solo (i,j)
    return [(i,j) for (i,j,_) in anchors]

# ───────────────────── núcleo DTW seguro ────────────────────────────
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
            match = 0.0 if token_equal(a[i], b[j]) else (
                STOP_WEIGHT if a[i] in STOP or b[j] in STOP else 1.0
            )
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

# ───────────────── utilidades de segmentación & WER ─────────────────
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
    base = Levenshtein.normalized_distance(
        [w.strip(".,;!") for w in ref_t],
        [w.strip(".,;!") for w in hyp_t],
    )
    if base <= 0.05:
        return "✅", wer * 100
    thr  = 0.20 if len(ref_t) < 5 else WARN_WER
    flag = "✅" if wer <= thr else ("⚠️" if wer <= 0.20 else "❌")
    return flag, wer * 100

# ─────────────────────────── build_rows (tokens) ────────────────────
def build_rows(ref: str, hyp: str) -> list[list]:
    """
    Alinea `ref` vs `hyp` por tokens, sin información temporal externa.

    Devuelve filas con orden:
        [ID, flag, WER, tc, Original, ASR]
    """
    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = normalize(hyp, strip_punct=False).split()

    # 1) anclas + DTW
    anchor = find_anchor_trigrams(ref_tok, hyp_tok)
    pairs: list[tuple[int, int]] = []
    seg = [(-1, -1)] + anchor + [(len(ref_tok) - 1, len(hyp_tok) - 1)]
    for (pi, pj), (ni, nj) in zip(seg[:-1], seg[1:]):
        if ni > pi + 1 and nj > pj + 1:
            sr, sh = ref_tok[pi + 1 : ni], hyp_tok[pj + 1 : nj]
            if sr and sh:
                sub = _safe_dtw(sr, sh)
                pairs += [(pi + 1 + ri, pj + 1 + hj) for ri, hj in sub]
        if 0 <= ni < len(ref_tok) and 0 <= nj < len(hyp_tok):
            pairs.append((ni, nj))
    pairs.sort()

    # 2) mapa ref→hyp (evita reuse de tokens hyp)
    used_h, map_h = {}, [-1] * len(ref_tok)
    for ri, hj in pairs:
        if hj not in used_h or abs(ri - used_h[hj]) > 1:
            used_h[hj] = ri
            map_h[ri] = hj

    # 3) construir filas
    rows, consumed, t_start = [], set(), 0.0
    for s, e in _sentence_spans(ref_tok):
        ref_seg = ref_tok[s:e]
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            #hs, he = min(idx), max(idx) + 1
            hs, he = _choose_run(idx, len(ref_seg))
            consumed.update(range(hs, he))
            asr_text = " ".join(hyp_tok[hs:he])
        else:
            asr_text = ""

        flag, werp = _flag_wer(ref_seg, asr_text.split())
        dur = len(asr_text.split()) / WORDS_PER_SEC  # aproximado
        rows.append([
            len(rows),                 # ID
            flag,                      # flag
            round(werp, 1),            # WER
            _sec_to_tc(t_start),       # tc acumulado
            " ".join(ref_seg),         # Original
            asr_text,                  # ASR
        ])
        t_start += dur

    # 4) tokens ASR sobrantes
    extra = [i for i in range(len(hyp_tok)) if i not in consumed]
    if extra:
        rows.append([
            len(rows),
            "❌",
            100.0,
            _sec_to_tc(t_start),
            "",
            " ".join(hyp_tok[min(extra):]),
        ])
    return rows

# ─────────────────── build_rows_wordlevel (con JSON) ────────────────
def build_rows_wordlevel(ref: str, asr_word_json: str) -> list[list]:
    """
    Igual que build_rows, pero usando time-codes reales del ASR
    (json → «segments» ↦ «words» con start/end).
    """
    data = json.loads(asr_word_json)
    words = [
        {
            "norm":  normalize(w.get("word", w.get("text", "")), strip_punct=False),
            "start": float(w.get("start", seg.get("start", 0.0))),
            "end":   float(w.get("end",   seg.get("end",   0.0))),
        }
        for seg in data.get("segments", data)
        for w   in seg.get("words", [])
    ]
    hyp_tok   = [w["norm"] for w in words]
    ref_tok   = normalize(ref, strip_punct=False).split()
    pairs     = _safe_dtw(ref_tok, hyp_tok)
    map_h     = [-1] * len(ref_tok)
    for i, j in pairs:
        if map_h[i] == -1:
            map_h[i] = j

    rows, consumed = [], set()
    for s, e in _sentence_spans(ref_tok):
        ref_seg = ref_tok[s:e]
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            #hs, he = min(idx), max(idx) + 1
            hs, he = _choose_run(idx, len(ref_seg))
            consumed.update(range(hs, he))
            asr_text = " ".join(hyp_tok[hs:he])
            tc       = _sec_to_tc(words[hs]["start"])
        else:
            asr_text, tc = "", _sec_to_tc(0.0)

        flag, werp = _flag_wer(ref_seg, asr_text.split())
        rows.append([
            len(rows),
            flag,
            round(werp, 1),
            tc,
            " ".join(ref_seg),
            asr_text,
        ])

    leftover = [i for i in range(len(hyp_tok)) if i not in consumed]
    if leftover:
        rows.append([
            len(rows),
            "KO",
            100.0,
            _sec_to_tc(words[min(leftover)]["start"]),
            "",
            " ".join(hyp_tok[min(leftover):]),
        ])

    if words and not os.getenv("QC_SKIP_REFINER"):
        try:
            csv_words = [w["norm"] for w in words]
            csv_tcs = [w["start"] for w in words]
            refined, report = rectify_rows(
                rows, csv_words, csv_tcs, log=_d, flag_fn=_flag_wer, return_report=True
            )
            rows = refined
            _log_rectify_report(report)
            if report.anomalies:
                for idx in report.anomalies:
                    if 0 <= idx < len(rows):
                        flag = str(rows[idx][1])
                        if flag not in {"KO", "bad"}:
                            rows[idx][1] = "??"
        except Exception as exc:
            _d(f"Rectify failed: {exc}")
    return rows

# NUEVA función: alinea usando palabras+tiempos del CSV
def build_rows_from_words(ref: str, csv_words: list[str], csv_tcs: list[float]) -> list[list]:
    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = [normalize(w, strip_punct=False) for w in csv_words]

    anchors = _find_anchors_fuzzy(ref_tok, hyp_tok)

    pairs: list[tuple[int,int]] = []
    seg = [(-1,-1)] + anchors + [(len(ref_tok)-1, len(hyp_tok)-1)]
    for (pi,pj),(ni,nj) in zip(seg[:-1], seg[1:]):
        if ni > pi+1 and nj > pj+1:
            sr, sh = ref_tok[pi+1:ni], hyp_tok[pj+1:nj]
            if sr and sh:
                sub = _safe_dtw(sr, sh)
                pairs += [(pi+1+ri, pj+1+hj) for ri,hj in sub]
        if 0 <= ni < len(ref_tok) and 0 <= nj < len(hyp_tok):
            pairs.append((ni, nj))
    pairs.sort()

    used_h: dict[int,int] = {}
    map_h = [-1]*len(ref_tok)
    for ri, hj in pairs:
        # permitimos re-uso lejano para manejar repeticiones largas
        if hj not in used_h or abs(ri - used_h[hj]) > 5:
            used_h[hj] = ri
            map_h[ri] = hj

    rows: list[list] = []
    consumed: set[int] = set()
    last_h = 0

    for s, e in _sentence_spans(ref_tok):
        ref_seg = ref_tok[s:e]
        idx = [map_h[k] for k in range(s,e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            #hs, he = min(idx), max(idx)+1
            hs, he = _choose_run(idx, len(ref_seg))

            # opcional: insertar huecos solo-ASR (activar con QC_INSERT_SOLO_ASR=1)
            if os.getenv("QC_INSERT_SOLO_ASR") and hs > last_h:
                rows.append([len(rows), "❌", 100.0, _sec_to_tc(csv_tcs[last_h]), "", " ".join(hyp_tok[last_h:hs])])

            consumed.update(range(hs, he))
            asr_text = " ".join(hyp_tok[hs:he])
            tc = _sec_to_tc(csv_tcs[hs])  # ← start real de la PRIMERA palabra
            last_h = he
        else:
            asr_text = ""
            tc = _sec_to_tc(csv_tcs[last_h]) if last_h < len(csv_tcs) else _sec_to_tc(0.0)

        flag, werp = _flag_wer(ref_seg, asr_text.split())
        rows.append([len(rows), flag, round(werp,1), tc, " ".join(ref_seg), asr_text])

    if last_h < len(hyp_tok):
        rows.append([
            len(rows), "KO", 100.0, _sec_to_tc(csv_tcs[last_h]), "", " ".join(hyp_tok[last_h:])
        ])

    if csv_words and not os.getenv("QC_SKIP_REFINER"):
        try:
            norm_words = [normalize(w, strip_punct=False) for w in csv_words]
            refined, report = rectify_rows(
                rows, norm_words, csv_tcs, log=_d, flag_fn=_flag_wer, return_report=True
            )
            rows = refined
            _log_rectify_report(report)
            if report.anomalies:
                for idx in report.anomalies:
                    if 0 <= idx < len(rows):
                        flag = str(rows[idx][1])
                        if flag not in {"KO", "bad"}:
                            rows[idx][1] = "??"
        except Exception as exc:
            _d(f"Rectify failed: {exc}")

    return rows
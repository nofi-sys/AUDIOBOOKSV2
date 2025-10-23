from __future__ import annotations
# alignment.py — versión conservadora y compatible (2025-08-14)
# - Mantiene la interfaz y formato de salida que espera qc_app.py
# - Alineación por oración con ventana local (rápida y estable)
# - “Barandas”:
#     * corta corridas por pausas reales y duración esperada
#     * post-paso de rebalance (mueve ≤3 palabras si baja WER total)
#     * protege anclas (fechas, meses) para no desordenarlas
#
# Salida: lista de filas [ID, flag, "", "", WER(0-100), tc(seg), Original, ASR]

import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Callable, List, Tuple

# ───────────────────────── Configuración ─────────────────────────

# Umbral que usa el GUI (Levenshtein.normalized_distance ∈ [0..1])
# ¡Ojo! El GUI espera FRACCIÓN, no porcentaje:
WARN_WER = 0.12  # 12%

# Tokenización y similitud
SIM_THRESH = 0.80              # similitud normalizada mínima para considerar “match” (tokens)
PUNCT_RE = r"""["#$%&'()*+,\-/:;<=>@\[\]^_`{|}~¡¿…—–]"""

# Ventanas / pausas / tiempos “razonables”
BAND_CAP = 64                  # límite interno de la “banda” para DP local
MAX_TOKEN_GAP = 12             # rompe corrida si gap de índices > N
PAUSE_SPLIT_SEC = 0.80         # corta corrida si hay hueco > 0.8 s entre palabras contiguas
WPS = 2.5                      # palabras/seg esperadas (≈150 wpm)
MAX_RATIO = 2.5                # límite ancho relativo a len(ref_seg)
MAX_EXTRA = 12                 # extra absoluto (tokens) permitido

# Rebalance de bordes
REBALANCE_MAX_MOVE = 3         # mueve hasta 3 palabras
REBALANCE_MIN_IMPROVE = 2.0    # mejora mínima (en puntos de WER)
REBALANCE_MAX_PAUSE = 0.80     # no cruza pausas largas

MONTHS = {
    "enero","febrero","marzo","abril","mayo","junio","julio",
    "agosto","septiembre","setiembre","octubre","noviembre","diciembre"
}

# Logger opcional
_DEBUG: Callable[[str], None] = lambda _msg: None
def set_debug_logger(logger: Callable[[str], None]) -> None:
    """Permite que la app inyecte su logger de debug."""
    global _DEBUG
    _DEBUG = logger

def _d(msg: str) -> None:  # pragma: no cover
    _DEBUG(msg)

# ───────────────────────── Utilidades ─────────────────────────

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def _normalize_token(s: str) -> str:
    s = _strip_accents(s.lower())
    # Preserve sentence-ending punctuation for splitting
    s = re.sub(r'[!"#$%&\'()*+,\-/:;<=>?@\[\]^_`{|}~¡¿…—–]', " ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _tokenize(text: str) -> List[str]:
    t = _normalize_token(text)
    return t.split() if t else []

def _sec_to_str(sec: float) -> str:
    """Devuelve segundos en str con 2 decimales (el GUI lo re-formatea)."""
    try:
        return f"{float(sec):.2f}"
    except Exception:
        return str(sec)

def _lev(a: List[str], b: List[str]) -> int:
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return lb if la == 0 else la
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]
        dp[0] = i
        ca = a[i-1]
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if ca == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[lb]

def _wer_pct(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 100.0
    return min(100.0, 100.0 * _lev(ref_tokens, hyp_tokens) / len(ref_tokens))

def _flag_for_wer(wer_pct: float, ref_len: int) -> str:
    thr = 20.0 if ref_len < 5 else (WARN_WER * 100.0)
    if wer_pct <= thr:
        return "✅"
    if wer_pct <= 20.0:
        return "⚠️"
    return "❌"

# Oraciones: partimos por signos de cierre
_SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+")
def _sentence_spans(ref_tokens: List[str]) -> List[Tuple[int,int]]:
    text = " ".join(ref_tokens)
    sents = _SENT_SPLIT_RE.split(text) if text else []
    spans, pos = [], 0
    for s in sents:
        w = s.split()
        if w:
            spans.append((pos, pos + len(w)))
            pos += len(w)
    if not spans and ref_tokens:
        spans = [(0, len(ref_tokens))]
    return spans

def _is_anchor(tok: str) -> bool:
    return tok in MONTHS or re.fullmatch(r"\d{1,2}|\d{4}", tok) is not None

# ───────────────────────── Selección por pausas/duración ─────────────────────────

def _split_runs_by_pause(idxs: List[int], tcs: List[float],
                         max_gap: int = MAX_TOKEN_GAP,
                         pause_sec: float = PAUSE_SPLIT_SEC) -> List[List[int]]:
    if not idxs:
        return []
    idxs = sorted(idxs)
    runs, cur = [], [idxs[0]]
    for a, b in zip(idxs, idxs[1:]):
        if (b - a) <= max_gap and (tcs[b] - tcs[a]) <= pause_sec:
            cur.append(b)
        else:
            runs.append(cur); cur = [b]
    runs.append(cur)
    return runs

def _choose_best_run(runs: List[List[int]], ref_len: int, tcs: List[float]) -> Tuple[int,int]:
    if not runs:
        return (0, 0)
    expected = max(1e-6, ref_len / max(1e-6, WPS))
    best = None
    for r in runs:
        hs, he = r[0], r[-1] + 1
        dur = max(1e-6, tcs[he-1] - tcs[hs])
        coverage = len(r)
        score = coverage / (1.0 + abs(dur - expected))
        cand = (score, -abs(dur-expected), coverage, hs, he)
        if best is None or cand > best:
            best = cand
    _, _, _, hs, he = best
    # limitar spans exagerados
    limit = int(ref_len * MAX_RATIO) + MAX_EXTRA
    if (he - hs) > limit:
        mid = hs + (he - hs)//2
        hs = max(hs, mid - limit//2)
        he = hs + limit
    return hs, he

# ───────────────────────── Alineación por oración (ventana local) ─────────────────────────

def _match_sent_window(seg: List[str], asr: List[str], lo: int, hi: int) -> List[int]:
    """
    Aproxima mapeos usando SequenceMatcher sobre una ventana local asr[lo:hi].
    Devuelve índices de ASR donde hay “match” (no necesariamente 1-a-1).
    """
    if lo >= hi or not seg:
        return []
    loc = asr[lo:hi]
    # SequenceMatcher sobre tokens
    sm = SequenceMatcher(a=seg, b=loc, autojunk=False)
    blocks = sm.get_matching_blocks()  # [(i, j, n), ...] + (len, len, 0)
    idxs = []
    for (i, j, n) in blocks:
        if n <= 0:
            continue
        # recoger tramo coincidente
        for off in range(n):
            idxs.append(lo + j + off)
    return sorted(set(idxs))

def _align_sentwise(ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> List[Tuple[int,int]]:
    """
    Devuelve pares (i_ref, j_asr) únicamente en posiciones “diagonales”
    (similares) por ventana local. Se usa para guiar el recorte por oración.
    """
    pairs: List[Tuple[int,int]] = []
    last = 0
    spans = _sentence_spans(ref_tokens)

    for (s, e) in spans:
        seg = ref_tokens[s:e]
        if not seg:
            continue
        # Ventana cerca de 'last': crece con el largo de la oración y un margen
        approx = last
        win = min(len(asr_tokens), approx + int(MAX_RATIO * max(8, (e - s))) + 200)
        lo = approx
        # Cortar por pausa grande cercana para no engordar demasiado la ventana
        for j in range(approx + 1, min(win, len(tcs))):
            if tcs[j] - tcs[j-1] > PAUSE_SPLIT_SEC:
                win = j + 1
                break

        idxs = _match_sent_window(seg, asr_tokens, lo, win)
        # proyectar a pares “suaves”: usamos solo algunos puntos de anclaje
        for j in idxs:
            # emparejamos con el token de ref “más cercano” dentro del segmento
            # (no perfecto, pero suficiente para cortar por corridas y pausas)
            i_ref = s + min(e - s - 1, max(0, j - lo))
            pairs.append((i_ref, j))

        if idxs:
            last = max(last, idxs[-1] + 1)

    return pairs

# ───────────────────────── Rebalance de fronteras ─────────────────────────

def _recompute_row(row, ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> None:
    seg_ref = ref_tokens[row['s']:row['e']]
    seg_asr = asr_tokens[row['hs']:row['he']]
    wer = _wer_pct(seg_ref, seg_asr)
    row['wer'] = wer
    row['flag'] = _flag_for_wer(wer, len(seg_ref))
    row['txt_ref'] = " ".join(seg_ref)
    row['txt_asr'] = " ".join(seg_asr)
    row['tc'] = _sec_to_str(tcs[row['hs']]) if row['hs'] < len(tcs) else _sec_to_str(0.0)

def _try_move_front(next_row, cur_row, k, asr_tokens, tcs, ref_tokens) -> bool:
    if k <= 0 or next_row['hs'] + k >= next_row['he']:
        return False
    if (tcs[next_row['hs']+k-1] - tcs[next_row['hs']]) > REBALANCE_MAX_PAUSE:
        return False
    moved = asr_tokens[next_row['hs']:next_row['hs']+k]
    if any(_is_anchor(w) for w in moved):
        return False
    old_sum = cur_row.get('wer', 100.0) + next_row.get('wer', 100.0)
    cur_row['he']  += k
    next_row['hs'] += k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    new_sum = cur_row['wer'] + next_row['wer']
    if (old_sum - new_sum) >= REBALANCE_MIN_IMPROVE:
        return True
    # revertir
    cur_row['he']  -= k
    next_row['hs'] -= k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    return False

def _try_move_back(cur_row, next_row, k, asr_tokens, tcs, ref_tokens) -> bool:
    if k <= 0 or cur_row['hs'] + k >= cur_row['he']:
        return False
    if (tcs[cur_row['he']-1] - tcs[cur_row['he']-k]) > REBALANCE_MAX_PAUSE:
        return False
    moved = asr_tokens[cur_row['he']-k:cur_row['he']]
    if any(_is_anchor(w) for w in moved):
        return False
    old_sum = cur_row.get('wer', 100.0) + next_row.get('wer', 100.0)
    cur_row['he']  -= k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    new_sum = cur_row['wer'] + next_row['wer']
    if (old_sum - new_sum) >= REBALANCE_MIN_IMPROVE:
        return True
    # revertir
    cur_row['he']  += k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    return False

def _rebalance_rows(rows_meta, ref_tokens, asr_tokens, tcs) -> None:
    # recalcula primero
    for r in rows_meta:
        if not r.get('solo'):
            _recompute_row(r, ref_tokens, asr_tokens, tcs)
    # una pasada sobre pares contiguos
    for i in range(len(rows_meta) - 1):
        a, b = rows_meta[i], rows_meta[i+1]
        if a.get('solo') or b.get('solo'):
            continue
        moved = False
        for k in range(1, REBALANCE_MAX_MOVE+1):
            if _try_move_front(b, a, k, asr_tokens, tcs, ref_tokens):
                moved = True; break
        if moved:
            continue
        for k in range(1, REBALANCE_MAX_MOVE+1):
            if _try_move_back(a, b, k, asr_tokens, tcs, ref_tokens):
                break

# ───────────────────────── API principal ─────────────────────────

def build_rows_from_words(ref: str, csv_words: List[str], csv_tcs: List[float]) -> List[List]:
    """
    Alinea por oraciones usando ventana local sobre el ASR.
    Aplica cortes por pausa/duración y rebalance de 1–3 palabras.
    Devuelve filas con 8 columnas: [ID, ✓, OK, AI, WER, tc, Original, ASR]
    """
    # tokens
    ref_tokens = _tokenize(ref)
    asr_tokens = [_normalize_token(w) for w in csv_words]
    tcs = csv_tcs

    if not ref_tokens:
        return []

    # 1) Ruta “suave” por oraciones → pares (i_ref, j_asr)
    pairs = _align_sentwise(ref_tokens, asr_tokens, tcs)
    map_h = [-1] * len(ref_tokens)
    for ri, hj in pairs:
        # último mapeo gana; estamos en orden creciente (aprox.)
        map_h[ri] = hj

    rows_meta = []
    consumed = set()
    last_h = 0

    # 2) Cortes por oración con corridas/pausas/duración
    for s, e in _sentence_spans(ref_tokens):
        ref_seg = ref_tokens[s:e]
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            runs = _split_runs_by_pause(idx, tcs)
            hs, he = _choose_best_run(runs, len(ref_seg), tcs)

            # “solo-ASR” opcional entre last_h y hs
            if hs > last_h:
                rows_meta.append({
                    'solo': True, 'id': len(rows_meta),
                    's': e, 'e': e, 'hs': last_h, 'he': hs
                })

            consumed.update(range(hs, he))
            row = {
                'solo': False, 'id': len(rows_meta),
                's': s, 'e': e, 'hs': hs, 'he': he
            }
            rows_meta.append(row)
            last_h = he
        else:
            # sin mapeo: fila vacía con tc en last_h
            rows_meta.append({
                'solo': False, 'id': len(rows_meta),
                's': s, 'e': e, 'hs': last_h, 'he': last_h
            })

    # resto del ASR como “solo-ASR” (opcional)
    if last_h < len(asr_tokens):
        rows_meta.append({
            'solo': True, 'id': len(rows_meta),
            's': len(ref_tokens), 'e': len(ref_tokens),
            'hs': last_h, 'he': len(asr_tokens)
        })

    # 3) Materializar textos + WER/flag
    for r in rows_meta:
        if r.get('solo'):
            r['txt_ref'] = ""
            r['txt_asr'] = " ".join(asr_tokens[r['hs']:r['he']])
            r['tc'] = _sec_to_str(tcs[r['hs']]) if r['hs'] < len(tcs) else _sec_to_str(0.0)
            r['wer'] = 100.0 if r['txt_asr'] else 0.0
            r['flag'] = "❌" if r['txt_asr'] else "✅"
        else:
            _recompute_row(r, ref_tokens, asr_tokens, tcs)

    # 4) Rebalance local (mueve ≤3 palabras si baja WER total)
    _rebalance_rows(rows_meta, ref_tokens, asr_tokens, tcs)

    # 5) Construir salida en 8 columnas
    out: List[List] = []
    rid = 0
    for r in rows_meta:
        # columnas: [ID, ✓, OK, AI, WER, tc, Original, ASR]
        out.append([
            rid,
            r['flag'],
            round(r['wer'], 1),  # WER en %
            r['tc'],
            r['txt_ref'],
            r['txt_asr'],
        ])
        rid += 1
    return out

def build_rows(ref: str, hyp: str) -> List[List]:
    """
    Compatibilidad: si llega ASR como TEXTO.
    Creamos tiempos sintéticos y reutilizamos el mismo pipeline “por palabras”.
    """
    hyp_tok = _tokenize(hyp)
    # tiempos sintéticos (palabra ~ 1/WPS s)
    csv_tcs = [i * (1.0 / WPS) for i in range(len(hyp_tok))]
    return build_rows_from_words(ref, hyp_tok, csv_tcs)
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from difflib import SequenceMatcher
from typing import Callable, List, Tuple
from collections import defaultdict
import csv

# alignment.py — versión conservadora y compatible (2025-08-14)
# - Mantiene la interfaz y formato de salida que espera qc_app.py
# - Alineación por oración con ventana local (rápida y estable)
# - “Barandas”:
#     * corta corridas por pausas reales y duración esperada
#     * post-paso de rebalance (mueve ≤3 palabras si baja WER total)
#     * protege anclas (fechas, meses) para no desordenarlas
#
# Salida: lista de filas [ID, flag, "", "", WER(0-100), tc(seg), Original, ASR]


# ───────────────────────── Configuración ─────────────────────────

# Umbral que usa el GUI (Levenshtein.normalized_distance ∈ [0..1])
# ¡Ojo! El GUI espera FRACCIÓN, no porcentaje:
WARN_WER = 0.12  # 12%

# Tokenización y similitud
SIM_THRESH = 0.80  # similitud normalizada mínima para considerar “match” (tokens)
PUNCT_RE = r"""["#$%&'()*+,\-/:;<=>@\[\]^_`{|}~¡¿…—–]"""

# Ventanas / pausas / tiempos “razonables”
BAND_CAP = 64  # límite interno de la “banda” para DP local
MAX_TOKEN_GAP = 12  # rompe corrida si gap de índices > N
PAUSE_SPLIT_SEC = 2.5  # tolera pausas más largas antes de cortar la corrida
WPS = 2.5  # palabras/seg esperadas (≈150 wpm)
MAX_RATIO = 2.5  # límite ancho relativo a len(ref_seg)
MAX_EXTRA = 12  # extra absoluto (tokens) permitido

# Rebalance de bordes
REBALANCE_MAX_MOVE = 3  # mueve hasta 3 palabras
REBALANCE_MIN_IMPROVE = 2.0  # mejora mínima (en puntos de WER)
REBALANCE_MAX_PAUSE = 0.80  # no cruza pausas largas

# limites para anclas globales (nuevo alineador)
MAX_SEGMENT_RATIO = 10.0      # cuantas veces puede ser mas largo un segmento que el otro
MAX_FORWARD_GAP = 400         # maximo de tokens ASR entre anclas hacia adelante
GAP_REF_THRESHOLD = 300       # si avanzo esto en ref sin hallar ancla, salto a busqueda desde el final

MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
    "agosto", "septiembre", "setiembre", "octubre", "noviembre", "diciembre"
}

ROMAN_RE = re.compile(r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")


@dataclass(frozen=True)
class Anchor:
    ref_idx: int
    asr_idx: int
    size: int


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


def _roman_to_int(s: str) -> str | None:
    if not ROMAN_RE.match(s):
        return None
    vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        val = vals[ch]
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return str(total)


def _normalize_token(s: str) -> str:
    s = _strip_accents(s.lower())
    # Preserve sentence-ending punctuation for splitting
    s = re.sub(r'[!"#$%&\'()*+,\-/:;<=>?@\[\]^_`{|}~¡¿…—–]', " ", s)
    s = re.sub(r'\s+', ' ', s).strip()
    roman = _roman_to_int(s)
    if roman:
        return roman
    return s




def _normalize_token_for_align(s: str) -> str:
    """Version sin signos finales para comparar tokens."""
    s = _normalize_token(s)
    return re.sub(r"[.!?]+$", "", s)

def _tokenize(text: str) -> List[str]:
    t = _normalize_token(text)
    return t.split() if t else []


def _sentence_spans(ref_tokens: List[str]) -> List[Tuple[int, int]]:
    text = " ".join(ref_tokens)
    sents = _SENT_SPLIT_RE.split(text) if text else []
    spans: List[Tuple[int, int]] = []
    pos = 0
    max_chunk = 80
    for s in sents:
        w = s.split()
        if w:
            if len(w) > max_chunk:
                for i in range(0, len(w), max_chunk):
                    spans.append((pos + i, pos + min(len(w), i + max_chunk)))
                pos += len(w)
            else:
                spans.append((pos, pos + len(w)))
                pos += len(w)
    if not spans and ref_tokens:
        spans = [(i, min(i + max_chunk, len(ref_tokens))) for i in range(0, len(ref_tokens), max_chunk)]
    return spans


def _paragraph_spans(ref_text: str) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
    """
    Returns reconstructed paragraphs, their token spans, and the flattened token list.
    """
    from text_utils import prepare_paragraphs
    import re

    paragraphs = prepare_paragraphs(ref_text)
    if len(paragraphs) <= 1:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', ref_text.replace("\n", " ")) if s.strip()]
        para_buf: List[str] = []
        cur_tokens = 0
        max_tokens = 120
        new_pars: List[str] = []
        for sent in sentences:
            toks = _tokenize(sent)
            if not toks:
                continue
            if cur_tokens + len(toks) > max_tokens and para_buf:
                new_pars.append(" ".join(para_buf))
                para_buf = []
                cur_tokens = 0
            para_buf.append(sent)
            cur_tokens += len(toks)
        if para_buf:
            new_pars.append(" ".join(para_buf))
        toks_all = _tokenize(ref_text)
        if new_pars and (len(new_pars) > 1 or len(toks_all) <= max_tokens):
            paragraphs = new_pars
        else:
            # Último recurso: dividir en bloques de tokens fijos si no hay puntuación.
            if toks_all:
                paragraphs = [" ".join(toks_all[i:i + max_tokens]) for i in range(0, len(toks_all), max_tokens)]

    spans: List[Tuple[int, int]] = []
    ref_tokens: List[str] = []
    offset = 0
    for para in paragraphs:
        toks = _tokenize(para)
        if not toks:
            continue
        spans.append((offset, offset + len(toks)))
        ref_tokens.extend(toks)
        offset += len(toks)
    if not spans and ref_tokens:
        spans = [(0, len(ref_tokens))]
    _d(f"[paragraphs] count={len(spans)} max_len={max((e-s) for s,e in spans) if spans else 0}")
    return paragraphs, spans, ref_tokens


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
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if ca == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def _wer_pct(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 100.0
    return min(100.0, 100.0 * _lev(ref_tokens, hyp_tokens) / len(ref_tokens))


def _flag_for_wer(wer_pct: float, ref_len: int) -> str:
    if wer_pct == 0.0:
        return "✅"
    thr = 20.0 if ref_len < 5 else (WARN_WER * 100.0)
    if wer_pct <= thr:
        return "✅"
    if wer_pct <= 20.0:
        return "⚠️"
    return "❌"


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
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    return runs


def _choose_best_run(runs: List[List[int]], ref_len: int, tcs: List[float]) -> Tuple[int, int]:
    if not runs:
        return 0, 0

    # Dominante: corrida con más coincidencias (densidad principal)
    best_run = max(runs, key=len)
    final_indices: List[int] = list(best_run)
    best_start, best_end = best_run[0], best_run[-1]

    # Límite razonable de ancho basado en el largo esperado de la oración
    max_span = int(ref_len * MAX_RATIO) + MAX_EXTRA
    MAX_MERGE_SEC = 5.0
    MAX_MERGE_IDX = MAX_TOKEN_GAP

    try:
        best_start_tc = tcs[best_start]
        best_end_tc = tcs[best_end]
    except Exception:
        best_start_tc = best_end_tc = None

    def _near_enough(run: List[int]) -> bool:
        # Prefiere corridas contiguas en índices; usa tiempo solo como respaldo.
        idx_gap = min(abs(run[0] - best_end), abs(best_start - run[-1]))
        if idx_gap <= MAX_MERGE_IDX:
            return True
        if best_start_tc is None or best_end_tc is None:
            return False
        try:
            run_start_tc = tcs[run[0]]
            run_end_tc = tcs[run[-1]]
        except Exception:
            return False
        time_gap = min(abs(run_start_tc - best_end_tc), abs(best_start_tc - run_end_tc))
        return time_gap <= MAX_MERGE_SEC

    for run in runs:
        if run is best_run:
            continue
        if not _near_enough(run):
            continue
        candidate = final_indices + run
        cand_hs, cand_he = min(candidate), max(candidate) + 1
        if (cand_he - cand_hs) <= max_span:
            final_indices.extend(run)

    hs = min(final_indices)
    he = max(final_indices) + 1

    # Si algo salió fuera de rango, conserva solo la corrida dominante.
    if (he - hs) > max_span:
        hs, he = best_start, best_end + 1

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


def _align_paragraphwise(ref_text: str, ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> List[Tuple[int, int]]:
    """
    Devuelve pares (i_ref, j_asr) únicamente en posiciones “diagonales”
    (similares) por ventana local. Se usa para guiar el recorte por párrafo.
    """
    pairs: List[Tuple[int, int]] = []
    last = 0
    _, spans, _ = _paragraph_spans(ref_text)

    for (s, e) in spans:
        seg = ref_tokens[s:e]
        if not seg:
            continue
        # Ventana cerca de 'last': crece con el largo de la oración y un margen
        approx = last
        win = min(len(asr_tokens), approx + int(MAX_RATIO * max(8, (e - s))) + 200)
        lo = approx
        # Cortar por pausa grande cercana para no engordar demasiado la ventana
        # No cortar la ventana aquí; los cortes por pausa se manejan en _split_runs_by_pause

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




def _filter_monotonic_pairs(pairs: List[tuple[int, int]], n_ref: int, n_asr: int) -> List[tuple[int, int]]:
    """Enforce monotonicity and valid ranges on (ref, asr) pairs preserving original order."""
    filtered: List[tuple[int, int]] = []
    last_h = -1
    dropped = 0
    for ri, hj in pairs:
        if not (0 <= ri < n_ref and 0 <= hj < n_asr):
            dropped += 1
            continue
        if hj < last_h:
            dropped += 1
            continue
        filtered.append((ri, hj))
        last_h = hj
    if dropped:
        _d(f"[align_sentwise] dropped {dropped} non-monotonic/out-of-range pairs")
    return filtered
def _align_sentwise(ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> List[Tuple[int, int]]:
    """
    Devuelve pares (i_ref, j_asr) usando ventanas locales guiadas por oraciones y pausas.
    """
    pairs: List[Tuple[int, int]] = []
    last = 0
    spans = _sentence_spans(ref_tokens)

    for (s, e) in spans:
        seg = ref_tokens[s:e]
        if not seg:
            continue
        approx = last
        win = min(len(asr_tokens), approx + int(MAX_RATIO * max(8, (e - s))) + 200)
        lo = approx
        # No recortar aquí por pausas; los cortes se manejan en _split_runs_by_pause

        idxs = _match_sent_window(seg, asr_tokens, lo, win)
        for j in idxs:
            i_ref = s + min(e - s - 1, max(0, j - lo))
            pairs.append((i_ref, j))

        if idxs:
            last = max(last, idxs[-1] + 1)

    return _filter_monotonic_pairs(pairs, len(ref_tokens), len(asr_tokens))


def _build_ngram_anchors(ref_tokens: List[str], asr_tokens: List[str],
                         max_size: int = 5, min_size: int = 2) -> List[Anchor]:
    """
    Construye anclas por n-gramas, priorizando la primera ocurrencia en ASR
    y manteniendo monotonicidad.
    """
    anchors: List[Anchor] = []
    used_ref: set[int] = set()
    used_asr: set[int] = set()
    last_asr = -1
    n_ref, n_asr = len(ref_tokens), len(asr_tokens)
    for size in range(max_size, min_size - 1, -1):
        if n_ref < size or n_asr < size:
            continue
        asr_map: dict[tuple[str, ...], List[int]] = {}
        for j in range(n_asr - size + 1):
            if any((j + off) in used_asr for off in range(size)):
                continue
            key = tuple(asr_tokens[j:j + size])
            asr_map.setdefault(key, []).append(j)

        for i in range(n_ref - size + 1):
            if any((i + off) in used_ref for off in range(size)):
                continue
            key = tuple(ref_tokens[i:i + size])
            if key not in asr_map:
                continue
            candidates = [
                pos for pos in sorted(asr_map[key])
                if pos > last_asr and all((pos + off) not in used_asr for off in range(size))
            ]
            if not candidates:
                continue
            pos = candidates[0]
            anchors.append(Anchor(i, pos, size))
            for off in range(size):
                used_ref.add(i + off)
                used_asr.add(pos + off)
            last_asr = pos

    anchors.sort(key=lambda a: a.ref_idx)
    return anchors


def _enforce_monotonic(anchors: List[Anchor]) -> List[Anchor]:
    """Filtra anclas para que asr_idx sea estrictamente creciente."""
    filtered: List[Anchor] = []
    last_asr = -1
    for a in sorted(anchors, key=lambda x: x.ref_idx):
        if a.asr_idx > last_asr:
            filtered.append(a)
            last_asr = a.asr_idx
    return filtered


def _prune_outlier_anchors(anchors: List[Anchor], max_ratio: float = MAX_RATIO * 2) -> List[Anchor]:
    """Elimina anclas con saltos desproporcionados para evitar colapsos al final."""
    if not anchors:
        return anchors
    pruned: List[Anchor] = []
    prev = Anchor(0, 0, 0)
    for a in anchors:
        dr = max(1, a.ref_idx - prev.ref_idx)
        dh = max(1, a.asr_idx - prev.asr_idx)
        ratio = dh / dr
        if ratio > max_ratio or ratio < (1 / max_ratio):
            continue
        pruned.append(a)
        prev = a
    return pruned if pruned else anchors


def _bidirectional_anchors(ref_tokens: List[str], asr_tokens: List[str],
                           max_size: int = 5, min_size: int = 2) -> List[Anchor]:
    """Combina anclas forward + reverse para cubrir extremos."""
    n_ref, n_asr = len(ref_tokens), len(asr_tokens)
    forward = _build_ngram_anchors(ref_tokens, asr_tokens, max_size, min_size)
    # Reverse pass
    rev_ref = list(reversed(ref_tokens))
    rev_asr = list(reversed(asr_tokens))
    back_raw = _build_ngram_anchors(rev_ref, rev_asr, max_size, min_size)
    back: List[Anchor] = []
    for a in back_raw:
        ref_idx = n_ref - a.ref_idx - a.size
        asr_idx = n_asr - a.asr_idx - a.size
        back.append(Anchor(ref_idx, asr_idx, a.size))

    combined = {(a.ref_idx, a.asr_idx): a for a in forward}
    for a in back:
        combined.setdefault((a.ref_idx, a.asr_idx), a)

    anchors = _enforce_monotonic(list(combined.values()))
    return _prune_outlier_anchors(anchors)


def _build_rows_sentwise(ref_tokens: List[str],
                         asr_tokens_norm: List[str],
                         asr_words: List[str],
                         tcs: List[float],
                         sentence_spans: List[Tuple[int, int]] | None = None) -> List[dict]:
    """
    Alinea por oraciones usando ventanas locales guiadas por tiempos y pausa.
    Devuelve rows_meta con filas normales y solo-ASR/solo-ref cuando corresponde.
    """
    spans = sentence_spans if sentence_spans is not None else _sentence_spans(ref_tokens)
    n_asr = len(asr_words)
    pairs = _align_sentwise(ref_tokens, asr_tokens_norm, tcs)
    map_h = [-1] * len(ref_tokens)
    for ri, hj in pairs:
        if 0 <= hj < n_asr:
            map_h[ri] = hj

    rows_meta: List[dict] = []
    consumed: set[int] = set()
    last_h = 0

    for s, e in spans:
        idx = [map_h[k] for k in range(s, e) if map_h[k] != -1 and map_h[k] not in consumed]
        if idx:
            runs = _split_runs_by_pause(idx, tcs)
            hs, he = _choose_best_run(runs, len(ref_tokens[s:e]), tcs)
            hs = max(0, min(hs, n_asr))
            he = max(hs, min(he, n_asr))

            if hs > last_h:
                rows_meta.append({
                    'solo': True,
                    'solo_ref': False,
                    'id': len(rows_meta),
                    's': e,
                    'e': e,
                    'hs': last_h,
                    'he': hs,
                })

            consumed.update(range(hs, he))
            rows_meta.append({
                'solo': False,
                'solo_ref': False,
                'id': len(rows_meta),
                's': s,
                'e': e,
                'hs': hs,
                'he': he,
            })
            last_h = he
        else:
            hs = he = max(0, min(last_h, n_asr))
            rows_meta.append({
                'solo': False,
                'solo_ref': True,
                'id': len(rows_meta),
                's': s,
                'e': e,
                'hs': hs,
                'he': he,
            })

    if last_h < n_asr:
        rows_meta.append({
            'solo': True,
            'solo_ref': False,
            'id': len(rows_meta),
            's': len(ref_tokens),
            'e': len(ref_tokens),
            'hs': last_h,
            'he': n_asr,
        })

    return rows_meta

def _with_sentinels(anchors: List[Anchor], n_ref: int, n_asr: int) -> List[Anchor]:
    items = list(anchors)
    if not items or items[0].ref_idx > 0:
        items.insert(0, Anchor(0, 0, 0))
    items.append(Anchor(n_ref, n_asr, 0))
    return items


def _map_ref_to_asr(pos: int, anchors: List[Anchor], n_ref: int, n_asr: int) -> int:
    if n_ref <= 0 or n_asr < 0:
        return 0
    chain = _with_sentinels(anchors, n_ref, n_asr)
    prev = chain[0]
    nxt = chain[-1]
    for anc in chain:
        if anc.ref_idx <= pos:
            prev = anc
        if anc.ref_idx >= pos:
            nxt = anc
            break

    if prev.ref_idx <= pos < (prev.ref_idx + prev.size):
        return min(n_asr, prev.asr_idx + (pos - prev.ref_idx))
    if nxt.ref_idx == prev.ref_idx:
        return min(n_asr, max(0, prev.asr_idx))

    span_ref = max(1, nxt.ref_idx - prev.ref_idx)
    frac = (pos - prev.ref_idx) / span_ref
    est = int(round(prev.asr_idx + frac * (nxt.asr_idx - prev.asr_idx)))
    return max(0, min(n_asr, est))


def _bounds_from_anchors(spans: List[Tuple[int, int]], anchors: List[Anchor],
                         n_ref: int, n_asr: int) -> List[Tuple[int, int]]:
    anchors = _enforce_monotonic(anchors)
    bounds: List[Tuple[int, int]] = []
    last_end = 0
    chain = _with_sentinels(anchors, n_ref, n_asr)
    for (s, e) in spans:
        hs = _map_ref_to_asr(s, chain, n_ref, n_asr)
        he = _map_ref_to_asr(e, chain, n_ref, n_asr)
        hs = max(last_end, max(0, min(hs, n_asr)))
        he = max(hs + (1 if n_asr else 0), min(he, n_asr))
        bounds.append((hs, he))
        last_end = he
    if bounds and n_asr:
        _, first_he = bounds[0]
        bounds[0] = (0, max(first_he, 1))
    if bounds and n_asr:
        last_hs, _ = bounds[-1]
        bounds[-1] = (last_hs, n_asr)
    return bounds


# ───────────────────────── Rebalance de fronteras ─────────────────────────



def _span_overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _materialize_rows(rows_meta, ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> None:
    """Completa textos, flags y WER para cada fila."""
    for r in rows_meta:
        if r.get('solo'):
            r['txt_ref'] = ''
            r['txt_asr'] = ' '.join(asr_tokens[r['hs']:r['he']])
            base_tc = tcs[r['hs']] if r['hs'] < len(tcs) else (tcs[-1] if tcs else 0.0)
            r['tc'] = _sec_to_str(base_tc)
            r['wer'] = 100.0 if r['txt_asr'] else 0.0
            r['flag'] = "❌" if r['txt_asr'] else "✅"
        elif r.get('solo_ref'):
            r['txt_ref'] = " ".join(ref_tokens[r['s']:r['e']])
            r['txt_asr'] = ''
            base_tc = tcs[r['hs']] if r['hs'] < len(tcs) else (tcs[-1] if tcs else 0.0)
            r['tc'] = _sec_to_str(base_tc)
            r['wer'] = 100.0
            r['flag'] = "❌"
        else:
            _recompute_row(r, ref_tokens, asr_tokens, tcs)


def _recompute_row(row, ref_tokens: List[str], asr_tokens: List[str], tcs: List[float]) -> None:
    seg_ref = ref_tokens[row['s']:row['e']]
    seg_asr = asr_tokens[row['hs']:row['he']]
    wer = _wer_pct(seg_ref, [_normalize_token(w) for w in seg_asr])
    row['wer'] = wer
    row['flag'] = _flag_for_wer(wer, len(seg_ref))
    row['txt_ref'] = " ".join(seg_ref)
    row['txt_asr'] = " ".join(asr_tokens[row['hs']:row['he']])
    row['tc'] = _sec_to_str(tcs[row['hs']]) if row['hs'] < len(tcs) else _sec_to_str(0.0)


def _try_move_front(next_row, cur_row, k, asr_tokens, tcs, ref_tokens) -> bool:
    if k <= 0 or next_row['hs'] + k >= next_row['he']:
        return False
    if (tcs[next_row['hs'] + k - 1] - tcs[next_row['hs']]) > REBALANCE_MAX_PAUSE:
        return False
    moved = asr_tokens[next_row['hs']:next_row['hs'] + k]
    if any(_is_anchor(w) for w in moved):
        return False
    old_sum = cur_row.get('wer', 100.0) + next_row.get('wer', 100.0)
    cur_row['he'] += k
    next_row['hs'] += k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    new_sum = cur_row['wer'] + next_row['wer']
    if (old_sum - new_sum) >= REBALANCE_MIN_IMPROVE:
        return True
    # revertir
    cur_row['he'] -= k
    next_row['hs'] -= k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    return False


def _try_move_back(cur_row, next_row, k, asr_tokens, tcs, ref_tokens) -> bool:
    if k <= 0 or cur_row['hs'] + k >= cur_row['he']:
        return False
    if (tcs[cur_row['he'] - 1] - tcs[cur_row['he'] - k]) > REBALANCE_MAX_PAUSE:
        return False
    moved = asr_tokens[cur_row['he'] - k:cur_row['he']]
    if any(_is_anchor(w) for w in moved):
        return False
    old_sum = cur_row.get('wer', 100.0) + next_row.get('wer', 100.0)
    cur_row['he'] -= k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    new_sum = cur_row['wer'] + next_row['wer']
    if (old_sum - new_sum) >= REBALANCE_MIN_IMPROVE:
        return True
    # revertir
    cur_row['he'] += k
    _recompute_row(cur_row, ref_tokens, asr_tokens, tcs)
    _recompute_row(next_row, ref_tokens, asr_tokens, tcs)
    return False


def _rebalance_rows(rows_meta, ref_tokens, asr_tokens, tcs) -> None:
    # recalcula primero
    for r in rows_meta:
        if not r.get('solo') and not r.get('solo_ref'):
            _recompute_row(r, ref_tokens, asr_tokens, tcs)
    # una pasada sobre pares contiguos
    for i in range(len(rows_meta) - 1):
        a, b = rows_meta[i], rows_meta[i + 1]
        if a.get('solo') or b.get('solo') or a.get('solo_ref') or b.get('solo_ref'):
            continue
        moved = False
        for k in range(1, REBALANCE_MAX_MOVE + 1):
            if _try_move_front(b, a, k, asr_tokens, tcs, ref_tokens):
                moved = True
                break
        if moved:
            continue
        for k in range(1, REBALANCE_MAX_MOVE + 1):
            if _try_move_back(a, b, k, asr_tokens, tcs, ref_tokens):
                break




def _word_alignment_ops(ref_seg: List[str], asr_seg: List[str]) -> List[tuple[str, int | None, int | None]]:
    """Levenshtein path for two segments returning operations."""
    m, n = len(ref_seg), len(asr_seg)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = 'del'
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = 'ins'
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_seg[i - 1] == asr_seg[j - 1] else 1
            choices = [
                (dp[i - 1][j] + 1, 'del'),
                (dp[i][j - 1] + 1, 'ins'),
                (dp[i - 1][j - 1] + cost, 'match' if cost == 0 else 'sub'),
            ]
            dp[i][j], back[i][j] = min(choices, key=lambda x: x[0])
    ops: List[tuple[str, int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        action = back[i][j]
        if action == 'del':
            ops.append(('del', i - 1, None))
            i -= 1
        elif action == 'ins':
            ops.append(('ins', None, j - 1))
            j -= 1
        else:
            tipo = 'match' if ref_seg[i - 1] == asr_seg[j - 1] else 'sub'
            ops.append((tipo, i - 1, j - 1))
            i -= 1
            j -= 1
    ops.reverse()
    return ops


def _collect_word_alignments(rows_meta: List[dict], ref_tokens: List[str], asr_tokens_norm: List[str]) -> List[dict]:
    alignments: List[dict] = []
    for row_id, row in enumerate(rows_meta):
        if row.get('solo'):
            for aj in range(row['hs'], row['he']):
                alignments.append({'ref_idx': -1, 'asr_idx': aj, 'tipo': 'ins', 'distancia': 1.0, 'row_id': row_id})
            continue
        ref_seg = ref_tokens[row['s']:row['e']]
        asr_seg = asr_tokens_norm[row['hs']:row['he']]
        if row.get('solo_ref'):
            for off, _ in enumerate(ref_seg):
                alignments.append({'ref_idx': row['s'] + off, 'asr_idx': -1, 'tipo': 'del', 'distancia': 1.0, 'row_id': row_id})
            continue
        for action, i_ref, j_asr in _word_alignment_ops(ref_seg, asr_seg):
            ref_idx = row['s'] + i_ref if i_ref is not None else -1
            asr_idx = row['hs'] + j_asr if j_asr is not None else -1
            dist = 0.0 if action == 'match' else 1.0
            alignments.append({'ref_idx': ref_idx, 'asr_idx': asr_idx, 'tipo': action, 'distancia': dist, 'row_id': row_id})
    return alignments

# ------------------------- NUEVA LOGICA: ALINEACION POR ANCLAS ------------------

@dataclass(frozen=True)
class WordOp:
    ref_idx: int | None
    asr_idx: int | None
    op: str  # 'match' | 'sub' | 'ins' | 'del'


def _tokenize_dual(text: str) -> tuple[list[str], list[str]]:
    """Devuelve (raw_tokens, norm_tokens) para un texto ya segmentado en parrafos."""
    raw_tokens: list[str] = re.findall(r"[A-Za-zÁÉÍÓÚáéíóúÜüÑñ0-9]+|[.!?]", text)
    norm_tokens: list[str] = [_normalize_token_for_align(tok) for tok in raw_tokens if _normalize_token_for_align(tok)]
    # Alinear largo: si algun token normalizado queda vacio, lo omitimos en ambas vistas
    filtered_raw: list[str] = []
    filtered_norm: list[str] = []
    for raw in raw_tokens:
        norm = _normalize_token_for_align(raw)
        if not norm:
            continue
        filtered_raw.append(raw)
        filtered_norm.append(norm)
    return filtered_raw, filtered_norm


def _prepare_ref_tokens(ref_text: str) -> tuple[list[str], list[str], list[int], list[tuple[int, int]], list[str]]:
    """Construye listas de tokens raw/norm de ref, ids de parrafo y spans por parrafo."""
    from text_utils import prepare_paragraphs
    paragraphs = prepare_paragraphs(ref_text)
    ref_raw: list[str] = []
    ref_norm: list[str] = []
    paragraph_ids: list[int] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    for pid, para in enumerate(paragraphs):
        raw_tokens, norm_tokens = _tokenize_dual(para)
        if not norm_tokens:
            continue
        ref_raw.extend(raw_tokens)
        ref_norm.extend(norm_tokens)
        paragraph_ids.extend([pid] * len(norm_tokens))
        spans.append((offset, offset + len(norm_tokens)))
        offset += len(norm_tokens)
    if not spans and ref_norm:
        spans = [(0, len(ref_norm))]
        paragraph_ids = [0] * len(ref_norm)
    return ref_raw, ref_norm, paragraph_ids, spans, paragraphs


def _prepare_asr_tokens(asr_words: list[str]) -> tuple[list[str], list[str]]:
    raw_tokens: list[str] = []
    norm_tokens: list[str] = []
    for w in asr_words:
        norm = _normalize_token_for_align(w)
        if not norm:
            continue
        raw_tokens.append(w)
        norm_tokens.append(norm)
    return raw_tokens, norm_tokens


# ---------- Nuevo buscador de anclas (heredado de LEGACY/EXTRA_UTILS/ALINEACIÓN.py) ----------

def _build_index_anchor(tokens_norm: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    index: dict[tuple[str, ...], list[int]] = defaultdict(list)
    limit = len(tokens_norm) - n + 1
    for j in range(limit):
        key = tuple(tokens_norm[j:j + n])
        index[key].append(j)
    return index


def _find_anchors_forward(ref_norm: list[str], asr_norm: list[str]) -> tuple[list[Anchor], int, int]:
    anchors: list[Anchor] = []
    NR = len(ref_norm)
    NA = len(asr_norm)

    last_ref = 0
    last_asr = 0
    first_anchor = True
    ref_since_last = 0

    for n in (5, 4, 3):
        if NR < n or NA < n:
            continue
        indexA = _build_index_anchor(asr_norm, n)
        i = last_ref
        while i <= NR - n:
            key = tuple(ref_norm[i:i + n])
            positions = indexA.get(key)
            if not positions:
                i += 1
                ref_since_last += 1
                if not first_anchor and ref_since_last > GAP_REF_THRESHOLD:
                    return anchors, last_ref, last_asr
                continue

            candidate_j = None
            for j in positions:
                if j < last_asr:
                    continue

                if first_anchor:
                    candidate_j = j
                    break

                if (j - last_asr) > MAX_FORWARD_GAP:
                    break

                seg_ref = i - last_ref if i > last_ref else 1
                seg_asr = j - last_asr if j > last_asr else 1
                ratio = seg_asr / max(seg_ref, 1)

                if 1.0 / MAX_SEGMENT_RATIO <= ratio <= MAX_SEGMENT_RATIO:
                    candidate_j = j
                    break

            if candidate_j is None:
                i += 1
                ref_since_last += 1
                if not first_anchor and ref_since_last > GAP_REF_THRESHOLD:
                    return anchors, last_ref, last_asr
                continue

            anchors.append(Anchor(i, candidate_j, n))
            first_anchor = False
            last_ref = i + n
            last_asr = candidate_j + n
            ref_since_last = 0
            i = last_ref

    return anchors, last_ref, last_asr


def _find_anchors_backward(ref_norm: list[str], asr_norm: list[str],
                           ref_limit: int, asr_limit: int) -> list[Anchor]:
    ref_suffix = ref_norm[:ref_limit][::-1]
    asr_suffix = asr_norm[:asr_limit][::-1]

    back_anchors, _, _ = _find_anchors_forward(ref_suffix, asr_suffix)

    anchors: list[Anchor] = []
    for a in back_anchors:
        orig_ref = ref_limit - (a.ref_idx + a.size)
        orig_asr = asr_limit - (a.asr_idx + a.size)
        anchors.append(Anchor(orig_ref, orig_asr, a.size))

    anchors.sort(key=lambda x: x.ref_idx)
    return anchors


def _find_anchors_bidirectional(ref_norm: list[str], asr_norm: list[str]) -> list[Anchor]:
    NR = len(ref_norm)
    NA = len(asr_norm)

    anchors_forward, last_ref, last_asr = _find_anchors_forward(ref_norm, asr_norm)

    if last_ref >= NR or last_asr >= NA:
        return anchors_forward

    anchors_backward = _find_anchors_backward(ref_norm, asr_norm, NR, NA)

    all_anchors = anchors_forward + anchors_backward
    all_anchors.sort(key=lambda a: (a.ref_idx, a.asr_idx))

    filtered: list[Anchor] = []
    last_r_end = -1
    last_a_end = -1
    for a in all_anchors:
        if a.ref_idx >= last_r_end and a.asr_idx >= last_a_end:
            filtered.append(a)
            last_r_end = a.ref_idx + a.size
            last_a_end = a.asr_idx + a.size

    return filtered


def _index_ngrams(tokens: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    idx: dict[tuple[str, ...], list[int]] = {}
    for j in range(0, len(tokens) - n + 1):
        key = tuple(tokens[j:j + n])
        idx.setdefault(key, []).append(j)
    return idx


def _extract_global_anchors(ref_tokens: list[str], asr_tokens: list[str]) -> list[Anchor]:
    return _find_anchors_bidirectional(ref_tokens, asr_tokens)


def _dp_align_segment(ref_norm: list[str], asr_norm: list[str],
                      r_start: int, r_end: int,
                      a_start: int, a_end: int) -> list[tuple[int | None, int | None]]:
    """
    Alinea ref_norm[r_start:r_end] con asr_norm[a_start:a_end] por DP.
    Devuelve lista de pares (ref_idx | None, asr_idx | None) en orden.
    """
    m = r_end - r_start
    n = a_end - a_start
    if m < 0 or n < 0:
        raise ValueError(f"Segmento negativo: ref {r_start}-{r_end}, asr {a_start}-{a_end}")

    if m == 0 and n == 0:
        return []
    if m == 0:
        return [(None, a_start + j) for j in range(n)]
    if n == 0:
        return [(r_start + i, None) for i in range(m)]

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            r_tok = ref_norm[r_start + i - 1]
            a_tok = asr_norm[a_start + j - 1]
            cost_match = 0 if r_tok == a_tok else 1

            best = dp[i - 1][j - 1] + cost_match
            op = "M" if cost_match == 0 else "S"

            cand = dp[i - 1][j] + 1
            if cand < best:
                best = cand
                op = "D"

            cand = dp[i][j - 1] + 1
            if cand < best:
                best = cand
                op = "I"

            dp[i][j] = best
            back[i][j] = op

    pairs: list[tuple[int | None, int | None]] = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            pairs.append((r_start + i - 1, a_start + j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            pairs.append((r_start + i - 1, None))
            i -= 1
        elif op == "I":
            pairs.append((None, a_start + j - 1))
            j -= 1
        else:
            raise RuntimeError("Backtracking inconsistente")

    pairs.reverse()
    return pairs


def _align_pairs_with_anchors(ref_norm: list[str], asr_norm: list[str], anchors: list[Anchor]) -> list[tuple[int | None, int | None, bool]]:
    """
    Devuelve lista de triples (ref_idx | None, asr_idx | None, is_anchor).
    """
    aligned: list[tuple[int | None, int | None, bool]] = []

    NR = len(ref_norm)
    NA = len(asr_norm)
    anchors = sorted(anchors, key=lambda a: a.ref_idx)

    if not anchors:
        for ref_idx, asr_idx in _dp_align_segment(ref_norm, asr_norm, 0, NR, 0, NA):
            aligned.append((ref_idx, asr_idx, False))
        return aligned

    prev_r = 0
    prev_a = 0

    for anchor in anchors:
        aligned_seg = _dp_align_segment(
            ref_norm, asr_norm,
            prev_r, anchor.ref_idx,
            prev_a, anchor.asr_idx,
        )
        aligned.extend((ri, aj, False) for (ri, aj) in aligned_seg)

        for k in range(anchor.size):
            aligned.append((anchor.ref_idx + k, anchor.asr_idx + k, True))

        prev_r = anchor.ref_idx + anchor.size
        prev_a = anchor.asr_idx + anchor.size

    aligned_seg = _dp_align_segment(
        ref_norm, asr_norm,
        prev_r, NR,
        prev_a, NA,
    )
    aligned.extend((ri, aj, False) for (ri, aj) in aligned_seg)

    return aligned


def _align_segment(ref_seg: list[str], asr_seg: list[str],
                   ref_offset: int, asr_offset: int) -> list[WordOp]:
    """Aplica DP clasico sobre un segmento y devuelve operaciones globales."""
    ops: list[WordOp] = []
    if ref_seg and asr_seg:
        for action, i_ref, j_asr in _word_alignment_ops(ref_seg, asr_seg):
            gi = ref_offset + i_ref if i_ref is not None else None
            gj = asr_offset + j_asr if j_asr is not None else None
            ops.append(WordOp(gi, gj, action))
    elif ref_seg and not asr_seg:
        for k in range(len(ref_seg)):
            ops.append(WordOp(ref_offset + k, None, 'del'))
    elif asr_seg and not ref_seg:
        for k in range(len(asr_seg)):
            ops.append(WordOp(None, asr_offset + k, 'ins'))
    return ops


def _build_word_alignment(ref_tokens: list[str], asr_tokens: list[str],
                           anchors: list[Anchor], pairs: list[tuple[int | None, int | None, bool]] | None = None) -> list[WordOp]:
    if pairs is None:
        pairs = _align_pairs_with_anchors(ref_tokens, asr_tokens, anchors)
    ops: list[WordOp] = []
    for ref_idx, asr_idx, _is_anchor in pairs:
        if ref_idx is not None and asr_idx is not None:
            op = 'match' if ref_tokens[ref_idx] == asr_tokens[asr_idx] else 'sub'
        elif ref_idx is None:
            op = 'ins'
        else:
            op = 'del'
        ops.append(WordOp(ref_idx, asr_idx, op))
    return ops


def _flag_label(wer: float) -> str:
    if wer == 0.0:
        return "OK"
    if wer <= WARN_WER * 100.0:
        return "WARN"
    if wer < 100.0:
        return "BAD"
    return "BAD"


def _build_paragraph_rows(paragraph_spans: list[tuple[int, int]], paragraph_ids: list[int],
                          word_ops: list[WordOp], ref_tokens_norm: list[str],
                          ref_tokens_raw: list[str], asr_tokens_raw: list[str],
                          asr_tokens_norm: list[str], tcs: list[float]) -> list[dict]:
    rows: list[dict] = []
    ops_by_par: dict[int, list[WordOp]] = {}
    for op in word_ops:
        if op.ref_idx is None:
            continue
        pid = paragraph_ids[op.ref_idx] if op.ref_idx < len(paragraph_ids) else -1
        ops_by_par.setdefault(pid, []).append(op)

    last_asr = 0
    for pid, (ps, pe) in enumerate(paragraph_spans):
        ops = ops_by_par.get(pid, [])
        asr_idxs = [op.asr_idx for op in ops if op.asr_idx is not None]
        if asr_idxs:
            hs = min(asr_idxs)
            he = max(asr_idxs) + 1
            if hs < last_asr:
                hs = last_asr
            if he < hs:
                he = hs
            ref_slice = ref_tokens_norm[ps:pe]
            asr_slice = asr_tokens_norm[hs:he]
            wer = _wer_pct(ref_slice, asr_slice)
            flag = _flag_label(wer)
            solo_ref = False
        else:
            hs = he = last_asr
            wer = 100.0
            flag = "SOLO_REF"
            solo_ref = True
        rows.append({
            'id': len(rows),
            'paragraph_id': pid,
            's': ps, 'e': pe,
            'hs': hs, 'he': he,
            'wer': wer,
            'flag': flag,
            'solo_ref': solo_ref,
            'solo': False,
            'txt_ref': " ".join(ref_tokens_raw[ps:pe]),
            'txt_asr': " ".join(asr_tokens_raw[hs:he]) if he > hs else "",
            'tc': _sec_to_str(tcs[hs]) if hs < len(tcs) else _sec_to_str(0.0),
        })
        last_asr = he
    return rows


def _write_alignment_db(db_path: str | Path,
                        ref_tokens_raw: list[str],
                        ref_tokens_norm: list[str],
                        ref_paragraph_ids: list[int],
                        asr_tokens_raw: list[str],
                        asr_tokens_norm: list[str],
                        tcs: list[float],
                        word_ops: list[WordOp],
                        paragraph_rows: list[dict],
                        anchors: list[Anchor] | None = None) -> None:
    import sqlite3
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript("""
    DROP TABLE IF EXISTS meta;
    DROP TABLE IF EXISTS anchors;
    DROP TABLE IF EXISTS ref_tokens;
    DROP TABLE IF EXISTS asr_tokens;
    DROP TABLE IF EXISTS word_alignment;
    DROP TABLE IF EXISTS paragraph_rows;
    DROP TABLE IF EXISTS paragraphs;
    DROP TABLE IF EXISTS alignments_word;
    DROP TABLE IF EXISTS asr_suspicions;
    """)
    cur.execute("""CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT
    )""")
    cur.execute("""CREATE TABLE anchors (
        ref_idx INTEGER,
        asr_idx INTEGER,
        size INTEGER
    )""")
    cur.execute("""CREATE TABLE ref_tokens (
        idx INTEGER PRIMARY KEY,
        token_norm TEXT NOT NULL,
        token_raw TEXT NOT NULL,
        paragraph_id INTEGER NOT NULL
    )""")
    cur.execute("""CREATE TABLE asr_tokens (
        idx INTEGER PRIMARY KEY,
        token_norm TEXT NOT NULL,
        token_raw TEXT NOT NULL,
        tc REAL,
        suspected_of TEXT,
        suspicion_score REAL,
        corrected_raw TEXT
    )""")
    cur.execute("""CREATE TABLE word_alignment (
        ref_idx INTEGER,
        asr_idx INTEGER,
        op TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE alignments_word (
        ref_idx INTEGER,
        asr_idx INTEGER,
        tipo TEXT,
        distancia REAL,
        row_id INTEGER,
        PRIMARY KEY (ref_idx, asr_idx)
    )""")
    cur.execute("""CREATE TABLE paragraph_rows (
        row_id INTEGER PRIMARY KEY,
        paragraph_id INTEGER NOT NULL,
        ref_start INTEGER NOT NULL,
        ref_end INTEGER NOT NULL,
        asr_start INTEGER NOT NULL,
        asr_end INTEGER NOT NULL,
        wer REAL NOT NULL,
        flag TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE paragraphs (
        id INTEGER,
        ref_start INTEGER,
        ref_end INTEGER,
        asr_start INTEGER,
        asr_end INTEGER,
        tc_start REAL,
        tc_end REAL,
        ref_text TEXT,
        asr_text TEXT,
        wer REAL,
        flag TEXT
    )""")
    cur.execute("""CREATE TABLE asr_suspicions (
        suspicion_id INTEGER PRIMARY KEY,
        asr_start_idx INTEGER NOT NULL,
        asr_end_idx INTEGER NOT NULL,
        candidate_ref TEXT NOT NULL,
        source TEXT NOT NULL,
        score REAL NOT NULL,
        status TEXT NOT NULL
    )""")

    if anchors:
        cur.executemany(
            "INSERT INTO anchors(ref_idx, asr_idx, size) VALUES (?, ?, ?)",
            [(a.ref_idx, a.asr_idx, a.size) for a in anchors],
        )

    cur.executemany(
        "INSERT INTO ref_tokens(idx, token_norm, token_raw, paragraph_id) VALUES (?, ?, ?, ?)",
        [(i, norm, ref_tokens_raw[i], ref_paragraph_ids[i]) for i, norm in enumerate(ref_tokens_norm)]
    )
    cur.executemany(
        "INSERT INTO asr_tokens(idx, token_norm, token_raw, tc, suspected_of, suspicion_score, corrected_raw) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                i,
                asr_tokens_norm[i],
                asr_tokens_raw[i],
                float(tcs[i]) if i < len(tcs) else None,
                None,
                None,
                None,
            )
            for i in range(len(asr_tokens_norm))
        ]
    )
    cur.executemany(
        "INSERT INTO word_alignment(ref_idx, asr_idx, op) VALUES (?, ?, ?)",
        [(op.ref_idx, op.asr_idx, op.op) for op in word_ops]
    )

    alignments_word = _collect_word_alignments(paragraph_rows, ref_tokens_norm, asr_tokens_norm)
    if alignments_word:
        cur.executemany(
            "INSERT OR REPLACE INTO alignments_word(ref_idx, asr_idx, tipo, distancia, row_id) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    a.get('ref_idx', -1) if a.get('ref_idx') is not None else -1,
                    a.get('asr_idx', -1) if a.get('asr_idx') is not None else -1,
                    a.get('tipo', ''),
                    float(a.get('distancia', 0.0)),
                    a.get('row_id'),
                )
                for a in alignments_word
            ],
        )

    cur.executemany(
        "INSERT INTO paragraph_rows(row_id, paragraph_id, ref_start, ref_end, asr_start, asr_end, wer, flag) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (row['id'], row['paragraph_id'], row['s'], row['e'], row['hs'], row['he'], row['wer'], row['flag'])
            for row in paragraph_rows
        ]
    )
    for row in paragraph_rows:
        start_tc = float(tcs[row['hs']]) if row['hs'] < len(tcs) else 0.0
        end_idx = max(row['he'] - 1, row['hs'])
        end_tc = float(tcs[end_idx]) if end_idx < len(tcs) else start_tc
        cur.execute(
            "INSERT INTO paragraphs(id, ref_start, ref_end, asr_start, asr_end, tc_start, tc_end, ref_text, asr_text, wer, flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.get('id'),
                row.get('s'),
                row.get('e'),
                row.get('hs'),
                row.get('he'),
                start_tc,
                end_tc,
                row.get('txt_ref', ''),
                row.get('txt_asr', ''),
                row.get('wer', 0.0),
                row.get('flag', ''),
            )
        )

    cur.executemany(
        "INSERT INTO meta(key, value) VALUES (?, ?)",
        [
            ("paragraphs", str(len(paragraph_rows))),
            ("ref_tokens", str(len(ref_tokens_norm))),
            ("asr_tokens", str(len(asr_tokens_norm))),
            ("alignments_word", str(len(alignments_word))),
            ("anchors", str(len(anchors or []))),
        ],
    )
    conn.commit()
    conn.close()


def _write_alignment_csv(csv_path: str | Path,
                         ref_tokens_raw: list[str],
                         asr_tokens_raw: list[str],
                         pairs: list[tuple[int | None, int | None, bool]]) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_word", "asr_word", "is_anchor"])
        for ref_idx, asr_idx, is_anchor in pairs:
            ref_word = ref_tokens_raw[ref_idx] if ref_idx is not None and ref_idx < len(ref_tokens_raw) else ""
            asr_word = asr_tokens_raw[asr_idx] if asr_idx is not None and asr_idx < len(asr_tokens_raw) else ""
            writer.writerow([ref_word, asr_word, "1" if is_anchor else "0"])


def _verify_invariants(ref_len: int, asr_len: int,
                       word_ops: list[WordOp], paragraph_rows: list[dict]) -> None:
    assert all(op.ref_idx is None or 0 <= op.ref_idx < ref_len for op in word_ops), "ref_idx fuera de rango"
    assert all(op.asr_idx is None or 0 <= op.asr_idx < asr_len for op in word_ops), "asr_idx fuera de rango"

    ref_seen = [False] * ref_len
    asr_seen = [False] * asr_len
    for op in word_ops:
        if op.ref_idx is not None:
            ref_seen[op.ref_idx] = True
        if op.asr_idx is not None:
            asr_seen[op.asr_idx] = True
    assert all(ref_seen), "No todos los tokens de ref quedaron alineados"
    assert all(asr_seen), "No todos los tokens de ASR quedaron alineados"

    last_hs = 0
    for row in paragraph_rows:
        assert 0 <= row['s'] < row['e'] <= ref_len, "span de ref inválido"
        assert 0 <= row['hs'] <= row['he'] <= asr_len, "span de ASR inválido"
        assert row['hs'] >= last_hs, "ASR no monótono en filas"
        last_hs = row['he']


def build_rows_from_words(ref: str, asr_words: list[str], tcs: list[float],
                          markdown_output: str | None = None,
                          debug_db_path: str | None = None,
                          debug_csv_path: str | None = None) -> list[list]:

    from text_utils import paragraphs_to_markdown
    ref_tokens_raw, ref_tokens_norm, ref_paragraph_ids, paragraph_spans, paragraphs = _prepare_ref_tokens(ref)
    if markdown_output:
        try:
            Path(markdown_output).write_text(paragraphs_to_markdown(paragraphs), encoding='utf-8')
        except Exception:
            pass

    asr_tokens_raw, asr_tokens_norm = _prepare_asr_tokens(asr_words)
    tcs = list(tcs)
    if len(tcs) < len(asr_tokens_norm):
        tcs += [tcs[-1] if tcs else 0.0] * (len(asr_tokens_norm) - len(tcs))

    anchors = _extract_global_anchors(ref_tokens_norm, asr_tokens_norm)
    pairs = _align_pairs_with_anchors(ref_tokens_norm, asr_tokens_norm, anchors)
    word_ops = _build_word_alignment(ref_tokens_norm, asr_tokens_norm, anchors, pairs=pairs)
    paragraph_rows = _build_paragraph_rows(paragraph_spans, ref_paragraph_ids, word_ops,
                                           ref_tokens_norm, ref_tokens_raw,
                                           asr_tokens_raw, asr_tokens_norm, tcs)
    _verify_invariants(len(ref_tokens_norm), len(asr_tokens_norm), word_ops, paragraph_rows)

    if debug_db_path:
        try:
            _write_alignment_db(debug_db_path, ref_tokens_raw, ref_tokens_norm, ref_paragraph_ids,
                                asr_tokens_raw, asr_tokens_norm, tcs, word_ops, paragraph_rows, anchors)
        except Exception as exc:  # pragma: no cover
            _d(f"[debug-db] fallo al escribir {debug_db_path}: {exc}")

    if debug_csv_path:
        try:
            _write_alignment_csv(debug_csv_path, ref_tokens_raw, asr_tokens_raw, pairs)
        except Exception as exc:  # pragma: no cover
            _d(f"[debug-csv] fallo al escribir {debug_csv_path}: {exc}")

    out: list[list] = []
    # Contrato canónico: [ID, Check, OK, AI, WER, tc, Original, ASR]
    # (la GUI inserta Score al vuelo; no se guarda en JSON)
    for r in paragraph_rows:
        out.append([
            r['id'],
            r['flag'],
            '',
            '',
            round(r['wer'], 1),
            r['tc'],
            r['txt_ref'],
            r['txt_asr']
        ])

    return out


def build_rows(ref: str, hyp: str,
               markdown_output: str | None = None,
               debug_db_path: str | None = None,
               debug_csv_path: str | None = None) -> list[list]:
    """
    Compatibilidad: si llega ASR como TEXTO.
    Creamos tiempos sinteticos y reutilizamos el mismo pipeline por palabras.
    """
    hyp_tok = _tokenize(hyp)
    csv_tcs = [i * (1.0 / WPS) for i in range(len(hyp_tok))]
    return build_rows_from_words(ref, hyp_tok, csv_tcs,
                                 markdown_output=markdown_output,
                                 debug_db_path=debug_db_path,
                                 debug_csv_path=debug_csv_path)

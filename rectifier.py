# rectifier_r2.py — R2: Rectificador de bordes y cobertura (segunda pasada)
# Objetivo:
#   - Cobertura 100% del ASR: toda palabra (csv_words) se asigna a exactamente 1 fila, en orden.
#   - No “romper” la alineación previa: sólo microajustes locales y corrección de gaps/solapes.
#   - Mantener formato de 6 columnas: [id, flag, wer, tc, original, asr]
#
# Uso:
#   from rectifier_r2 import rectify_rows
#   rows = rectify_rows(rows, csv_words, csv_tcs)

from __future__ import annotations
import unicodedata, re
from typing import List, Tuple

# ----------------------------- Configs suaves -----------------------------
SIM_THRESH = 0.82            # similitud mínima para considerar match de token
LOOKAHEAD = 140              # cuántas palabras adelante buscar el “inicio” de la próxima fila
PAUSE_SPLIT_SEC = 0.80       # pausa grande
MAX_MICRO_MOVE = 3           # mover ≤ 3 tokens en el microajuste de frontera
MIN_WER_IMPROVE = 2.0        # bajar ≥ 2 puntos de WER para aceptar un movimiento

MONTHS = {
    "enero","febrero","marzo","abril","mayo","junio","julio",
    "agosto","septiembre","setiembre","octubre","noviembre","diciembre"
}

# ----------------------------- Utils básicas -----------------------------
_PUNCT = r"""[!"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~¡¿…—–]"""

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")

def _norm_token(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(_PUNCT, " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tok(text: str) -> List[str]:
    t = _norm_token(text)
    return t.split() if t else []

def _lev(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    dp = list(range(lb+1))
    for i in range(1, la+1):
        prev = dp[0]; dp[0] = i
        ca = a[i-1]
        for j in range(1, lb+1):
            cur = dp[j]
            cost = 0 if ca == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[lb]

def _sim(a: str, b: str) -> float:
    if not a and not b: return 1.0
    m = max(len(a), len(b))
    return 1.0 - (_lev(a, b) / m)

def _tok_sim(a: str, b: str) -> bool:
    return _sim(a, b) >= SIM_THRESH

def _wer_pct(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 100.0
    # WER clásico a nivel token
    lr, lh = len(ref_tokens), len(hyp_tokens)
    dp = list(range(lh+1))
    for i in range(1, lr+1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lh+1):
            cur = dp[j]
            cost = 0 if ref_tokens[i-1] == hyp_tokens[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return min(100.0, 100.0 * dp[lh] / max(1, lr))

def _is_anchor(tok: str) -> bool:
    return tok in MONTHS or re.fullmatch(r"\d{1,2}|\d{4}", tok) is not None

# ----------------------------- Núcleo R2 -----------------------------
def _find_next_prefix(asr_tokens: List[str], pattern: List[str], start: int, tcs: List[float]) -> int | None:
    """
    Busca el primer índice >= start donde el comienzo de 'pattern' encaja
    razonablemente (fuzzy). Se prefieren posiciones justo después de una pausa.
    Devuelve j (inicio) o None si no hay buena coincidencia en la ventana.
    """
    if not pattern:
        return None
    end = min(len(asr_tokens), start + LOOKAHEAD)
    best = None
    best_score = 0.0

    plen = min(4, len(pattern))   # usá hasta 4 tokens del inicio
    pat = pattern[:plen]

    for j in range(start, end):
        # score: coincidencias en paralelo de 0..plen-1
        matches = 0
        for k in range(plen):
            if j+k >= len(asr_tokens): break
            if _tok_sim(asr_tokens[j+k], pat[k]):
                matches += 1
        score = matches / max(1, plen)

        # preferir tras una pausa clara
        if j > 0 and j < len(tcs) and (tcs[j] - tcs[j-1]) >= PAUSE_SPLIT_SEC:
            score += 0.15

        if score >= 0.60 and (best is None or score > best_score):
            best, best_score = j, score
            if score >= 0.95:
                break

    return best

def _contiguity_pass(bounds: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Fuerza he[i] == hs[i+1] (sin perder orden)."""
    if not bounds: return bounds
    fixed = [list(b) for b in bounds]
    for i in range(len(fixed)-1):
        hs, he = fixed[i]
        nhs, nhe = fixed[i+1]
        # si hay hueco -> pegamos a la izquierda
        if nhs > he:
            fixed[i][1] = nhs
        # si hay solape -> cortamos en el medio
        elif nhs < he:
            mid = (nhs + he) // 2
            fixed[i][1] = mid
            fixed[i+1][0] = mid
        # ya contiguo
    return [(a,b) for a,b in fixed]

def _micro_adjust_border(i: int,
                         bounds: List[Tuple[int,int]],
                         rows: List[List],
                         asr_tokens: List[str],
                         ref_tokens_rows: List[List[str]],
                         tcs: List[float]) -> None:
    """Mueve ≤ MAX_MICRO_MOVE tokens entre fila i e i+1 si mejora claramente."""
    if i < 0 or i >= len(bounds)-1: return
    hs, he = bounds[i]
    nhs, nhe = bounds[i+1]
    if not (0 <= hs <= he <= len(asr_tokens)): return
    if not (0 <= nhs <= nhe <= len(asr_tokens)): return

    # candidatos a mover (cola de i / cabeza de i+1)
    max_k_left = min(MAX_MICRO_MOVE, he - hs)
    max_k_right = min(MAX_MICRO_MOVE, nhe - nhs)

    # No cruzar pausas largas
    def pause_ok_cut_left(k: int) -> bool:
        if k <= 0 or he - k - 1 < hs or he - 1 >= len(tcs): return False
        return (tcs[he-1] - tcs[he-k-1]) <= PAUSE_SPLIT_SEC

    def pause_ok_cut_right(k: int) -> bool:
        if k <= 0 or nhs + k >= nhe or nhs + k < len(tcs):
            return (nhs + k < len(tcs) and (tcs[nhs + k] - tcs[nhs]) <= PAUSE_SPLIT_SEC)
        return False

    refL = ref_tokens_rows[i]
    refR = ref_tokens_rows[i+1]

    def score_pair(hs, he, nhs, nhe) -> float:
        # menor es mejor (WER total)
        left = asr_tokens[hs:he]
        right = asr_tokens[nhs:nhe]
        return _wer_pct(refL, left) + _wer_pct(refR, right)

    base = score_pair(hs, he, nhs, nhe)
    best = (base, hs, he, nhs, nhe)

    # mover del inicio de la derecha hacia la izquierda
    for k in range(1, max_k_right+1):
        if not (nhs + k <= nhe): break
        moved = asr_tokens[nhs:nhs+k]
        # no muevas anclas (fechas/meses)
        if any(_is_anchor(w) for w in moved): break
        if (tcs[nhs+k-1] - tcs[nhs]) > PAUSE_SPLIT_SEC: break
        cand = score_pair(hs, he + k, nhs + k, nhe)
        if base - cand >= MIN_WER_IMPROVE and cand < best[0]:
            best = (cand, hs, he + k, nhs + k, nhe)
            break  # con una mejora clara alcanza

    # mover del final de la izquierda hacia la derecha
    for k in range(1, max_k_left+1):
        if not (hs + k <= he): break
        moved = asr_tokens[he-k:he]
        if any(_is_anchor(w) for w in moved): break
        if (tcs[he-1] - tcs[he-k]) > PAUSE_SPLIT_SEC: break
        cand = score_pair(hs, he - k, nhs, nhe + k)
        if base - cand >= MIN_WER_IMPROVE and cand < best[0]:
            best = (cand, hs, he - k, nhs, nhe + k)
            break

    _, hs2, he2, nhs2, nhe2 = best
    bounds[i] = (hs2, he2)
    bounds[i+1] = (nhs2, nhe2)

def _recompute_rows(bounds: List[Tuple[int,int]],
                    rows: List[List],
                    asr_tokens: List[str],
                    asr_tcs: List[float]) -> List[List]:
    """Actualiza asr, tc, wer, flag; mantiene IDs."""
    out = []
    for rid, (hs, he), row in zip(range(len(bounds)), bounds, rows):
        _id, _flag, _wer, _tc, original, _old_asr = row
        hyp = asr_tokens[hs:he]
        ref = _tok(original)
        wer = _wer_pct(ref, hyp)
        flag = "✅" if wer <= 12.0 else ("⚠️" if wer <= 20.0 else "❌")
        tc = f"{asr_tcs[hs]:.2f}" if 0 <= hs < len(asr_tcs) else _tc
        out.append([rid, flag, round(wer, 1), tc, original, " ".join(hyp)])
    return out

# ----------------------------- API pública -----------------------------
def rectify_rows(rows: List[List],
                 csv_words: List[str],
                 csv_tcs: List[float]) -> List[List]:
    """
    Segunda pasada (R2) sobre la salida actual de alineación.
    Garantiza: cobertura total del ASR, contiguidad y microajustes seguros.
    """
    if not rows:
        return rows

    # 1) Normalizaciones
    asr_tokens = [_norm_token(w) for w in csv_words]
    ref_tokens_rows = [_tok(r[4]) for r in rows]  # original por fila
    old_asr_rows = [_tok(r[5]) for r in rows]     # asr por fila (lo existente)

    # 2) Búsqueda de “inicio probable” de cada fila usando el prefijo de la SIGUIENTE fila
    starts_hint = [None] * len(rows)
    cursor = 0
    for i in range(len(rows)-1):
        # prefijo del ref de la siguiente fila; si está vacío, usá su asr
        pat = ref_tokens_rows[i+1][:4] or old_asr_rows[i+1][:4]
        starts_hint[i+1] = _find_next_prefix(asr_tokens, pat, cursor, csv_tcs)
        # avanza un poco la ventana para no quedarse atrás
        if starts_hint[i+1] is not None:
            cursor = max(cursor, starts_hint[i+1])

    # 3) Bounds iniciales: asignar de manera streaming (sin perder nada)
    bounds: List[Tuple[int,int]] = []
    j = 0
    for i in range(len(rows)):
        if i < len(rows)-1 and starts_hint[i+1] is not None and starts_hint[i+1] >= j:
            hs, he = j, starts_hint[i+1]
        else:
            # si no hay pista, asigná al menos el largo del ASR actual de la fila
            need = max(0, len(old_asr_rows[i]))
            hs, he = j, min(len(asr_tokens), j + need)
        bounds.append((hs, he))
        j = he
    # última fila se queda con el resto
    if bounds:
        last_hs, last_he = bounds[-1]
        bounds[-1] = (last_hs, len(asr_tokens))

    # 4) Pasada de contigüidad (sin huecos ni solapes)
    bounds = _contiguity_pass(bounds)

    # 5) Microajuste seguro en cada frontera (≤3 tokens, sólo si mejora WER)
    for i in range(len(bounds)-1):
        _micro_adjust_border(i, bounds, rows, asr_tokens, ref_tokens_rows, csv_tcs)

    # 6) Recalcular filas (asr, tc, wer, flag) y devolver
    return _recompute_rows(bounds, rows, asr_tokens, csv_tcs)

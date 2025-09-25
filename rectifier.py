"""Second-pass alignment rectifier (R2) used by the QC app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import re
import unicodedata

from text_utils import normalize

SIM_THRESH = 0.82
LOOKAHEAD = 140
PAUSE_SPLIT_SEC = 0.80
MAX_MICRO_MOVE = 3
MIN_WER_IMPROVE = 2.0

MONTHS = {
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "setiembre",
    "octubre",
    "noviembre",
    "diciembre",
}

_DIGIT_RE = re.compile(r"^\d{1,4}$")
_PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]')


@dataclass
class RectifyReport:
    """Diagnostics produced by :func:`rectify_rows`."""

    anchors: List[int]
    anomalies: List[int]
    empty_rows: List[int]
    total_moves: int
    notes: List[str]

    def as_dict(self) -> dict:
        return {
            "anchors": self.anchors,
            "anomalies": self.anomalies,
            "empty_rows": self.empty_rows,
            "total_moves": self.total_moves,
            "notes": list(self.notes),
        }


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _norm_token(token: str) -> str:
    token = _strip_accents(token.lower())
    token = _PUNCT_RE.sub(" ", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def _tok(text: str) -> List[str]:
    text = normalize(text, strip_punct=False)
    return [t for t in text.split() if t]


def _lev(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
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


def _sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - (_lev(a, b) / m)


def _tok_sim(a: str, b: str) -> bool:
    return _sim(a, b) >= SIM_THRESH


def _wer_pct(ref_tokens: Sequence[str], hyp_tokens: Sequence[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 100.0
    lr, lh = len(ref_tokens), len(hyp_tokens)
    dp = list(range(lh + 1))
    for i in range(1, lr + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lh + 1):
            cur = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return min(100.0, 100.0 * dp[lh] / max(1, lr))


def _is_anchor(token: str) -> bool:
    return token in MONTHS or _DIGIT_RE.match(token) is not None


def _find_next_prefix(
    asr_tokens: Sequence[str],
    pattern: Sequence[str],
    start: int,
    tcs: Sequence[float],
) -> int | None:
    if not pattern:
        return None
    end = min(len(asr_tokens), start + LOOKAHEAD)
    best_idx: int | None = None
    best_score = 0.0
    plen = min(4, len(pattern))
    prefix = pattern[:plen]
    for j in range(start, end):
        matches = 0
        for k in range(plen):
            if j + k >= len(asr_tokens):
                break
            if _tok_sim(asr_tokens[j + k], prefix[k]):
                matches += 1
        score = matches / max(1, plen)
        if j > 0 and j < len(tcs) and (tcs[j] - tcs[j - 1]) >= PAUSE_SPLIT_SEC:
            score += 0.15
        if score >= 0.60 and (best_idx is None or score > best_score):
            best_idx = j
            best_score = score
            if score >= 0.95:
                break
    return best_idx


def _contiguity_pass(bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not bounds:
        return bounds
    fixed = [list(b) for b in bounds]
    for i in range(len(fixed) - 1):
        hs, he = fixed[i]
        nhs, nhe = fixed[i + 1]
        if nhs > he:
            fixed[i][1] = nhs
        elif nhs < he:
            mid = (nhs + he) // 2
            fixed[i][1] = mid
            fixed[i + 1][0] = mid
    return [(a, b) for a, b in fixed]


def _micro_adjust_border(
    idx: int,
    bounds: List[Tuple[int, int]],
    ref_tokens_rows: Sequence[Sequence[str]],
    asr_tokens: Sequence[str],
    tcs: Sequence[float],
) -> bool:
    if idx < 0 or idx >= len(bounds) - 1:
        return False
    hs, he = bounds[idx]
    nhs, nhe = bounds[idx + 1]
    if not (0 <= hs <= he <= len(asr_tokens)):
        return False
    if not (0 <= nhs <= nhe <= len(asr_tokens)):
        return False
    ref_left = ref_tokens_rows[idx]
    ref_right = ref_tokens_rows[idx + 1]

    def score_pair(hs_val: int, he_val: int, nhs_val: int, nhe_val: int) -> float:
        left = asr_tokens[hs_val:he_val]
        right = asr_tokens[nhs_val:nhe_val]
        return _wer_pct(ref_left, left) + _wer_pct(ref_right, right)

    base = score_pair(hs, he, nhs, nhe)
    best = (base, hs, he, nhs, nhe)
    changed = False

    max_k_left = min(MAX_MICRO_MOVE, he - hs)
    max_k_right = min(MAX_MICRO_MOVE, nhe - nhs)

    for k in range(1, max_k_right + 1):
        if nhs + k > nhe:
            break
        moved = asr_tokens[nhs:nhs + k]
        if any(_is_anchor(tok) for tok in moved):
            break
        if (tcs[min(len(tcs) - 1, nhs + k - 1)] - tcs[min(len(tcs) - 1, nhs)]) > PAUSE_SPLIT_SEC:
            break
        cand = score_pair(hs, he + k, nhs + k, nhe)
        if base - cand >= MIN_WER_IMPROVE and cand < best[0]:
            best = (cand, hs, he + k, nhs + k, nhe)
            changed = True
            break

    for k in range(1, max_k_left + 1):
        if hs + k > he:
            break
        moved = asr_tokens[he - k:he]
        if any(_is_anchor(tok) for tok in moved):
            break
        if (tcs[min(len(tcs) - 1, he - 1)] - tcs[min(len(tcs) - 1, he - k)]) > PAUSE_SPLIT_SEC:
            break
        cand = score_pair(hs, he - k, nhs, nhe + k)
        if base - cand >= MIN_WER_IMPROVE and cand < best[0]:
            best = (cand, hs, he - k, nhs, nhe + k)
            changed = True
            break

    _, hs2, he2, nhs2, nhe2 = best
    bounds[idx] = (hs2, he2)
    bounds[idx + 1] = (nhs2, nhe2)
    return changed


def _recompute_rows(
    bounds: Sequence[Tuple[int, int]],
    rows: Sequence[Sequence],
    asr_tokens: Sequence[str],
    asr_tcs: Sequence[float],
    flag_fn: Callable[[Sequence[str], Sequence[str]], Tuple[str, float]] | None,
) -> List[List]:
    out: List[List] = []
    for rid, (hs, he), row in zip(range(len(bounds)), bounds, rows):
        original = str(row[4]) if len(row) > 4 else ""
        hyp_tokens = asr_tokens[hs:he]
        asr_text = " ".join(hyp_tokens)
        ref_tokens = _tok(original)
        if flag_fn is not None:
            flag, wer_pct = flag_fn(ref_tokens, hyp_tokens)
            wer = round(float(wer_pct), 1)
        else:
            wer = round(_wer_pct(ref_tokens, hyp_tokens), 1)
            flag = "ok" if wer <= 12.0 else ("warn" if wer <= 20.0 else "bad")
        if 0 <= hs < len(asr_tcs):
            tc_val = float(asr_tcs[hs])
        else:
            try:
                tc_val = float(row[3])
            except Exception:
                tc_val = 0.0
        out.append([rid, flag, wer, tc_val, original, asr_text])
    return out


def rectify_rows(
    rows: List[List],
    csv_words: Sequence[str],
    csv_tcs: Sequence[float],
    *,
    log: Callable[[str], None] | None = None,
    flag_fn: Callable[[Sequence[str], Sequence[str]], Tuple[str, float]] | None = None,
    return_report: bool = False,
):
    if not rows:
        report = RectifyReport([], [], [], 0, [])
        return (rows, report) if return_report else rows

    logger = log or (lambda *_: None)
    if len(csv_words) != len(csv_tcs):
        raise ValueError("csv_words and csv_tcs must have the same length")

    asr_tokens: List[str] = []
    tcs: List[float] = []
    for word, tc in zip(csv_words, csv_tcs):
        tok = _norm_token(word)
        if not tok:
            continue
        asr_tokens.append(tok)
        tcs.append(float(tc))

    ref_tokens_rows = [_tok(r[4]) for r in rows]
    old_asr_rows = [_tok(r[5]) for r in rows]

    starts_hint: List[int | None] = [None] * len(rows)
    cursor = 0
    for i in range(len(rows) - 1):
        pattern = ref_tokens_rows[i + 1][:4] or old_asr_rows[i + 1][:4]
        starts_hint[i + 1] = _find_next_prefix(asr_tokens, pattern, cursor, tcs)
        if starts_hint[i + 1] is not None:
            cursor = max(cursor, starts_hint[i + 1])

    bounds: List[Tuple[int, int]] = []
    j = 0
    for i in range(len(rows)):
        if (
            i < len(rows) - 1
            and starts_hint[i + 1] is not None
            and starts_hint[i + 1] >= j
        ):
            hs, he = j, starts_hint[i + 1]
        else:
            need = max(len(old_asr_rows[i]), len(ref_tokens_rows[i]))
            hs = j
            he = min(len(asr_tokens), j + need)
        bounds.append((hs, he))
        j = he
    if bounds:
        last_hs, _ = bounds[-1]
        bounds[-1] = (last_hs, len(asr_tokens))

    bounds = _contiguity_pass(bounds)

    move_count = 0
    for i in range(len(bounds) - 1):
        if _micro_adjust_border(i, bounds, ref_tokens_rows, asr_tokens, tcs):
            move_count += 1

    bounds = _contiguity_pass(bounds)

    out_rows = _recompute_rows(bounds, rows, asr_tokens, tcs, flag_fn)

    filtered: List[List] = [r for r in out_rows if str(r[4]).strip() or str(r[5]).strip()]
    if len(filtered) != len(out_rows):
        for idx, row in enumerate(filtered):
            row[0] = idx
        out_rows = filtered

    anchors = [
        idx for idx, toks in enumerate(ref_tokens_rows)
        if any(_is_anchor(tok) for tok in toks)
    ]
    empty_rows = [idx for idx, r in enumerate(out_rows) if not r[-1]]
    anomalies: list[int] = []
    prev_tc = -1.0
    for idx, row in enumerate(out_rows):
        try:
            tc_val = float(row[3])
        except Exception:
            tc_val = prev_tc
        if tc_val < prev_tc - 0.5:
            anomalies.append(idx)
            tc_val = prev_tc
            row[3] = tc_val
        else:
            prev_tc = max(prev_tc, tc_val)
        try:
            wer_val = float(row[2])
        except Exception:
            continue
        if wer_val >= 60.0:
            if idx not in anomalies:
                anomalies.append(idx)
            row[1] = "??"
            row[5] = ""
        elif wer_val >= 35.0 and idx not in anomalies:
            anomalies.append(idx)

    notes: List[str] = []
    if move_count:
        notes.append(f"boundary_adjustments={move_count}")
    if empty_rows:
        notes.append(f"empty_rows={len(empty_rows)}")
    if anomalies:
        notes.append(f"anomalies={len(anomalies)}")

    report = RectifyReport(anchors=anchors, anomalies=anomalies, empty_rows=empty_rows, total_moves=move_count, notes=notes)

    logger(
        f"R2 refine anchors={len(anchors)} moves={move_count} empty={len(empty_rows)} anomalies={len(anomalies)}"
    )

    return (out_rows, report) if return_report else out_rows

"""Alignment routines used by the QC application."""

from typing import List, Tuple

from rapidfuzz.distance import Levenshtein

from text_utils import (
    normalize,
    token_equal,
    STOP,
    STOP_WEIGHT,
)

LINE_LEN = 12
COARSE_W = 40
WARN_WER = 0.08
GAMMA_TIME = 0.3


def dtw_band(a: List[str], b: List[str], w: int) -> List[Tuple[int, int]]:
    """Dynamic time warping with positional cost."""

    n, m = len(a), len(b)
    W = max(w, abs(n - m))
    BIG = 1e9

    D = {(-1, -1): (0, None)}
    back = {}

    for i in range(n):
        lo = max(0, i - W)
        hi = min(m - 1, i + W)
        for j in range(lo, hi + 1):
            best_cost = BIG
            best_prev = None
            for di, dj, mv in ((-1, 0, 1), (0, -1, 1), (-1, -1, 0)):
                prev = (i + di, j + dj)
                if prev in D:
                    c = D[prev][0] + mv
                    if c < best_cost:
                        best_cost, best_prev = c, prev

            if token_equal(a[i], b[j]):
                match_cost = 0
            else:
                if a[i] in STOP or b[j] in STOP:
                    match_cost = STOP_WEIGHT
                else:
                    match_cost = 1
            pos_cost = GAMMA_TIME * abs((i / n) - (j / m))
            total = best_cost + match_cost + pos_cost

            D[(i, j)] = (total, best_prev)
            back[(i, j)] = best_prev

    if (n - 1, m - 1) not in back:
        raise RuntimeError("DTW fuera de ventana")

    path = []
    i, j = n - 1, m - 1
    while (i, j) != (-1, -1):
        path.append((i, j))
        prev = back[(i, j)]
        if prev is None:
            break
        i, j = prev
    return path[::-1]


def fallback_pairs(
    ref_tokens: List[str], hyp_tokens: List[str]
) -> List[Tuple[int, int]]:
    """Monotonic matching when DTW fails."""

    pairs = []
    j_last = 0
    for i, t in enumerate(ref_tokens):
        for j in range(j_last, len(hyp_tokens)):
            if token_equal(t, hyp_tokens[j]):
                pairs.append((i, j))
                j_last = j + 1
                break
    return pairs


def safe_dtw(a: List[str], b: List[str], w: int) -> List[Tuple[int, int]]:
    """Try dtw_band, widening the band, falling back if needed."""

    band = w
    max_band = max(len(a), len(b)) * 2
    while band <= max_band:
        try:
            return dtw_band(a, b, band)
        except RuntimeError:
            band *= 2
    return fallback_pairs(a, b)


def build_rows(ref: str, hyp: str) -> List[List]:
    """Align reference and ASR transcript returning QC rows."""

    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = normalize(hyp, strip_punct=False).split()

    from text_utils import find_anchor_trigrams

    anchor_pairs = find_anchor_trigrams(ref_tok, hyp_tok)

    full_pairs: List[Tuple[int, int]] = []
    seg_starts = [
        (-1, -1),
        *anchor_pairs,
        (len(ref_tok) - 1, len(hyp_tok) - 1),
    ]
    for (prev_i, prev_j), (next_i, next_j) in zip(seg_starts[:-1], seg_starts[1:]):
        if next_i > prev_i + 1 and next_j > prev_j + 1:
            sub_ref = ref_tok[prev_i + 1 : next_i]
            sub_hyp = hyp_tok[prev_j + 1 : next_j]
            if sub_ref and sub_hyp:
                # remove stop words for alignment but keep position maps
                ref_idx = [i for i, t in enumerate(sub_ref) if t not in STOP]
                hyp_idx = [j for j, t in enumerate(sub_hyp) if t not in STOP]
                sub_r_sw = [sub_ref[i] for i in ref_idx]
                sub_h_sw = [sub_hyp[j] for j in hyp_idx]
                pairs = safe_dtw(sub_r_sw, sub_h_sw, COARSE_W)
                for ri, hj in pairs:
                    full_pairs.append(
                        (prev_i + 1 + ref_idx[ri], prev_j + 1 + hyp_idx[hj])
                    )
        if 0 <= next_i < len(ref_tok) and 0 <= next_j < len(hyp_tok):
            full_pairs.append((next_i, next_j))

    # ensure one-to-one mapping and avoid propagating indexes
    full_pairs.sort()
    used_h: dict[int, int] = {}
    dedup_pairs: List[Tuple[int, int]] = []
    for ri, hj in full_pairs:
        if hj not in used_h or abs(ri - used_h[hj]) > 1:
            dedup_pairs.append((ri, hj))
            used_h[hj] = ri

    map_h = [-1] * len(ref_tok)
    for ri, hj in dedup_pairs:
        if (
            0 <= ri < len(ref_tok)
            and 0 <= hj < len(hyp_tok)
            and map_h[ri] == -1
        ):
            map_h[ri] = hj

    rows = []
    pos = 0
    line_id = 0
    consumed_h = set()

    while pos < len(ref_tok):
        max_end = min(pos + LINE_LEN, len(ref_tok))
        span_end = max_end
        for k in range(pos, max_end):
            tok = ref_tok[k]
            if tok.endswith(".") or tok in {".", "?", "!"}:
                span_end = k + 1
                break
            if tok.endswith(",") or tok.endswith(";") or tok in {",", ";"}:
                cond = k + 1 < len(ref_tok) and k + 2 - pos <= LINE_LEN
                add = 2 if cond else 1
                span_end = min(k + add, max_end)
                break

        block = ref_tok[pos:span_end]
        span_start = pos
        pos = span_end

        # First collect all candidate hyp-indices in this block,
        # then drop any that have already been consumed by previous rows
        h_idxs_all = [map_h[k] for k in range(span_start, span_end) if map_h[k] != -1]
        h_idxs = [h for h in h_idxs_all if h not in consumed_h]

        if h_idxs:
            h_start = min(h_idxs)
            h_end = max(h_idxs) + 1

            # Backfill up to two preceding stop words, but only if not already consumed
            for _ in range(2):
                if (
                    h_start > 0
                    and hyp_tok[h_start - 1] in STOP
                    and (h_start - 1) not in consumed_h
                ):
                    h_start -= 1
                else:
                    break

            # Extend for any missing reference tokens by including prior hyp tokens,
            # as long as they are not "." or ";" and not already consumed
            missing = sum(1 for k in range(span_start, span_end) if map_h[k] == -1)
            for _ in range(missing):
                if (
                    h_start > 0
                    and hyp_tok[h_start - 1] not in {".", ";"}
                    and (h_start - 1) not in consumed_h
                ):
                    h_start -= 1
                else:
                    break

            asr_line = " ".join(hyp_tok[h_start:h_end])
            # Mark these hyp indices as “consumed” so future rows won’t reuse them
            consumed_h.update(range(h_start, h_end))
        else:
            asr_line = ""

        orig_line = " ".join(block)
        ref_tokens = orig_line.split()
        hyp_tokens = asr_line.split()
        if hyp_tokens:
            wer_val = Levenshtein.normalized_distance(ref_tokens, hyp_tokens)
        else:
            wer_val = 1.0
        flag = (
            "✅" if wer_val <= WARN_WER else ("⚠️" if wer_val <= 0.20 else "❌")
        )
        dur = round(len(asr_line.split()) / 3.0, 2)

        rows.append([line_id, flag, round(wer_val * 100, 1), dur, orig_line, asr_line])
        line_id += 1

    return rows
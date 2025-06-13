"""Alignment routines used by the QC application."""

from typing import List, Tuple
import json
import re

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
    seg_starts = [(-1, -1)] + anchor_pairs + [
        (len(ref_tok) - 1, len(hyp_tok) - 1)
    ]
    for (prev_i, prev_j), (next_i, next_j) in zip(
        seg_starts[:-1], seg_starts[1:]
    ):
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
    consumed_h = set()
    line_id = 0

    # divide reference into sentence spans for cleaner rows
    text_norm = normalize(ref, strip_punct=False)
    sentences = re.split(r"(?<=[\.\?\!])\s+", text_norm)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for sent in sentences:
        tokens_sent = sent.split()
        length = len(tokens_sent)
        if length > 0:
            spans.append((pos, pos + length))
            pos += length

    for span_start, span_end in spans:
        block = ref_tok[span_start:span_end]

        h_idxs_all = [map_h[k] for k in range(span_start, span_end) if map_h[k] != -1]
        h_idxs = [h for h in h_idxs_all if h not in consumed_h]
        if h_idxs:
            h_start = min(h_idxs)
            h_end = max(h_idxs) + 1
            for _ in range(2):
                if (
                    h_start > 0
                    and hyp_tok[h_start - 1] in STOP
                    and (h_start - 1) not in consumed_h
                ):
                    h_start -= 1
                else:
                    break

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

            consumed_h.update(range(h_start, h_end))
            asr_line = " ".join(hyp_tok[h_start:h_end])
        else:
            asr_line = ""

        orig_line = " ".join(block)
        ref_tokens = orig_line.split()
        hyp_tokens = asr_line.split()
        if hyp_tokens:
            wer_val = Levenshtein.normalized_distance(ref_tokens, hyp_tokens)
            base_ref = [r.strip(".,;!") for r in ref_tokens]
            base_hyp = [h.strip(".,;!") for h in hyp_tokens]
            base_wer = Levenshtein.normalized_distance(base_ref, base_hyp)
        else:
            wer_val = 1.0
            base_wer = 1.0

        if base_wer <= 0.05:
            flag = "✅"
        else:
            threshold = 0.20 if len(ref_tokens) < 5 else WARN_WER
            flag = "✅" if wer_val <= threshold else ("⚠️" if wer_val <= 0.20 else "❌")
        dur = round(len(asr_line.split()) / 3.0, 2)

        rows.append([line_id, flag, round(wer_val * 100, 1), dur, orig_line, asr_line])
        line_id += 1

    return refine_segments(rows)


def refine_segments(rows: List[List], max_shift: int = 2) -> List[List]:
    """Fine-tune ASR segments by shifting words across boundaries."""

    rows = [r[:] for r in rows]

    def _wer(ref_t: List[str], hyp_t: List[str]) -> float:
        return Levenshtein.normalized_distance(ref_t, hyp_t)

    for i in range(len(rows) - 1):
        ref_i = rows[i][4].split()
        ref_next = rows[i + 1][4].split()
        asr_i = rows[i][5].split()
        asr_next = rows[i + 1][5].split()
        cost = _wer(ref_i, asr_i) + _wer(ref_next, asr_next)
        improved = True
        while improved:
            improved = False
            for k in range(1, max_shift + 1):
                if k <= len(asr_next):
                    new_i = asr_i + asr_next[:k]
                    new_n = asr_next[k:]
                    c = _wer(ref_i, new_i) + _wer(ref_next, new_n)
                    if c < cost:
                        asr_i, asr_next = new_i, new_n
                        cost = c
                        improved = True
                        break
            for k in range(1, max_shift + 1):
                if k <= len(asr_i):
                    new_i = asr_i[:-k]
                    if not new_i:
                        continue
                    new_n = asr_i[-k:] + asr_next
                    c = _wer(ref_i, new_i) + _wer(ref_next, new_n)
                    if c < cost:
                        asr_i, asr_next = new_i, new_n
                        cost = c
                        improved = True
                        break
        rows[i][5] = " ".join(asr_i)
        rows[i + 1][5] = " ".join(asr_next)

    for r in rows:
        ref_tokens = r[4].split()
        hyp_tokens = r[5].split()
        if hyp_tokens:
            wer_val = Levenshtein.normalized_distance(ref_tokens, hyp_tokens)
            base_ref = [t.strip(".,;!") for t in ref_tokens]
            base_hyp = [t.strip(".,;!") for t in hyp_tokens]
            base_wer = Levenshtein.normalized_distance(base_ref, base_hyp)
        else:
            wer_val = 1.0
            base_wer = 1.0
        if base_wer <= 0.05:
            flag = "✅"
        else:
            threshold = 0.20 if len(ref_tokens) < 5 else WARN_WER
            flag = "✅" if wer_val <= threshold else ("⚠️" if wer_val <= 0.20 else "❌")
        r[1] = flag
        r[2] = round(wer_val * 100, 1)
        r[3] = round(len(hyp_tokens) / 3.0, 2)
    return rows


def build_rows_wordlevel(ref: str, asr_word_json: str) -> List[List]:
    """Align using word-level timestamps and compute real durations."""

    data = json.loads(asr_word_json)
    segments = data.get("segments", data)
    words = []
    for seg in segments:
        for w in seg.get("words", []):
            tok = w.get("word", w.get("text", ""))
            words.append(
                {
                    "word": tok,
                    "norm": normalize(tok, strip_punct=False),
                    "start": float(w.get("start", seg.get("start", 0.0))),
                    "end": float(w.get("end", seg.get("end", 0.0))),
                }
            )

    hyp_tok = [w["norm"] for w in words]
    ref_tok = normalize(ref, strip_punct=False).split()

    pairs = safe_dtw(ref_tok, hyp_tok, COARSE_W)
    map_h = [-1] * len(ref_tok)
    prev_i = prev_j = -1
    for i, j in pairs:
        if i > prev_i and j > prev_j:
            map_h[i] = j
        prev_i, prev_j = i, j

    rows = []
    consumed_h = set()
    line_id = 0

    text_norm = normalize(ref, strip_punct=False)
    sentences = re.split(r"(?<=[\.\?\!])\s+", text_norm)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for sent in sentences:
        tokens_sent = sent.split()
        length = len(tokens_sent)
        if length > 0:
            spans.append((pos, pos + length))
            pos += length

    for span_start, span_end in spans:
        block = ref_tok[span_start:span_end]
        h_idxs = [map_h[k] for k in range(span_start, span_end) if map_h[k] != -1 and map_h[k] not in consumed_h]
        if h_idxs:
            h_start = min(h_idxs)
            h_end = max(h_idxs) + 1
            consumed_h.update(range(h_start, h_end))
            asr_line = " ".join(w["norm"] for w in words[h_start:h_end])
            start_time = words[h_start]["start"]
            end_time = words[h_end - 1]["end"]
            dur = round(end_time - start_time, 2)
        else:
            asr_line = ""
            dur = 0.0

        orig_line = " ".join(block)
        ref_tokens = orig_line.split()
        hyp_tokens = asr_line.split()
        if hyp_tokens:
            wer_val = Levenshtein.normalized_distance(ref_tokens, hyp_tokens)
            base_ref = [r.strip(".,;!") for r in ref_tokens]
            base_hyp = [h.strip(".,;!") for h in hyp_tokens]
            base_wer = Levenshtein.normalized_distance(base_ref, base_hyp)
        else:
            wer_val = 1.0
            base_wer = 1.0

        if base_wer <= 0.05:
            flag = "✅"
        else:
            threshold = 0.20 if len(ref_tokens) < 5 else WARN_WER
            flag = "✅" if wer_val <= threshold else ("⚠️" if wer_val <= 0.20 else "❌")

        rows.append([line_id, flag, round(wer_val * 100, 1), dur, orig_line, asr_line])
        line_id += 1

    return rows

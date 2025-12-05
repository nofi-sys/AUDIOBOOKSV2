from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Sequence
from difflib import SequenceMatcher

from text_utils import (
    ANCHOR_STOPWORDS,
    normalize,
    prepare_paragraphs,
    token_equal,
    find_anchor_trigrams,
)
import alignment

CLIP_SIM_TOKENS = 50
CLIP_SIM_THRESHOLD = 0.55


def _tokenize_normalized(text: str) -> List[str]:
    """Lowercase, strip punctuation and split."""
    return normalize(text, strip_punct=True).split()


def _paragraph_tokens(paragraphs: Sequence[str]) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Flattens paragraph tokens and records spans (start, end) for each paragraph."""
    ref_tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    offset = 0
    for para in paragraphs:
        toks = _tokenize_normalized(para)
        if not toks:
            continue
        spans.append((offset, offset + len(toks)))
        ref_tokens.extend(toks)
        offset += len(toks)
    return ref_tokens, spans


def _best_window(pattern: List[str], ref_tokens: List[str], direction: str) -> Tuple[int, float]:
    """Returns (idx, score) of the best consecutive match for pattern."""
    n = len(pattern)
    best_idx, best_score = -1, -1.0
    for i in range(0, len(ref_tokens) - n + 1):
        window = ref_tokens[i:i + n]
        score = sum(1 for a, b in zip(pattern, window) if token_equal(a, b)) / n
        if score > best_score or (score == best_score and direction == "end" and i > best_idx):
            best_idx = i
            best_score = score
    return best_idx, best_score


def _search_anchor_patterns(
    ref_tokens: List[str],
    patterns: List[List[str]],
    direction: str,
) -> Tuple[int, List[str], float]:
    """
    Busca la mejor ventana favoreciendo primero patrones mas largos.
    Devuelve (idx, pattern, score).
    """

    def _content_score(pat: List[str]) -> int:
        return sum(1 for t in pat if t.isdigit() or (len(t) >= 4 and t not in {"para", "como", "pero"}))

    def _valid_anchor(pat: List[str]) -> bool:
        stop_hits = sum(1 for t in pat if t in ANCHOR_STOPWORDS)
        return stop_hits < max(1, len(pat) // 2)

    for size in (5, 4, 3):
        best_idx, best_score, best_pat = -1, -1.0, []
        pats = [p for p in patterns if len(p) == size and _content_score(p) >= 2 and _valid_anchor(p)]
        if not pats:
            continue
        for pat in pats:
            idx, score = _best_window(pat, ref_tokens, direction)
            if score >= 0.8 and any(tok.isdigit() for tok in pat):
                window = ref_tokens[idx:idx + len(pat)]
                # Las cifras deben coincidir (aceptando equivalentes como "17"/"diecisiete")
                if not all(token_equal(a, b) for a, b in zip(pat, window) if a.isdigit()):
                    score = -1.0
            if score > best_score:
                best_idx, best_score, best_pat = idx, score, pat
        if best_score >= 0.8:
            return best_idx, best_pat, best_score
    return -1, [], -1.0


def _clip_similarity(ref_tokens: List[str], asr_tokens: List[str], start_tok: int, end_tok: int) -> Tuple[float, float]:
    """Rough check to flag dubious clip proposals comparing ends of the window."""
    if not ref_tokens or not asr_tokens:
        return 0.0, 0.0
    head_ref = ref_tokens[max(0, start_tok - CLIP_SIM_TOKENS // 2): start_tok + CLIP_SIM_TOKENS]
    tail_ref = ref_tokens[max(0, end_tok - CLIP_SIM_TOKENS): min(len(ref_tokens), end_tok + CLIP_SIM_TOKENS // 2)]
    head_asr = asr_tokens[:CLIP_SIM_TOKENS]
    tail_asr = asr_tokens[-CLIP_SIM_TOKENS:]
    sim_start = SequenceMatcher(a=head_ref, b=head_asr, autojunk=False).ratio() if head_ref and head_asr else 0.0
    sim_end = SequenceMatcher(a=tail_ref, b=tail_asr, autojunk=False).ratio() if tail_ref and tail_asr else 0.0
    return sim_start, sim_end


def propose_clip(original_text: str, asr_tokens: List[str], tcs: Optional[List[float]] = None) -> dict:
    """
    Propone un rango de párrafos [start, end) del original que cubra mejor el ASR.

    Estrategia principal: anclar por coincidencia normalizada de las primeras y
    últimas palabras del ASR en el texto original, con tamaños 5→4→3.
    Si no hay anclas confiables, se recurre al alineador sentwise.
    """
    paragraphs = prepare_paragraphs(original_text)
    ref_tokens, spans = _paragraph_tokens(paragraphs)
    if not ref_tokens or not asr_tokens:
        return {
            "paragraphs": paragraphs,
            "start_par": 0,
            "end_par": len(paragraphs),
            "start_tok": 0,
            "end_tok": len(ref_tokens),
            "anchors": [],
        }

    asr_norm = [alignment._normalize_token(w) for w in asr_tokens]
    if tcs is None:
        tcs = [i * (1.0 / alignment.WPS) for i in range(len(asr_norm))]

    start_tok = 0
    end_tok = len(ref_tokens)
    start_pattern = None
    end_pattern = None

    # Ancla de inicio (primeras ~15 palabras ASR)
    asr_head = [tok for tok in asr_norm[:15] if tok]
    head_cands: List[List[str]] = []
    for size in (5, 4, 3):
        if len(asr_head) >= size:
            head_cands.extend(asr_head[i:i + size] for i in range(0, len(asr_head) - size + 1))
    idx, pat, _ = _search_anchor_patterns(ref_tokens, head_cands, direction="start")
    if idx != -1:
        start_tok = idx
        start_pattern = pat

    # Ancla de fin (últimas ~30 palabras ASR)
    asr_tail = [tok for tok in asr_norm[-30:] if tok]
    tail_cands: List[List[str]] = []
    for size in (5, 4, 3):
        if len(asr_tail) >= size:
            tail_cands.extend(asr_tail[i:i + size] for i in range(0, len(asr_tail) - size + 1))
    idx, pat, _ = _search_anchor_patterns(ref_tokens, tail_cands, direction="end")
    if idx != -1:
        end_tok = idx + len(pat)
        end_pattern = pat

    # Fallback/upgrade: trigrama poco frecuente
    anchors = find_anchor_trigrams(ref_tokens, asr_norm)
    if anchors:
        if start_pattern is None:
            start_tok = anchors[0][0]
            start_pattern = ref_tokens[start_tok:start_tok + 3]
        end_tok = min(len(ref_tokens), anchors[-1][0] + 3)
        end_pattern = ref_tokens[max(0, end_tok - 3):end_tok]

    # Fallback 2: usar alineación sentwise si aún falta ancla
    if start_pattern is None or end_pattern is None or start_tok >= end_tok:
        pairs = alignment._align_sentwise(ref_tokens, asr_norm, tcs)
        if not pairs:
            return {
                "paragraphs": paragraphs,
                "start_par": 0,
                "end_par": len(paragraphs),
                "start_tok": 0,
                "end_tok": len(ref_tokens),
                "anchors": anchors,
                "start_pattern": start_pattern,
                "end_pattern": end_pattern,
                "sim_start": 0.0,
                "sim_end": 0.0,
                "dubious": True,
            }
        start_tok = min(ri for ri, _ in pairs)
        end_tok = max(ri for ri, _ in pairs) + 1

    start_tok = max(0, min(start_tok, len(ref_tokens)))
    end_tok = max(start_tok, min(end_tok, len(ref_tokens)))
    sim_start, sim_end = _clip_similarity(ref_tokens, asr_norm, start_tok, end_tok)
    # Si uno de los extremos se parece poco al ASR, marcar como dudoso.
    dubious = (sim_start < CLIP_SIM_THRESHOLD) or (sim_end < CLIP_SIM_THRESHOLD)

    # Mapear tokens a párrafos
    start_par = 0
    end_par = len(paragraphs)
    for idx, (s, e) in enumerate(spans):
        if s <= start_tok < e:
            start_par = idx
            break
    for idx, (s, e) in enumerate(spans):
        if e >= end_tok:
            end_par = idx + 1
            break

    return {
        "paragraphs": paragraphs,
        "start_par": start_par,
        "end_par": end_par,
        "start_tok": start_tok,
        "end_tok": end_tok,
        "anchors": anchors,
        "start_pattern": start_pattern,
        "end_pattern": end_pattern,
        "sim_start": sim_start,
        "sim_end": sim_end,
        "dubious": dubious,
    }


def save_clip(paragraphs: Sequence[str], start_par: int, end_par: int, dest_path: Path) -> Path:
    """Persist clipped paragraphs to disk."""
    text = "\n\n".join(paragraphs[start_par:end_par])
    dest_path.write_text(text, encoding="utf-8")
    return dest_path

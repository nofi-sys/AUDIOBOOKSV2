from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

from text_utils import normalize


@dataclass
class AlignedWord:
    ref_idx: Optional[int]
    asr_idx: Optional[int]
    ref_word: str
    asr_word: str
    is_anchor: bool
    row_id_ref: Optional[int] = None
    row_id_asr: Optional[int] = None
    tc: Optional[float] = None  # approximate start time from ASR tokens


@dataclass
class Block:
    block_id: int
    kind: str  # "anchor" | "inter"
    words: List[AlignedWord]
    ref_range: Optional[Tuple[int, int]] = None
    asr_range: Optional[Tuple[int, int]] = None
    row_ids: set[int] = field(default_factory=set)
    possible_repeat: bool = False
    repeat_source: Optional[str] = None
    repeat_span: Optional[Tuple[int, int]] = None  # offsets inside block.asr tokens

    def ref_text(self) -> str:
        return " ".join(w.ref_word for w in self.words if w.ref_word)

    def asr_text(self) -> str:
        return " ".join(w.asr_word for w in self.words if w.asr_word)

    def asr_tokens(self) -> list[str]:
        tokens: list[str] = []
        for w in self.words:
            if w.asr_word:
                tokens.extend(normalize(w.asr_word).split())
        return tokens

    def ref_tokens(self) -> list[str]:
        tokens: list[str] = []
        for w in self.words:
            if w.ref_word:
                tokens.extend(normalize(w.ref_word).split())
        return tokens


@dataclass
class AlignmentBlocks:
    words: List[AlignedWord]
    blocks: List[Block]
    ref_to_row: List[Optional[int]] = field(default_factory=list)
    asr_to_row: List[Optional[int]] = field(default_factory=list)
    asr_tcs: List[Optional[float]] = field(default_factory=list)

    def mark_repetitions(
        self,
        min_len: int = 3,
        max_len: int = 6,
        threshold: float = 85.0,
    ) -> None:
        """Flag inter-anchor blocks that look like short repeats of neighbors."""
        for idx, block in enumerate(self.blocks):
            if block.kind != "inter":
                continue
            prev_anchor = self._find_anchor(idx, -1)
            next_anchor = self._find_anchor(idx, 1)
            asr_tokens = block.asr_tokens()
            for source, anchor in (("prev_anchor", prev_anchor), ("next_anchor", next_anchor)):
                if not anchor:
                    continue
                match = _best_overlap(anchor.asr_tokens(), asr_tokens, min_len, max_len, threshold)
                if not match:
                    match = _best_overlap(anchor.ref_tokens(), asr_tokens, min_len, max_len, threshold)
                if match:
                    block.possible_repeat = True
                    block.repeat_source = source
                    block.repeat_span = (match["start"], match["end"])
                    break

    def _find_anchor(self, start_idx: int, step: int) -> Optional[Block]:
        i = start_idx + step
        while 0 <= i < len(self.blocks):
            blk = self.blocks[i]
            if blk.kind == "anchor":
                return blk
            i += step
        return None

    def block_to_time(
        self,
        block: Block,
        rows: Optional[Sequence[Sequence]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return (start, end) seconds for a block using row or ASR timing."""
        row_tc: dict[int, float] = {}
        if rows:
            for row in rows:
                try:
                    row_id = int(row[0])
                    tc_val = _coerce_tc(row[6])
                except Exception:
                    continue
                if tc_val is not None:
                    row_tc[row_id] = tc_val

        row_ids = sorted(r for r in block.row_ids if r is not None)
        start: Optional[float] = None
        end: Optional[float] = None
        if row_tc and row_ids:
            start = row_tc.get(row_ids[0])
            sorted_ids = sorted(row_tc)
            last = row_ids[-1]
            try:
                next_idx = sorted_ids.index(last) + 1
                if next_idx < len(sorted_ids):
                    end = row_tc.get(sorted_ids[next_idx])
            except ValueError:
                pass
        if start is None and block.asr_range and self.asr_tcs:
            start = _get_tc(self.asr_tcs, block.asr_range[0])
        if end is None and block.asr_range and self.asr_tcs:
            end = _get_tc(self.asr_tcs, block.asr_range[1] + 1) or _get_tc(self.asr_tcs, block.asr_range[1])
        if start is None or end is None:
            return None
        return (start, end)


def load_alignment_paths(
    asr_path: str | Path,
    *,
    align_csv_path: str | Path | None = None,
    align_db_path: str | Path | None = None,
) -> AlignmentBlocks:
    """Load alignment pairs, blockify anchor/inter-anchor segments, and attach row links."""
    asr_path = Path(asr_path)
    align_csv = Path(align_csv_path) if align_csv_path else asr_path.with_suffix(".align.csv")
    align_db = Path(align_db_path) if align_db_path else asr_path.with_suffix(".align.db")

    ref_to_row: list[Optional[int]] = []
    asr_to_row: list[Optional[int]] = []
    asr_tcs: list[Optional[float]] = []
    if align_db.exists():
        ref_to_row, asr_to_row, asr_tcs = _load_db_maps(align_db)

    words = _read_alignment_csv(align_csv, ref_to_row, asr_to_row, asr_tcs)
    blocks = _build_blocks(words)
    ab = AlignmentBlocks(words=words, blocks=blocks, ref_to_row=ref_to_row, asr_to_row=asr_to_row, asr_tcs=asr_tcs)
    ab.mark_repetitions()
    return ab


# --------------------------------------------------------------------------- utils

def _read_alignment_csv(
    csv_path: Path,
    ref_to_row: Sequence[Optional[int]],
    asr_to_row: Sequence[Optional[int]],
    asr_tcs: Sequence[Optional[float]],
) -> list[AlignedWord]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Alineacion CSV no encontrado: {csv_path}")
    words: list[AlignedWord] = []
    ref_idx = 0
    asr_idx = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True
                if row and row[0].lower() == "ref_word":
                    continue
            if len(row) < 3:
                continue
            ref_word = row[0].strip()
            asr_word = row[1].strip()
            is_anchor = row[2].strip() in {"1", "true", "True"}
            current_ref = ref_idx if ref_word else None
            current_asr = asr_idx if asr_word else None
            words.append(
                AlignedWord(
                    ref_idx=current_ref,
                    asr_idx=current_asr,
                    ref_word=ref_word,
                    asr_word=asr_word,
                    is_anchor=is_anchor,
                    row_id_ref=_safe_index(ref_to_row, current_ref),
                    row_id_asr=_safe_index(asr_to_row, current_asr),
                    tc=_safe_index(asr_tcs, current_asr),
                )
            )
            if ref_word:
                ref_idx += 1
            if asr_word:
                asr_idx += 1
    return words


def _build_blocks(words: Sequence[AlignedWord]) -> list[Block]:
    blocks: list[Block] = []
    i = 0
    block_id = 0
    while i < len(words):
        is_anchor = words[i].is_anchor
        start = i
        while i < len(words) and words[i].is_anchor == is_anchor:
            i += 1
        chunk = list(words[start:i])
        ref_idxs = [w.ref_idx for w in chunk if w.ref_idx is not None]
        asr_idxs = [w.asr_idx for w in chunk if w.asr_idx is not None]
        row_ids: set[int] = set()
        for w in chunk:
            if w.row_id_asr is not None:
                row_ids.add(w.row_id_asr)
            if w.row_id_ref is not None:
                row_ids.add(w.row_id_ref)
        blocks.append(
            Block(
                block_id=block_id,
                kind="anchor" if is_anchor else "inter",
                words=chunk,
                ref_range=(min(ref_idxs), max(ref_idxs)) if ref_idxs else None,
                asr_range=(min(asr_idxs), max(asr_idxs)) if asr_idxs else None,
                row_ids=row_ids,
            )
        )
        block_id += 1
    return blocks


def _load_db_maps(db_path: Path) -> tuple[list[Optional[int]], list[Optional[int]], list[Optional[float]]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ref_len = cur.execute("SELECT COUNT(*) FROM ref_tokens").fetchone()[0] or 0
    asr_len = cur.execute("SELECT COUNT(*) FROM asr_tokens").fetchone()[0] or 0
    ref_to_row: list[Optional[int]] = [None] * ref_len
    asr_to_row: list[Optional[int]] = [None] * asr_len
    asr_tcs: list[Optional[float]] = [None] * asr_len

    for row_id, ref_start, ref_end, asr_start, asr_end in cur.execute(
        "SELECT row_id, ref_start, ref_end, asr_start, asr_end FROM paragraph_rows"
    ):
        for i in range(ref_start, min(ref_end, ref_len)):
            ref_to_row[i] = row_id
        for j in range(asr_start, min(asr_end, asr_len)):
            asr_to_row[j] = row_id
    for idx, tc in cur.execute("SELECT idx, tc FROM asr_tokens"):
        if 0 <= idx < len(asr_tcs):
            try:
                asr_tcs[idx] = float(tc)
            except Exception:
                asr_tcs[idx] = None
    conn.close()
    return ref_to_row, asr_to_row, asr_tcs


def _best_overlap(
    anchor_tokens: Sequence[str],
    block_tokens: Sequence[str],
    min_len: int,
    max_len: int,
    threshold: float,
) -> Optional[dict]:
    if not anchor_tokens or not block_tokens:
        return None
    best: Optional[dict] = None
    for length in range(min(max_len, len(anchor_tokens), len(block_tokens)), min_len - 1, -1):
        for ai in range(0, len(anchor_tokens) - length + 1):
            a_text = " ".join(anchor_tokens[ai : ai + length])
            for bi in range(0, len(block_tokens) - length + 1):
                b_text = " ".join(block_tokens[bi : bi + length])
                score = fuzz.ratio(a_text, b_text)
                if score >= threshold and (best is None or score > best["score"]):
                    best = {"start": bi, "end": bi + length, "score": score}
        if best:
            break
    return best


def _safe_index(seq: Sequence, idx: Optional[int]):
    if idx is None:
        return None
    try:
        return seq[idx]
    except Exception:
        return None


def _coerce_tc(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        text = str(val)
        if ":" in text:
            try:
                h, m, s = text.split(":")
                return float(h) * 3600 + float(m) * 60 + float(s)
            except Exception:
                return None
    return None


def _get_tc(seq: Sequence[Optional[float]], idx: int) -> Optional[float]:
    if idx < 0 or idx >= len(seq):
        return None
    return seq[idx]

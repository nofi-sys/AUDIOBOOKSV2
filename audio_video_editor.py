from __future__ import annotations
"""Generate continuous edit intervals from word-level transcripts."""

import json
from pathlib import Path
from typing import List, Dict

from difflib import SequenceMatcher

from text_utils import normalize, token_equal


def load_transcript(path: str) -> List[Dict]:
    """Return list of word dicts with text, start and end from stable-ts JSON."""
    data = json.loads(Path(path).read_text(encoding="utf8"))
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
    return words


def align_script(script: str, words: List[Dict]) -> List[int]:
    """Return mapping from script tokens to transcript word indexes."""
    ref_tok = normalize(script, strip_punct=False).split()
    hyp_tok = [w["norm"] for w in words]

    sm = SequenceMatcher(None, ref_tok, hyp_tok, autojunk=False)
    map_idx = [-1] * len(ref_tok)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for i, j in zip(range(i1, i2), range(j1, j2)):
                map_idx[i] = j
    # keep last occurrence for repeated lines
    for i in range(len(ref_tok) - 1):
        if map_idx[i] != -1:
            j = map_idx[i]
            while j + 1 < len(hyp_tok) and token_equal(ref_tok[i], hyp_tok[j + 1]):
                map_idx[i] = j + 1
                j += 1
    return map_idx


def build_intervals(script: str, transcript_path: str, merge_gap: float = 0.5) -> List[List[float]]:
    """Return continuous intervals [start, end] from script and transcript."""
    words = load_transcript(transcript_path)
    mapping = align_script(script, words)
    intervals: List[List[float]] = []
    current = None
    last_j = None
    for idx, j in enumerate(mapping):
        if j == -1:
            if current is not None:
                intervals.append([words[current]["start"], words[last_j]["end"]])
                current = None
            continue
        if current is None:
            current = j
        elif j != last_j + 1:
            intervals.append([words[current]["start"], words[last_j]["end"]])
            current = j
        last_j = j
    if current is not None:
        intervals.append([words[current]["start"], words[last_j]["end"]])

    if not intervals:
        return []

    merged: List[List[float]] = [intervals[0]]
    for start, end in intervals[1:]:
        prev_end = merged[-1][1]
        if start - prev_end <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate edit intervals from transcript")
    parser.add_argument("--script", required=True, help="Original script text file")
    parser.add_argument("--transcript", required=True, help="Word-level JSON transcript")
    parser.add_argument("--output-edl", required=True, help="Path to save intervals JSON")
    parser.add_argument("--gap", type=float, default=0.5, help="Max gap to merge segments")
    args = parser.parse_args()

    script_text = Path(args.script).read_text(encoding="utf8")
    intervals = build_intervals(script_text, args.transcript, merge_gap=args.gap)
    Path(args.output_edl).write_text(json.dumps(intervals, indent=2))
    print(f"Saved {len(intervals)} intervals to {args.output_edl}")


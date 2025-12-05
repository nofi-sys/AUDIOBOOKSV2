from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from alignment import Anchor


@dataclass
class ParagraphRow:
    id: int
    ref_start: int
    ref_end: int
    asr_start: int
    asr_end: int
    tc_start: float
    tc_end: float
    ref_text: str
    asr_text: str
    wer: float
    flag: str


@dataclass
class AlignmentSummary:
    paragraphs: int
    anchors_by_size: Dict[int, int]
    ref_tokens: int
    asr_tokens: int
    sample: List[ParagraphRow]


def store_alignment_snapshot(
    db_path: str | Path,
    paragraphs: Sequence[str],
    ref_tokens: Sequence[str],
    asr_tokens: Sequence[str],
    asr_norm_tokens: Sequence[str],
    anchors: Sequence[Anchor],
    rows_meta: Sequence[dict],
    tcs: Sequence[float],
    paragraph_spans: Sequence[tuple[int, int]] | None = None,
    sentence_spans: Sequence[tuple[int, int]] | None = None,
    alignments_word: Sequence[dict] | None = None,
) -> None:
    """
    Guarda una vista persistente del alineado en SQLite para inspeccion offline.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        DROP TABLE IF EXISTS meta;
        DROP TABLE IF EXISTS anchors;
        DROP TABLE IF EXISTS paragraphs;
        DROP TABLE IF EXISTS ref_tokens;
        DROP TABLE IF EXISTS asr_tokens;
        DROP TABLE IF EXISTS alignments_word;
        """
    )
    cur.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    cur.execute("CREATE TABLE anchors(ref_idx INTEGER, asr_idx INTEGER, size INTEGER)")
    cur.execute(
        """
        CREATE TABLE paragraphs(
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
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE ref_tokens(
            idx INTEGER PRIMARY KEY,
            token TEXT NOT NULL,
            paragraph_id INTEGER,
            sentence_id INTEGER,
            prev_idx INTEGER,
            next_idx INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE asr_tokens(
            idx INTEGER PRIMARY KEY,
            token TEXT NOT NULL,
            norm TEXT NOT NULL,
            tc_start REAL,
            tc_end REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE alignments_word(
            ref_idx INTEGER,
            asr_idx INTEGER,
            tipo TEXT,
            distancia REAL,
            row_id INTEGER,
            PRIMARY KEY (ref_idx, asr_idx)
        )
        """
    )

    para_spans = list(paragraph_spans or [])
    sent_spans = list(sentence_spans or [])

    para_ids = [-1] * len(ref_tokens)
    for pid, (s, e) in enumerate(para_spans):
        for k in range(s, min(e, len(para_ids))):
            para_ids[k] = pid

    sent_ids = [-1] * len(ref_tokens)
    for sid, (s, e) in enumerate(sent_spans):
        for k in range(s, min(e, len(sent_ids))):
            sent_ids[k] = sid

    cur.executemany(
        "INSERT INTO anchors(ref_idx, asr_idx, size) VALUES (?, ?, ?)",
        [(a.ref_idx, a.asr_idx, a.size) for a in anchors],
    )
    cur.executemany(
        "INSERT INTO ref_tokens(idx, token, paragraph_id, sentence_id, prev_idx, next_idx) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                i,
                tok,
                para_ids[i] if i < len(para_ids) else None,
                sent_ids[i] if i < len(sent_ids) else None,
                i - 1 if i > 0 else None,
                i + 1 if i + 1 < len(ref_tokens) else None,
            )
            for i, tok in enumerate(ref_tokens)
        ],
    )

    def _tc_end(idx: int) -> float | None:
        if idx + 1 < len(tcs):
            return float(tcs[idx + 1])
        if idx < len(tcs):
            return float(tcs[idx])
        return None

    cur.executemany(
        "INSERT INTO asr_tokens(idx, token, norm, tc_start, tc_end) VALUES (?, ?, ?, ?, ?)",
        [
            (
                i,
                tok,
                asr_norm_tokens[i] if i < len(asr_norm_tokens) else tok,
                float(tcs[i]) if i < len(tcs) else None,
                _tc_end(i),
            )
            for i, tok in enumerate(asr_tokens)
        ],
    )

    alignments_word = list(alignments_word or [])
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

    for row in rows_meta:
        start_tc = float(tcs[row['hs']]) if row['hs'] < len(tcs) else 0.0
        end_idx = max(row['he'] - 1, row['hs'])
        end_tc = float(tcs[end_idx]) if end_idx < len(tcs) else start_tc
        cur.execute(
            """
            INSERT INTO paragraphs(id, ref_start, ref_end, asr_start, asr_end,
                                   tc_start, tc_end, ref_text, asr_text, wer, flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
            ),
        )

    cur.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", [
        ("paragraphs", str(len(rows_meta))),
        ("ref_tokens", str(len(ref_tokens))),
        ("asr_tokens", str(len(asr_tokens))),
        ("alignments_word", str(len(alignments_word or []))),
    ])

    conn.commit()
    conn.close()


def summarize_alignment(db_path: str | Path, sample: int = 5) -> AlignmentSummary:
    conn = sqlite3.connect(Path(db_path))
    cur = conn.cursor()
    anchors_by_size = {
        size: count for size, count in cur.execute("SELECT size, COUNT(*) FROM anchors GROUP BY size")
    }
    paragraphs = cur.execute("SELECT COUNT(*) FROM paragraphs").fetchone()[0] or 0
    ref_tokens = cur.execute("SELECT COUNT(*) FROM ref_tokens").fetchone()[0] or 0
    asr_tokens = cur.execute("SELECT COUNT(*) FROM asr_tokens").fetchone()[0] or 0

    sample_rows: List[ParagraphRow] = []
    for row in cur.execute(
        "SELECT id, ref_start, ref_end, asr_start, asr_end, tc_start, tc_end, ref_text, asr_text, wer, flag "
        "FROM paragraphs ORDER BY id LIMIT ?",
        (sample,),
    ):
        sample_rows.append(ParagraphRow(*row))

    conn.close()
    return AlignmentSummary(
        paragraphs=paragraphs,
        anchors_by_size=anchors_by_size,
        ref_tokens=ref_tokens,
        asr_tokens=asr_tokens,
        sample=sample_rows,
    )


def _format_summary(summary: AlignmentSummary) -> str:
    lines = [
        f"Parrafos: {summary.paragraphs}",
        f"Tokens ref/ASR: {summary.ref_tokens}/{summary.asr_tokens}",
        "Anchors por tamano: " + (", ".join(f"{k}={v}" for k, v in sorted(summary.anchors_by_size.items(), reverse=True)) or "0"),
    ]
    if summary.sample:
        lines.append("Muestra de parrafos:")
        for p in summary.sample:
            lines.append(
                f"  #{p.id} ref[{p.ref_start}:{p.ref_end}] asr[{p.asr_start}:{p.asr_end}] "
                f"tc {p.tc_start:.2f}->{p.tc_end:.2f} wer={p.wer:.1f} flag={p.flag}"
            )
            lines.append(f"    REF: {p.ref_text}")
            lines.append(f"    ASR: {p.asr_text}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspeccion rapida de la base de alineacion.")
    parser.add_argument("db", help="Ruta al archivo SQLite generado por store_alignment_snapshot")
    parser.add_argument("--sample", type=int, default=5, help="Cantidad de parrafos a mostrar")
    args = parser.parse_args()
    summary = summarize_alignment(args.db, sample=args.sample)
    print(_format_summary(summary))


if __name__ == "__main__":  # pragma: no cover
    main()

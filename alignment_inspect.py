"""
Herramienta de inspeccion CLI para las bases de alineacion generadas por alignment.build_rows_from_words.

Funciones principales:
- listar filas de paragraph_rows
- mostrar una fila con sus textos y operaciones palabra-a-palabra
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, Optional


def _connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.exists():
        raise SystemExit(f"El archivo {path} no existe")
    return sqlite3.connect(path)


def list_rows(db_path: str | Path) -> None:
    con = _connect(db_path)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT row_id, paragraph_id, ref_start, ref_end, asr_start, asr_end, wer, flag "
        "FROM paragraph_rows ORDER BY row_id"
    ).fetchall()
    print("row_id | paragraph_id | ref_start-ref_end | asr_start-asr_end | wer | flag")
    for r in rows:
        row_id, pid, rs, re, hs, he, wer, flag = r
        print(f"{row_id:6d} | {pid:12d} | {rs:5d}-{re:<5d} | {hs:5d}-{he:<5d} | {wer:5.1f} | {flag}")
    con.close()


def _slice_tokens(cur: sqlite3.Cursor, table: str, start: int, end: int) -> list[str]:
    tokens = cur.execute(
        f"SELECT token_raw FROM {table} WHERE idx >= ? AND idx < ? ORDER BY idx",
        (start, end),
    ).fetchall()
    return [t[0] for t in tokens]


def _fetch_word_ops(cur: sqlite3.Cursor, ref_range: Tuple[int, int], asr_range: Tuple[int, int]) -> list[Tuple[Optional[int], Optional[int], str]]:
    rs, re = ref_range
    hs, he = asr_range
    rows = cur.execute(
        """
        SELECT ref_idx, asr_idx, op
        FROM word_alignment
        WHERE (ref_idx BETWEEN ? AND ? OR ref_idx IS NULL)
          AND (asr_idx BETWEEN ? AND ? OR asr_idx IS NULL)
        """,
        (rs, re - 1, hs, he - 1),
    ).fetchall()
    def _sort_key(row: Tuple[Optional[int], Optional[int], str]) -> Tuple[int, int]:
        ref_idx, asr_idx, _ = row
        key_ref = ref_idx if ref_idx is not None else (re + 1000000)
        key_asr = asr_idx if asr_idx is not None else (he + 1000000)
        return key_ref, key_asr
    return sorted(rows, key=_sort_key)


def show_row(db_path: str | Path, row_id: int) -> None:
    con = _connect(db_path)
    cur = con.cursor()
    row = cur.execute(
        "SELECT row_id, paragraph_id, ref_start, ref_end, asr_start, asr_end, wer, flag "
        "FROM paragraph_rows WHERE row_id = ?",
        (row_id,),
    ).fetchone()
    if not row:
        raise SystemExit(f"No existe row_id={row_id}")
    rid, pid, rs, re, hs, he, wer, flag = row
    ref_text = " ".join(_slice_tokens(cur, "ref_tokens", rs, re))
    asr_text = " ".join(_slice_tokens(cur, "asr_tokens", hs, he))
    print(f"row_id: {rid}  paragraph: {pid}")
    print(f"ref[{rs}:{re}] -> \"{ref_text}\"")
    print(f"asr[{hs}:{he}] -> \"{asr_text}\"")
    print(f"wer={wer:.1f} flag={flag}")
    print("ops (ref_idx, asr_idx, op):")
    ops = _fetch_word_ops(cur, (rs, re), (hs, he))
    for ref_idx, asr_idx, op in ops:
        print(f"  ({'' if ref_idx is None else ref_idx:>4}, {'' if asr_idx is None else asr_idx:>4}) {op}")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspector de alineaciones almacenadas en SQLite.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Listar filas de paragraph_rows")
    p_list.add_argument("db", help="Ruta al archivo .db generado por alignment")

    p_show = sub.add_parser("show", help="Mostrar una fila y sus operaciones palabra-a-palabra")
    p_show.add_argument("db", help="Ruta al archivo .db generado por alignment")
    p_show.add_argument("row_id", type=int, help="ID de fila a inspeccionar")

    args = parser.parse_args()
    if args.cmd == "list":
        list_rows(args.db)
    elif args.cmd == "show":
        show_row(args.db, args.row_id)


if __name__ == "__main__":  # pragma: no cover
    main()

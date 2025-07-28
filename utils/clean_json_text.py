from __future__ import annotations

"""Clean QC JSON rows removing punctuation from text columns."""

import json
import re
import sys
from pathlib import Path


def _clean_text(text: str) -> str:
    """Return ``text`` keeping only letters and spaces."""
    cleaned = ''.join(c for c in text if c.isalpha() or c.isspace())
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_rows(rows: list[list]) -> list[list]:
    """Modify ``rows`` in place cleaning Original and ASR columns."""
    for row in rows:
        if len(row) >= 2:
            row[-2] = _clean_text(str(row[-2]))
            row[-1] = _clean_text(str(row[-1]))
    return rows


def clean_file(json_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Clean ``json_path`` writing result to ``out_path`` or ``*.clean.json``."""
    p = Path(json_path)
    rows = json.loads(p.read_text(encoding="utf8"))
    clean_rows(rows)
    out = Path(out_path) if out_path else p.with_suffix(".clean.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
    return out


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Usage: clean_json_text.py input.json [output.json]")
        return
    out = clean_file(argv[0], argv[1] if len(argv) > 1 else None)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

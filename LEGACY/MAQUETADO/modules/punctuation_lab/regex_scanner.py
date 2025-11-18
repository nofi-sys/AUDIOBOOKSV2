#!/usr/bin/env python
"""
Herramienta experimental para detectar patrones de puntuación sospechosos en un archivo Markdown.

Uso:
    python modules/punctuation_lab/regex_scanner.py RUTA_ARCHIVO.md
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List


PATTERNS = {
    "comillas_dialogo": re.compile(r'(^|\s)"[^"]+"'),
    "doble_signo": re.compile(r"[!?]{2,}"),
    "rayas_mal_espaciadas": re.compile(r"(^|\s)--\s"),
}


def scan_lines(lines: Iterable[str]) -> List[str]:
    results: List[str] = []
    for idx, line in enumerate(lines, start=1):
        for name, regex in PATTERNS.items():
            if regex.search(line):
                results.append(f"{idx:05d}:{name}:{line.rstrip()}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Escáner de puntuación experimental.")
    parser.add_argument("path", type=Path, help="Ruta al archivo Markdown")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Archivo no encontrado: {args.path}")

    lines = args.path.read_text(encoding="utf-8").splitlines()
    hits = scan_lines(lines)
    if not hits:
        print("No se detectaron patrones sospechosos.")
        return
    print("Coincidencias encontradas:")
    for hit in hits:
        print(hit)


if __name__ == "__main__":
    main()

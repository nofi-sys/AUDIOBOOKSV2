from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def gather_inputs(inputs: List[str], glob_pattern: str | None = None, recursive: bool = False) -> Iterable[Path]:
    """Yield unique input files from paths, directories or glob expressions."""
    seen: set[Path] = set()
    cwd = Path.cwd()
    for item in inputs:
        path = Path(item)
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved
            continue
        if path.is_dir():
            pattern = glob_pattern or "*.txt"
            iterator = path.rglob(pattern) if recursive else path.glob(pattern)
            for candidate in iterator:
                if candidate.is_file():
                    resolved = candidate.resolve()
                    if resolved in seen:
                        continue
                    if glob_pattern is None and resolved.suffix.lower() != ".txt":
                        continue
                    seen.add(resolved)
                    yield resolved
            continue
        for candidate in cwd.glob(item):
            if candidate.is_file():
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                if glob_pattern is None and resolved.suffix.lower() != ".txt":
                    continue
                seen.add(resolved)
                yield resolved

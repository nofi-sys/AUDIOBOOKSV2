"""Utilities for applying configurable textual formatting markers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Callable

_RULE_CACHE: Dict[str, List[Dict[str, object]]] = {}

PROFILE_DIR = Path(__file__).resolve().parent / "format_rules"


def _compile_rule(rule: Dict[str, object]) -> Dict[str, object]:
    pattern = rule.get("pattern")
    if not pattern:
        raise ValueError("Cada regla debe incluir 'pattern'.")
    flags_value = 0
    for flag_name in rule.get("flags", []):
        flag_obj = getattr(re, flag_name.upper(), None)
        if flag_obj is None:
            raise ValueError(f"Bandera de regex desconocida: {flag_name}")
        flags_value |= flag_obj
    regex = re.compile(pattern, flags=flags_value)
    replacement = rule.get("replacement", "{match}")

    def _repl(match: re.Match[str]) -> str:
        groups = match.groupdict()
        if not groups:
            groups["match"] = match.group(0)
        else:
            groups.setdefault("match", match.group(0))
        groups.setdefault("content", groups.get("match", match.group(0)))
        return replacement.format(**groups)

    compiled: Dict[str, object] = dict(rule)
    compiled["_regex"] = regex
    compiled["_replacer"] = _repl
    return compiled


def _load_profile(profile: str) -> List[Dict[str, Any]]:
    if profile in _RULE_CACHE:
        return _RULE_CACHE[profile]
    path = PROFILE_DIR / f"{profile}.json"
    if not path.exists():
        raise FileNotFoundError(f"Perfil de formato no encontrado: {profile}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    patterns = data.get("patterns", [])
    compiled = [_compile_rule(rule) for rule in patterns]
    _RULE_CACHE[profile] = compiled
    return compiled


def apply_formatting(text: str, profile: Optional[str] = None) -> str:
    """Return text after applying formatting rules for the given profile."""
    profile_id = profile or "gutenberg"
    rules = _load_profile(profile_id)
    formatted = text
    for rule in rules:
        regex: re.Pattern[str] = rule["_regex"]  # type: ignore[index]
        repl: Callable[[re.Match[str]], str] = rule["_replacer"]  # type: ignore[index]
        formatted = regex.sub(repl, formatted)
    return formatted


def available_profiles() -> List[str]:
    profiles = []
    for path in PROFILE_DIR.glob("*.json"):
        stem = path.stem
        if stem == "formats":
            continue
        profiles.append(stem)
    profiles.sort()
    return profiles

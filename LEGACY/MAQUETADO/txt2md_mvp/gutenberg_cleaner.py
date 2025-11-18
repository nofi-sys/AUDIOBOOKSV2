import re
from typing import Optional, Tuple

_START_REGEX = re.compile(
    r"\*\*\*\s*(?:START|BEGIN)\s+OF\s+(?:THIS\s+|THE\s+)?PROJECT\s+GUTENBERG(?:'S)?\s+(?:E\s*-?\s*BOOK|E\s*-?\s*TEXT|FILE)\b[^\n]*",
    re.IGNORECASE,
)
_END_REGEX = re.compile(
    r"\*\*\*\s*END\s+OF\s+(?:THIS\s+|THE\s+)?PROJECT\s+GUTENBERG(?:'S)?\s+(?:E\s*-?\s*BOOK|E\s*-?\s*TEXT|FILE)\b[^\n]*",
    re.IGNORECASE,
)
_FALLBACK_START_REGEX = re.compile(
    r"^\s*THE\s+PROJECT\s+GUTENBERG\s+(?:E\s*-?\s*BOOK|E\s*-?\s*TEXT|FILE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_FALLBACK_END_REGEX = re.compile(
    r"^\s*END\s+OF\s+THE\s+PROJECT\s+GUTENBERG\s+(?:E\s*-?\s*BOOK|E\s*-?\s*TEXT|FILE)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _find_marker(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    match = pattern.search(text)
    if match:
        return match
    return None


START_PATTERNS = [
    re.compile(r"\*\*\*\s*START(?: OF)?(?: THE)? PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\*\s*START(?: OF)?(?: THE)? PROJECT GUTENBERG.*\*\*\*", re.IGNORECASE),
]
END_PATTERNS = [
    re.compile(r"\*\*\*\s*END(?: OF)?(?: THE)? PROJECT GUTENBERG EBOOK.*\*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\*\s*END(?: OF)?(?: THE)? PROJECT GUTENBERG.*\*\*\*", re.IGNORECASE),
]

FALLBACK_SKIP_KEYWORDS = (
    "project gutenberg",
    "gutenberg license",
    "online distributed proofread",
    "www.gutenberg",
    "gutenberg.org",
    "ebook",
    "e-book",
)


def _trim_leading_boilerplate(text: str) -> Tuple[str, bool]:
    lines = text.splitlines()
    trimmed = 0
    for idx, line in enumerate(lines):
        lower = line.lower()
        if trimmed == 0 and not lower.strip():
            # Ignore leading blank lines before boilerplate.
            continue
        if any(keyword in lower for keyword in FALLBACK_SKIP_KEYWORDS):
            trimmed = idx + 1
            continue
        if trimmed and not lower.strip():
            trimmed = idx + 1
            continue
        break
    if trimmed:
        remaining = "\n".join(lines[trimmed:])
        return remaining.lstrip("\n"), True
    return text, False


def clean_text(raw_text: str) -> str:
    """
    Cleans a string by removing Project Gutenberg headers and footers.
    Handles common variants such as 'THIS PROJECT GUTENBERG', missing spaces
    after the asterisks and legacy FILE/E-TEXT markers, and falls back to
    trimming boilerplate when explicit markers are absent.
    """
    working_text = raw_text
    start_pos: Optional[int] = None

    start_match = _find_marker(_START_REGEX, working_text) or _find_marker(_FALLBACK_START_REGEX, working_text)
    if start_match:
        start_pos = start_match.end()
    else:
        for pattern in START_PATTERNS:
            match = pattern.search(working_text)
            if match:
                start_pos = match.end()
                break

    if start_pos is None:
        working_text, trimmed = _trim_leading_boilerplate(working_text)
        if not trimmed:
            return raw_text
        start_pos = 0

    end_match = _find_marker(_END_REGEX, working_text) or _find_marker(_FALLBACK_END_REGEX, working_text)
    if end_match and end_match.start() > start_pos:
        end_pos = end_match.start()
    else:
        end_pos = None
        for pattern in END_PATTERNS:
            match = pattern.search(working_text)
            if match and match.start() > start_pos:
                end_pos = match.start()
                break
        if end_pos is None:
            end_pos = len(working_text)

    cleaned_text = working_text[start_pos:end_pos]
    return cleaned_text.strip()

"""Normalization engine for punctuation and graphic signs.

This module implements the algorithm detailed in
``docs/punctuation_and_graphic_signs_normalizer_algorithm.md``.  It is rule
driven, block aware and produces an auditable list of atomic changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import difflib
import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Iterable, TypedDict, Any, Callable, Set

# region: Constants and Enums
LANGUAGE_ES = "ES"
LANGUAGE_EN_US = "EN_US"
LANGUAGE_EN_UK = "EN_UK"

GENRE_NARRATIVE = "narrativa"
GENRE_ESSAY = "ensayo"
GENRE_TECH = "tecnico_academico"

NBSP = "\u00A0"
EM_DASH = "\u2014"
EN_DASH = "\u2013"
ELLIPSIS = "\u2026"
RIGHT_APOSTROPHE = "\u2019"

APOSTROPHE_CHARS = {"'", RIGHT_APOSTROPHE}
DOUBLE_QUOTE_CHARS = {'"', "\u201C", "\u201D", "\u00AB", "\u00BB"}
SINGLE_QUOTE_CHARS = {"'", "\u2018", "\u2019", "\u2039", "\u203A"}

class Severity(str, Enum):
    INFO = "info"
    SUGGESTION = "suggestion"
    FIX = "fix"

class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    DIALOGUE = "dialogue"
    QUOTE = "quote"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    SCENE_BREAK = "scene_break"
    FRONT_MATTER = "front_matter"
    HORIZONTAL_RULE = "horizontal_rule"
    OTHER = "other"

class Risk(Enum):
    SAFE = auto()
    RISKY = auto()
# endregion

# region: Data Structures
class NormalizerSettings(TypedDict, total=False):
    mode: str
    language: str
    genre: str
    threshold_safe: float
    threshold_all: float

@dataclass
class Change:
    rule_id: str
    description: str
    before: str
    after: str
    start_idx: int
    end_idx: int
    severity: str
    confidence: float
    notes: str = ""

@dataclass
class RuleResult:
    changes: List[Change]

@dataclass
class Rule:
    rule_id: str
    risk: Risk
    fn_plan: Callable[[str, Dict], RuleResult]

@dataclass
class NormalizationResult:
    normalized_text: str
    changes: List[Change] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

@dataclass
class Block:
    type: BlockType
    text: str
    start: int
    end: int
    meta: Dict[str, Any] = field(default_factory=dict)
# endregion

# region: Rule Definitions (placeholders)
def plan_ellipsis(text: str, settings: dict) -> RuleResult:
    changes = []
    for match in re.finditer(r"\.{3,}", text):
        changes.append(Change(
            rule_id="unicode.ellipsis",
            description="Convert multiple dots to ellipsis",
            before=match.group(0),
            after=ELLIPSIS,
            start_idx=match.start(),
            end_idx=match.end(),
            severity=Severity.FIX,
            confidence=1.0
        ))
    return RuleResult(changes=changes)


def plan_dialogue_dash(text: str, settings: dict) -> RuleResult:
    changes = []
    for match in re.finditer(r"^\s*-\s*(.*)", text, re.MULTILINE):
        content = match.group(1)
        needs_inverted_q = content.rstrip().endswith("?") and not content.lstrip().startswith("¿")
        needs_inverted_e = content.rstrip().endswith("!") and not content.lstrip().startswith("¡")

        new_content = content
        if needs_inverted_q:
            new_content = "¿" + new_content
        elif needs_inverted_e:
            new_content = "¡" + new_content

        after = f"{EM_DASH}{NBSP}{new_content}"

        changes.append(Change(
            rule_id="dialog.es.dash",
            description="Convert leading hyphen to em-dash and add inverted mark",
            before=match.group(0),
            after=after,
            start_idx=match.start(),
            end_idx=match.end(),
            severity=Severity.FIX,
            confidence=1.0
        ))
    return RuleResult(changes=changes)

def plan_numeric_ranges(text: str, settings: dict) -> RuleResult:
    changes = []
    for match in re.finditer(r"(\d+)\s*-\s*(\d+)", text):
        changes.append(Change(
            rule_id="range.en_dash",
            description="Normalize numeric range to en-dash",
            before=match.group(0),
            after=f"{match.group(1)}{EN_DASH}{match.group(2)}",
            start_idx=match.start(),
            end_idx=match.end(),
            severity=Severity.FIX,
            confidence=1.0
        ))
    return RuleResult(changes=changes)

def plan_quotes_balance(text: str, settings: dict) -> RuleResult:
    if settings.get("quotes_balance") is False:
        return RuleResult(changes=[])

    block_type = settings.get("__block_type")
    if block_type in (BlockType.HEADING, BlockType.QUOTE):
        return RuleResult(changes=[])

    lang = settings.get("language", LANGUAGE_ES)
    style = QUOTE_STYLES.get(lang, QUOTE_STYLES[LANGUAGE_ES])
    primary = style["primary"]
    secondary = style["secondary"]
    tertiary = style["tertiary"]

    changes: List[Change] = []
    stack: List[Tuple[str, str]] = []
    i = 0

    while i < len(text):
        ch = text[i]

        if (
            ch in APOSTROPHE_CHARS
            and i > 0
            and i + 1 < len(text)
            and text[i - 1].isalnum()
            and text[i + 1].isalnum()
        ):
            i += 1
            continue

        is_dbl = ch in DOUBLE_QUOTE_CHARS
        is_sgl = ch in SINGLE_QUOTE_CHARS
        if not (is_dbl or is_sgl):
            i += 1
            continue

        prev = text[i - 1] if i > 0 else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""

        opens = (i == 0 or prev.isspace() or prev in "([{-" or prev == EM_DASH)
        closes = (prev and not prev.isspace() and (nxt == "" or nxt.isspace() or nxt in ".,;:!?)]}"))

        if opens and not closes:
            if not stack:
                pair = primary if is_dbl else secondary
            elif stack[-1] == primary:
                pair = secondary
            else:
                pair = tertiary
            stack.append(pair)
            replacement = pair[0]
        else:
            pair = stack.pop() if stack else (primary if is_dbl else secondary)
            replacement = pair[1]

        if ch != replacement:
            changes.append(Change(
                rule_id="quotes.balance",
                description="Balance and style quotes",
                before=ch,
                after=replacement,
                start_idx=i,
                end_idx=i + 1,
                severity=Severity.FIX,
                confidence=0.85,
            ))

        i += 1

    return RuleResult(changes=changes)

def plan_dialog_es_to_emdash(text: str, settings: dict) -> RuleResult:
    """Convert dialogue enclosed in quotes into Spanish em-dash format."""
    lang = settings.get("language", LANGUAGE_ES)
    policy = (settings.get("dialogue_policy") or "").lower()

    if policy == "quotes_dialogue":
        return RuleResult(changes=[])
    if lang != LANGUAGE_ES and policy not in {"es_raya", "raya_dialogo"}:
        return RuleResult(changes=[])

    changes: List[Change] = []
    seen_spans: Set[Tuple[int, int]] = set()

    def _quote_range(open_ch: str, close_ch: str) -> str:
        return rf"{re.escape(open_ch)}(?P<seg>[^{re.escape(close_ch)}]+){re.escape(close_ch)}"

    quote_pairs = [
        ("\u00AB", "\u00BB"),      # Spanish guillemets
        ("\u201C", "\u201D"),      # Double smart quotes
        ("\u2018", "\u2019"),      # Single smart quotes
        ("\u0022", "\u0022"),             # ASCII double quotes
    ]

    verb_pattern = r"(?P<verb>[^\s,;:\u00AB\u00BB\u201C\u201D\u2018\u2019\"\u2026]+)"


    def build_pattern(open_ch: str, close_ch: str, follow: str) -> re.Pattern:
        inner = rf"(?P<a>[^{re.escape(close_ch)}]+)"
        inner_b = rf"(?P<b>[^{re.escape(close_ch)}]+)"
        open_pat = re.escape(open_ch)
        close_pat = re.escape(close_ch)
        return re.compile(
            rf"{open_pat}{inner}{close_pat}\s*,\s*{verb_pattern}\s*(?P<who>{follow})\s*,\s*{open_pat}{inner_b}{close_pat}",
            re.IGNORECASE,
        )

    def build_simple_pattern(open_ch: str, close_ch: str, follow: str) -> re.Pattern:
        inner = rf"(?P<a>[^{re.escape(close_ch)}]+)"
        open_pat = re.escape(open_ch)
        close_pat = re.escape(close_ch)
        return re.compile(
            rf"{open_pat}{inner}{close_pat}\s*,\s*{verb_pattern}\s*(?P<who>{follow})\.(?=\s|$)",
            re.IGNORECASE,
        )

    follow_chars = r"[^,\u00AB\u00BB\u201C\u201D\u2018\u2019\"\u2026]+"


    def _normalize_segment(seg: str) -> str:
        return _strip_trailing_comma(seg.strip())

    for open_ch, close_ch in quote_pairs:
        inciso_pat = build_pattern(open_ch, close_ch, follow_chars)
        for match in inciso_pat.finditer(text):
            a = _normalize_segment(match.group("a"))
            b = _normalize_segment(match.group("b"))
            verb = match.group("verb").strip()
            who = match.group("who").strip()
            if not _is_speech_verb(verb):
                continue
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            after = f"{EM_DASH}{NBSP}{a} {EM_DASH}{verb} {who}{EM_DASH} {b}"
            after = re.sub(r"\s{2,}", " ", after)
            changes.append(Change(
                rule_id="dialog.es.raya.inciso",
                description="Transform quoted dialogue with inciso to em-dash form",
                before=match.group(0),
                after=after,
                start_idx=match.start(),
                end_idx=match.end(),
                severity=Severity.FIX,
                confidence=0.95,
            ))
            seen_spans.add(span)

        simple_pat = build_simple_pattern(open_ch, close_ch, follow_chars)
        for match in simple_pat.finditer(text):
            a = _normalize_segment(match.group("a"))
            verb = match.group("verb").strip()
            who = match.group("who").strip()
            if not _is_speech_verb(verb):
                continue
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            suffix = "."
            after = f"{EM_DASH}{NBSP}{a} {EM_DASH}{verb} {who}{suffix}"
            after = re.sub(r"\s{2,}", " ", after)
            changes.append(Change(
                rule_id="dialog.es.raya",
                description="Transform quoted dialogue with tag to em-dash form",
                before=match.group(0),
                after=after,
                start_idx=match.start(),
                end_idx=match.end(),
                severity=Severity.FIX,
                confidence=0.92,
            ))
            seen_spans.add(span)

        tagless_pat = re.compile(
            rf"{re.escape(open_ch)}(?P<a>[^{re.escape(close_ch)}]+){re.escape(close_ch)}\s*,\s*{verb_pattern}\.(?=\s|$)",
            re.IGNORECASE,
        )
        for match_tagless in tagless_pat.finditer(text):
            a = _normalize_segment(match_tagless.group("a"))
            verb = match_tagless.group("verb").strip()
            if not _is_speech_verb(verb):
                continue
            span_tagless = (match_tagless.start(), match_tagless.end())
            if span_tagless in seen_spans:
                continue
            after = f"{EM_DASH}{NBSP}{a} {EM_DASH}{verb}."
            after = re.sub(r"\s{2,}", " ", after)
            changes.append(Change(
                rule_id="dialog.es.raya.tagless",
                description="Transform quoted dialogue with implicit speaker to em-dash form",
                before=match_tagless.group(0),
                after=after,
                start_idx=match_tagless.start(),
                end_idx=match_tagless.end(),
                severity=Severity.FIX,
                confidence=0.9,
            ))
            seen_spans.add(span_tagless)

        pure_pat = re.compile(
            rf"^\s*{re.escape(open_ch)}(?P<a>[^{re.escape(close_ch)}]+){re.escape(close_ch)}\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in pure_pat.finditer(text):
            a = match.group("a").strip()
            if not a:
                continue
            emotive = bool(re.search(r"[\u00A1\u00BF!?]", a))
            conf = 0.85 if emotive else 0.6
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            after = f"{EM_DASH}{NBSP}{a}"
            changes.append(Change(
                rule_id="dialog.es.raya.no_verb",
                description="Transform standalone quoted dialogue to em-dash form",
                before=match.group(0),
                after=after,
                start_idx=match.start(),
                end_idx=match.end(),
                severity=Severity.FIX,
                confidence=conf,
            ))
            seen_spans.add(span)

        leading_pat = re.compile(
            rf"(?m)^(?P<indent>\s*){re.escape(open_ch)}(?P<a>[^{re.escape(close_ch)}]+){re.escape(close_ch)}(?P<post>\s*)"
        )
        for match in leading_pat.finditer(text):
            span = (match.start(), match.end())
            if span in seen_spans:
                continue
            indent = match.group("indent")
            a = match.group("a").strip()
            post = match.group("post")
            if not a:
                continue
            next_idx = _first_nonspace(text, match.end(), len(text))
            if next_idx is not None and text[next_idx] == ",":
                continue
            emotive = bool(re.search(r"[\u00A1\u00BF!?]", a))
            conf = 0.8 if emotive else 0.55
            after = f"{indent}{EM_DASH}{NBSP}{a}{post}"
            changes.append(Change(
                rule_id="dialog.es.raya.no_verb",
                description="Transform leading quoted dialogue to em-dash form",
                before=match.group(0),
                after=after,
                start_idx=match.start(),
                end_idx=match.end(),
                severity=Severity.FIX,
                confidence=conf,
            ))
            seen_spans.add(span)

        mid_pat = re.compile(
            rf"(?P<prefix>[\.!\?\u2026]\s*){re.escape(open_ch)}(?P<a>[^{re.escape(close_ch)}]+){re.escape(close_ch)}"
        )
        for match in mid_pat.finditer(text):
            prefix = match.group("prefix")
            start = match.start() + len(prefix)
            end = match.end()
            span = (start, end)
            if span in seen_spans:
                continue
            a = match.group("a").strip()
            if not a:
                continue
            next_idx = _first_nonspace(text, end, len(text))
            if next_idx is not None and text[next_idx] == ",":
                continue
            heuristic_signal = re.search(r"[\u00A1\u00BF!?]", a) or len(a.split()) > 1
            conf = 0.75 if heuristic_signal else 0.55
            after = f"{EM_DASH}{NBSP}{a}"
            before = text[start:end]
            changes.append(Change(
                rule_id="dialog.es.raya.no_verb",
                description="Transform mid-sentence quoted dialogue to em-dash form",
                before=before,
                after=after,
                start_idx=start,
                end_idx=end,
                severity=Severity.FIX,
                confidence=conf,
            ))
            seen_spans.add(span)

    return RuleResult(changes=changes)


def plan_dialogue_dash_spacing(text: str, settings: dict) -> RuleResult:
    if settings.get("language") != LANGUAGE_ES:
        return RuleResult(changes=[])

    block_type = settings.get("__block_type")
    if block_type in (BlockType.CODE, BlockType.TABLE):
        return RuleResult(changes=[])

    changes: List[Change] = []

    line_start_pattern = re.compile(r"(?m)^(?P<indent>\s*)\u2014(?P<gap>[ \t\u00A0]*)")
    for match in line_start_pattern.finditer(text):
        gap = match.group("gap")
        if gap == NBSP:
            continue
        dash_idx = match.start() + len(match.group("indent"))
        gap_end = dash_idx + len(EM_DASH) + len(gap)
        before = text[dash_idx:gap_end]
        after = EM_DASH + NBSP
        if before == after:
            continue
        changes.append(Change(
            rule_id="dialog.es.dash_spacing",
            description="Normalize NBSP after dialogue dash",
            before=before,
            after=after,
            start_idx=dash_idx,
            end_idx=gap_end,
            severity=Severity.FIX,
            confidence=0.85,
        ))

    def _is_dialogue_lead(char: str) -> bool:
        if not char:
            return False
        if char in {"\u00A1", "\u00BF", "\u201C", "\u00AB", "\"", "'"}:
            return True
        return char.isupper()

    mid_start_pattern = re.compile(
        rf"(?P<context>[\.\?!\u2026\"\u201D\u00BB]\s+)(?P<dash>{EM_DASH})(?P<gap>[ \t]*)(?P<lead>\S)"
    )
    for match in mid_start_pattern.finditer(text):
        lead = match.group("lead")
        if not _is_dialogue_lead(lead):
            continue
        gap = match.group("gap")
        if gap == NBSP:
            continue
        dash_idx = match.start("dash")
        lead_idx = match.start("lead")
        before = text[dash_idx:lead_idx + 1]
        after = f"{EM_DASH}{NBSP}{lead}"
        changes.append(Change(
            rule_id="dialog.es.dash_spacing",
            description="Ensure NBSP after dialogue dash mid-paragraph",
            before=before,
            after=after,
            start_idx=dash_idx,
            end_idx=lead_idx + 1,
            severity=Severity.FIX,
            confidence=0.8,
        ))

    cleanup_pattern = re.compile(r"\u2014[ \t]*\u2014")
    for match in cleanup_pattern.finditer(text):
        start_idx, end_idx = match.span()
        line_start = text.rfind("\n", 0, start_idx) + 1
        if start_idx == line_start:
            replacement = EM_DASH + NBSP
        else:
            replacement = EM_DASH + " "
        before = text[start_idx:end_idx]
        if before == replacement:
            continue
        changes.append(Change(
            rule_id="cleanup.double_em_dash",
            description="Collapse duplicated em dashes",
            before=before,
            after=replacement,
            start_idx=start_idx,
            end_idx=end_idx,
            severity=Severity.FIX,
            confidence=0.9,
        ))

    return RuleResult(changes=changes)

def plan_fix_inverted_dash(text: str, settings: dict) -> RuleResult:
    """Move misplaced inverted marks that appear before a dialogue dash."""
    if settings.get("language") != LANGUAGE_ES:
        return RuleResult(changes=[])

    block_type = settings.get("__block_type")
    if block_type in (BlockType.CODE, BlockType.TABLE):
        return RuleResult(changes=[])

    dash_chars = "-\u2010\u2011\u2012\u2013\u2014\u2015"
    pattern = re.compile(
        rf"(?m)^(?P<indent>\s*)(?P<mark>[¡¿])\s*[{dash_chars}]\s*(?P<rest>[¡¿][^\n]*)"
    )

    changes: List[Change] = []
    for match in pattern.finditer(text):
        indent = match.group("indent")
        rest = match.group("rest").lstrip()
        if not rest:
            continue
        after = f"{indent}{EM_DASH}{NBSP}{rest}"
        changes.append(Change(
            rule_id="dialog.es.fix_inverted_dash",
            description="Move dialogue dash ahead of inverted mark",
            before=match.group(0),
            after=after,
            start_idx=match.start(),
            end_idx=match.end(),
            severity=Severity.FIX,
            confidence=0.9,
        ))

    return RuleResult(changes=changes)


def plan_inverted_marks(text: str, settings: dict) -> RuleResult:
    if settings.get("language") != LANGUAGE_ES:
        return RuleResult(changes=[])

    block_type = settings.get("__block_type")
    if block_type in (BlockType.HEADING, BlockType.CODE, BlockType.TABLE):
        return RuleResult(changes=[])

    style = QUOTE_STYLES[LANGUAGE_ES]
    spans = list(_iter_quote_spans(text, style["primary"], style["secondary"], style["tertiary"]))
    changes: List[Change] = []

    needed = {
        "!": max(0, text.count("!") - text.count("¡")),
        "?": max(0, text.count("?") - text.count("¿")),
    }

    for match in re.finditer(r"[?!]", text):
        end_idx = match.start()
        closing_char = text[end_idx]
        target = "¿" if closing_char == "?" else "¡"

        if needed.get(closing_char, 0) <= 0:
            continue

        inside_span: Optional[Tuple[int, int]] = None
        for open_idx, close_idx in spans:
            if open_idx < end_idx < close_idx:
                inside_span = (open_idx, close_idx)
                break

        if inside_span:
            open_idx, close_idx = inside_span
            if target in text[open_idx:end_idx]:
                continue
            insert_at = _first_nonspace(text, open_idx + 1, close_idx)
            if insert_at is None:
                continue
            if text[insert_at] in ("¿", "¡"):
                continue
            before = text[insert_at]
            after = target + before
            changes.append(Change(
                rule_id="punctuation.es.inverted_marks",
                description="Insert inverted mark inside opening quote",
                before=before,
                after=after,
                start_idx=insert_at,
                end_idx=insert_at + 1,
                severity=Severity.FIX,
                confidence=0.9,
            ))
            needed[closing_char] -= 1
            continue

        line_start = text.rfind("\n", 0, end_idx) + 1
        if text.startswith(EM_DASH, line_start):
            after_dash = _first_nonspace(text, line_start + len(EM_DASH), end_idx + 1)
            if after_dash is not None and text[after_dash] == target:
                continue
        dash_pos = _after_dialog_dash_nbsp(text, line_start)
        if dash_pos is not None:
            if target in text[line_start:end_idx]:
                continue
            insert_at = _first_nonspace(text, dash_pos, end_idx + 1)
            if insert_at is not None and text[insert_at] not in ("¿", "¡"):
                before = text[insert_at]
                after = target + before
                changes.append(Change(
                    rule_id="punctuation.es.inverted_marks",
                    description="Insert inverted mark after dialogue em-dash",
                    before=before,
                    after=after,
                    start_idx=insert_at,
                    end_idx=insert_at + 1,
                    severity=Severity.FIX,
                    confidence=0.85,
                ))
                needed[closing_char] -= 1
            continue

        mid_dash = text.rfind(EM_DASH, line_start, end_idx)
        if mid_dash != -1:
            ins_at = _first_nonspace(text, mid_dash + 1, end_idx + 1)
            if ins_at is None:
                pass
            elif text[ins_at] in ("¿", "¡"):
                continue
            else:
                before = text[ins_at]
                after = target + before
                changes.append(Change(
                    rule_id="punctuation.es.inverted_marks",
                    description="Insert inverted mark after mid-sentence em-dash",
                    before=before,
                    after=after,
                    start_idx=ins_at,
                    end_idx=ins_at + 1,
                    severity=Severity.FIX,
                    confidence=0.8,
                ))
                needed[closing_char] -= 1
                continue

        sentence_start = max(
            text.rfind(".", 0, end_idx),
            text.rfind("!", 0, end_idx),
            text.rfind("?", 0, end_idx),
            text.rfind("\n", 0, end_idx),
        ) + 1
        insert_at = _first_nonspace(text, sentence_start, end_idx + 1)
        if insert_at is None:
            continue
        if text[insert_at] in ("¿", "¡"):
            continue
        if text[insert_at] == EM_DASH:
            continue
        if target in text[sentence_start:end_idx]:
            continue
        before = text[insert_at]
        after = target + before
        changes.append(Change(
            rule_id="punctuation.es.inverted_marks",
            description="Insert inverted mark at sentence start",
            before=before,
            after=after,
            start_idx=insert_at,
            end_idx=insert_at + 1,
            severity=Severity.FIX,
            confidence=0.75,
        ))
        needed[closing_char] -= 1

    return RuleResult(changes=changes)





RULES: List[Rule] = [
    Rule("unicode.ellipsis", Risk.SAFE, plan_ellipsis),
    Rule("range.en_dash", Risk.SAFE, plan_numeric_ranges),
    Rule("quotes.balance", Risk.SAFE, plan_quotes_balance),
    Rule("dialog.es.raya", Risk.RISKY, plan_dialog_es_to_emdash),
    Rule("dialog.es.dash_spacing", Risk.SAFE, plan_dialogue_dash_spacing),
    Rule("dialog.es.fix_inverted_dash", Risk.RISKY, plan_fix_inverted_dash),
    Rule("punctuation.es.inverted_marks", Risk.RISKY, plan_inverted_marks),
    Rule("dialog.es.dash", Risk.SAFE, plan_dialogue_dash),
]

PRIORITY = {
    "dialog.es.dash": 110,
    "dialog.es.raya": 100,
    "dialog.es.raya.inciso": 100,
    "dialog.es.raya.no_verb": 100,
    "dialog.es.raya.tagless": 100,
    "dialog.es.fix_inverted_dash": 95,
    "cleanup.double_em_dash": 93,
    "dialog.es.dash_spacing": 94,
    "unicode.ellipsis": 90,
    "range.en_dash": 70,
    "quotes.balance": 60,
    "punctuation.es.inverted_marks": 50,
}

CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
SCENE_BREAK_PATTERNS = {"***", "* * *", "---", "\u2014\u2014\u2014", "###"}
LIST_BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]|[0-9]+[.)])\s+")
DIALOGUE_DASH_RE = re.compile(r"^\s*[-\u2013\u2014]")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")

ES_SPEECH_VERBS = (
    "dijo|pregunt[o\u00F3]|respond[i\u00ED]\u00F3|exclam[o\u00F3]|murmur[o\u00F3]|grit[o\u00F3]|susurr[o\u00F3]|"
    "a\u00F1ad[i\u00ED]\u00F3|replic[o\u00F3]|contest[o\u00F3]|llam[o\u00F3]|orden[o\u00F3]|se\u00F1al[o\u00F3]|explic[o\u00F3]|"
    "advirti[o\u00F3]|observ[o\u00F3]|indic[o\u00F3]|coment[o\u00F3]|apunt[o\u00F3]|insist[i\u00ED]\u00F3|ri\u00F3|sonri\u00F3"
)

ES_SPEECH_VERBS_RE = re.compile(rf"\b(?:{ES_SPEECH_VERBS})\b", re.IGNORECASE)

ES_SPEECH_VERB_STEMS = {
    "dij", "pregunt", "respond", "exclam", "murmur", "grit", "susurr",
    "a\u00F1ad", "anad", "replic", "contest", "llam", "orden", "se\u00F1al", "senal",
    "explic", "advirti", "observ", "indic", "coment", "apunt", "insist",
    "rio", "sonrio"
}

QUOTE_STYLES = {
    LANGUAGE_ES: {
        "primary": ("\u00AB", "\u00BB"),
        "secondary": ("\u201C", "\u201D"),
        "tertiary": ("\u2018", "\u2019"),
    },
    LANGUAGE_EN_US: {
        "primary": ("\u201C", "\u201D"),
        "secondary": ("\u2018", "\u2019"),
        "tertiary": ("\u2039", "\u203A"),
    },
    LANGUAGE_EN_UK: {
        "primary": ("\u2018", "\u2019"),
        "secondary": ("\u201C", "\u201D"),
        "tertiary": ("\u2039", "\u203A"),
    },
}
def _iter_quote_spans(text: str, primary: Tuple[str, str], secondary: Tuple[str, str], tertiary: Tuple[str, str]) -> Iterable[Tuple[int, int]]:
    opens = {primary[0], secondary[0], tertiary[0]}
    closes = {primary[1], secondary[1], tertiary[1]}
    stack: List[Tuple[str, int]] = []
    for idx, char in enumerate(text):
        if char in opens:
            stack.append((char, idx))
        elif char in closes and stack:
            _, open_idx = stack.pop()
            yield (open_idx, idx)


def _first_nonspace(text: str, start: int, end: int) -> Optional[int]:
    pos = start
    while pos < end and text[pos].isspace():
        pos += 1
    return pos if pos < end else None


def _after_dialog_dash_nbsp(text: str, line_start: int) -> Optional[int]:
    """Return the index immediately after an em dash + NBSP at the line start."""
    if text.startswith(EM_DASH + NBSP, line_start):
        return line_start + len(EM_DASH + NBSP)
    return None


def _strip_trailing_comma(segment: str) -> str:
    return segment[:-1] if segment.endswith(",") else segment


def _ensure_space_after(segment: str) -> str:
    return segment if not segment or segment[-1].isspace() else segment + " "

def _is_speech_verb(token: str) -> bool:
    clean = token.strip().strip('.,;:!?"\'»«')
    if not clean:
        return False
    if ES_SPEECH_VERBS_RE.search(clean):
        return True
    normalized = ''.join(ch for ch in unicodedata.normalize('NFD', clean.lower()) if not unicodedata.combining(ch))
    normalized = normalized.replace('ñ', 'n')
    for stem in ES_SPEECH_VERB_STEMS:
        if normalized.startswith(stem):
            return True
    return False

def _preprocess_blocks(text: str) -> List[Block]:
    blocks: List[Block] = []
    lines = text.splitlines(keepends=True)
    buffer: List[str] = []
    buffer_start: Optional[int] = None
    cursor = 0
    in_code_fence = False

    def flush_buffer(end_idx: int) -> None:
        nonlocal buffer, buffer_start
        if not buffer:
            return
        start_idx = buffer_start if buffer_start is not None else end_idx - sum(len(item) for item in buffer)
        block_text = ''.join(buffer)
        block_type = _guess_buffer_type(buffer)
        blocks.append(Block(type=block_type, text=block_text, start=start_idx, end=end_idx))
        buffer = []
        buffer_start = None

    for raw_line in lines:
        line_start = cursor
        cursor += len(raw_line)
        stripped = raw_line.strip()

        if CODE_FENCE_RE.match(raw_line):
            flush_buffer(line_start)
            in_code_fence = not in_code_fence
            blocks.append(Block(type=BlockType.CODE, text=raw_line, start=line_start, end=cursor))
            continue

        if in_code_fence:
            blocks.append(Block(type=BlockType.CODE, text=raw_line, start=line_start, end=cursor))
            continue

        if not stripped:
            flush_buffer(line_start)
            blocks.append(Block(type=BlockType.OTHER, text=raw_line, start=line_start, end=cursor, meta={"blank": True}))
            continue

        if buffer_start is None:
            buffer_start = line_start
        buffer.append(raw_line)

    flush_buffer(cursor)
    return blocks


def _guess_buffer_type(buffer: List[str]) -> BlockType:
    stripped = [line.strip() for line in buffer if line.strip()]
    if not stripped:
        return BlockType.OTHER
    if all(HEADING_RE.match(line) for line in stripped):
        return BlockType.HEADING
    if all(DIALOGUE_DASH_RE.match(line) for line in stripped):
        return BlockType.DIALOGUE
    if all(LIST_BULLET_RE.match(line) for line in stripped):
        return BlockType.LIST
    if all(line.startswith(">") for line in stripped):
        return BlockType.QUOTE
    if all(TABLE_ROW_RE.match(line) for line in stripped):
        return BlockType.TABLE
    if stripped[0] in SCENE_BREAK_PATTERNS:
        return BlockType.SCENE_BREAK
    return BlockType.PARAGRAPH


# region: Plan and Apply Logic
def build_plan(blocks: List[Block], settings: dict) -> List[Change]:
    plan: List[Change] = []
    for block in blocks:
        if block.type == BlockType.CODE:
            continue
        settings["__block_type"] = block.type
        settings["__block_start"] = block.start
        settings["__block_text"] = block.text
        for rule in RULES:
            rr = rule.fn_plan(block.text, settings)
            for ch in rr.changes:
                ch.severity = "suggestion"
                ch.start_idx += block.start
                ch.end_idx += block.start
            plan.extend(rr.changes)
        settings.pop("__block_type", None)
        settings.pop("__block_start", None)
        settings.pop("__block_text", None)

    plan = resolve_conflicts(plan, settings)
    return plan

def resolve_conflicts(changes: List[Change], settings: dict) -> List[Change]:
    out = []
    occupied = []
    def conflict_key(change: Change) -> Tuple[int, float, float]:
        priority = PRIORITY.get(change.rule_id, 0)
        return (change.start_idx, -priority, -change.confidence)

    for ch in sorted(changes, key=conflict_key):
        if any(not (ch.end_idx <= s or ch.start_idx >= e) for (s, e) in occupied):
            continue
        occupied.append((ch.start_idx, ch.end_idx))
        out.append(ch)
    return out

def filter_by_mode(plan: List[Change], mode: str, settings: dict) -> List[Change]:
    t_safe = settings.get("threshold_safe", 0.8)
    t_all = settings.get("threshold_all", 0.4)

    def is_safe(rule_id):
        return any(r.rule_id == rule_id and r.risk == Risk.SAFE for r in RULES)

    if mode == "scan-only":
        for ch in plan: ch.severity = "suggestion"
        return []

    if mode == "fix-safe":
        applied = []
        for ch in plan:
            if is_safe(ch.rule_id) and ch.confidence >= t_safe:
                ch.severity = "fix"
                applied.append(ch)
            else:
                ch.severity = "suggestion"
        return applied

    if mode == "fix-all":
        for ch in plan:
            ch.severity = "fix" if ch.confidence >= t_all else "suggestion"
        return [c for c in plan if c.severity == "fix"]

    raise ValueError(f"Unknown mode: {mode}")

def apply_plan(text: str, plan: List[Change]) -> str:
    # We need to apply changes in reverse order to not mess up the indices
    for ch in sorted(plan, key=lambda c: c.start_idx, reverse=True):
        text = text[:ch.start_idx] + ch.after + text[ch.end_idx:]
    return text

def aggregate_stats(plan: List[Change], applied: List[Change]) -> dict:
    stats = {
        "proposed_changes": len(plan),
        "applied_changes": len(applied),
        "dialogue_blocks_fixed": 0,
        "inverted_marks_added_or_removed": 0,
        "ranges_normalized": 0,
    }
    for ch in applied:
        stats[ch.rule_id] = stats.get(ch.rule_id, 0) + 1
        if ch.rule_id.startswith("dialog."):
            stats["dialogue_blocks_fixed"] += 1
        if ch.rule_id == "punctuation.es.inverted_marks" or (ch.rule_id == "dialog.es.dash" and ("¿" in ch.after or "¡" in ch.after)):
            stats["inverted_marks_added_or_removed"] += 1
        if ch.rule_id == "range.en_dash":
            stats["ranges_normalized"] += 1
    return stats
def to_utf16_offsets(text: str, start_char_idx: int, end_char_idx: int) -> Tuple[int, int]:
    """Converts character offsets to UTF-16 code unit offsets."""
    prefix = text[:start_char_idx]
    span = text[start_char_idx:end_char_idx]
    utf16_start = len(prefix.encode('utf-16-le')) // 2
    utf16_end = utf16_start + len(span.encode('utf-16-le')) // 2
    return utf16_start, utf16_end
# endregion

# region: Main Entry Point
def normalize_punctuation(text: str, settings: NormalizerSettings) -> NormalizationResult:
    """Public entry point for punctuation normalization."""
    mode = settings.get("mode", "fix-all")

    blocks = _preprocess_blocks(text)
    plan = build_plan(blocks, settings)
    to_apply = filter_by_mode(plan, mode, settings)

    normalized_text = apply_plan(text, to_apply)

    stats = aggregate_stats(plan, to_apply)

    # Here you would add UTF-16 offset conversion if needed

    return NormalizationResult(
        normalized_text=normalized_text,
        changes=plan,
        stats=stats,
    )
# endregion


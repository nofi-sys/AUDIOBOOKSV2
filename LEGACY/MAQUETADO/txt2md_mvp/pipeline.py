from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json, os, re, datetime, copy, subprocess, tempfile, shutil
try:
    from .gutenberg_cleaner import clean_text
    from .utils import (
        normalize_text,
        strip_bom,
        clean_line,
        collapse_soft_linebreaks,
        is_uppercase_segment,
        titlecase_hint,
        detect_subtitle,
        is_title_like_for_merge,
    )
    from .rules import match_patterns, detect_indentation_pattern
    from .hierarchy import enforce
    from .render import render
    from .ai_supervisor import supervise_heading
    from .index_finder import IndexFinder
    from .pattern_learner import PatternLearner
    from .md2docx.core import Block, build_document, configure_headers, add_page_numbers, DEFAULT_STYLESET, DEFAULT_PAGE
    from .punctuation_normalizer import (
        normalize_punctuation,
        NormalizerSettings,
        NormalizationResult,
        LANGUAGE_ES,
        LANGUAGE_EN_US,
        GENRE_NARRATIVE,
    )
except ImportError:  # pragma: no cover - soporte para ejecución directa
    from gutenberg_cleaner import clean_text  # type: ignore
    from utils import (
        normalize_text,
        strip_bom,
        clean_line,
        collapse_soft_linebreaks,
        is_uppercase_segment,
        titlecase_hint,
        detect_subtitle,
        is_title_like_for_merge,
    )  # type: ignore
    from rules import match_patterns, detect_indentation_pattern  # type: ignore
    from hierarchy import enforce  # type: ignore
    from render import render  # type: ignore
    from ai_supervisor import supervise_heading # type: ignore
    from index_finder import IndexFinder # type: ignore
    from pattern_learner import PatternLearner # type: ignore
    from md2docx.core import Block, build_document, configure_headers, add_page_numbers, DEFAULT_STYLESET, DEFAULT_PAGE # type: ignore
    from punctuation_normalizer import (  # type: ignore
        normalize_punctuation,
        NormalizerSettings,
        NormalizationResult,
        LANGUAGE_ES,
        LANGUAGE_EN_US,
        GENRE_NARRATIVE,
    )

AI_SUPERVISION_THRESHOLD = 0.7
HUMAN_SUPERVISION_THRESHOLD_LOW = 0.50
HUMAN_SUPERVISION_THRESHOLD_HIGH = 0.85

AUTO_REFRESH_FIELDS = os.environ.get("TXT2MD_AUTO_REFRESH_FIELDS", "").lower() in ("1", "true", "yes")
LIBREOFFICE_BIN = os.environ.get("TXT2MD_LIBREOFFICE_BIN") or shutil.which("soffice")


def _auto_refresh_docx_fields(docx_path: str) -> None:
    if not AUTO_REFRESH_FIELDS or not LIBREOFFICE_BIN:
        return
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [LIBREOFFICE_BIN, "--headless", "--convert-to", "docx", "--outdir", tmpdir, docx_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode != 0:
                return
            generated = os.path.join(tmpdir, os.path.basename(docx_path))
            if os.path.exists(generated):
                shutil.move(generated, docx_path)
    except Exception:
        return

def _read_file_with_fallback(in_path: str) -> Tuple[str, str]:
    """Return decoded text and the encoding that succeeded."""
    tried_encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    for encoding in tried_encodings:
        try:
            with open(in_path, "r", encoding=encoding) as fh:
                return fh.read(), encoding
        except UnicodeDecodeError:
            continue
    # Last resort: replace undecodable bytes but avoid dropping information entirely
    with open(in_path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(), "utf-8"


def _merge_heading_wrapped_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    idx = 0
    total = len(lines)
    while idx < total:
        line = lines[idx]
        if idx + 1 < total:
            current = line.strip()
            nxt = lines[idx + 1].strip()
            if current and nxt and current.endswith('-') and is_uppercase_segment(current[:-1]) and is_uppercase_segment(nxt):
                prefix = line.rstrip()
                merged_line = prefix + lines[idx + 1].lstrip()
                merged.append(merged_line)
                idx += 2
                continue
        merged.append(line)
        idx += 1
    return merged


def _get_marker_role(text: str) -> Optional[str]:
    """Determines if a text is a chapter-like or container-like marker."""
    text = text.lower()
    if any(marker in text for marker in ("chapter", "capítulo", "section", "sección")):
        return "chapter_marker"
    if any(marker in text for marker in ("part", "parte", "book", "libro")):
        return "container_marker"
    return None

def _heading_remainder_attaches(lines: List[str]) -> bool:
    meaningful = [ln.strip() for ln in lines if ln.strip()]
    if not meaningful:
        return False
    for item in meaningful:
        if any(ch in item for ch in ('.', '!', '?', ':')):
            return False
        words = item.split()
        if len(words) > 6:
            return False
        if not (is_uppercase_segment(item) or titlecase_hint(item)):
            return False
    return True


def _segment_into_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current: List[str] = []
    start_idx: Optional[int] = None
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if start_idx is None:
                start_idx = idx
            current.append(line)
        else:
            if current:
                blocks.append({"lines": current, "start": start_idx})
                current = []
                start_idx = None
    if current:
        blocks.append({"lines": current, "start": start_idx})
    return blocks


from typing import List, Dict, Any, Tuple, Optional, Callable

FRONT_MATTER_MAX_SCAN = 80
FRONT_MATTER_SKIP_KEYWORDS = (
    "project gutenberg",
    "gutenberg",
    "ebook",
    "license",
    "transcriber's note",
    "distributed proofread",
    "prepared by",
    "***start",
)

ROMAN_NUMERALS = {
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
    "xiii",
    "xiv",
    "xv",
}

LETTER_SALUTATIONS = (
    "dear ",
    "dearest",
    "my dear",
    "querido",
    "querida",
    "mis queridos",
    "estimado",
    "estimada",
)

LETTER_CLOSINGS = (
    "sincerely",
    "yours",
    "affectionately",
    "cordially",
    "faithfully",
    "atentamente",
    "afectuosamente",
    "tu amigo",
    "your friend",
    "lovingly",
    "gratefully",
)


def _normalize_capitalization(text: str) -> str:
    parts = re.split(r"(\W+)", text.strip().lower())
    normalized: List[str] = []
    for part in parts:
        if not part or not part.strip():
            normalized.append(part)
            continue
        if not part.isalpha():
            normalized.append(part)
            continue
        if part.upper() in ROMAN_NUMERALS or (len(part) <= 3 and part.isupper()):
            normalized.append(part.upper())
        else:
            normalized.append(part.capitalize())
    return "".join(normalized).strip()


def _looks_like_title_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    if letters < 4:
        return False
    words = stripped.split()
    if len(words) > 10:
        return False
    upper_ratio = sum(1 for c in stripped if c.isupper()) / letters
    if upper_ratio >= 0.6:
        return True
    if stripped.istitle() and len(words) <= 8:
        return True
    return False


def _looks_like_author_line(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower.startswith("by "):
        return True
    if any(char.isdigit() for char in stripped):
        return False
    words = stripped.split()
    if len(words) == 1 and len(stripped) <= 3:
        return False
    if len(words) > 8:
        return False
    letters = sum(1 for c in stripped if c.isalpha())
    if letters == 0:
        return False
    upper_ratio = sum(1 for c in stripped if c.isupper()) / letters
    return upper_ratio >= 0.6


def _extract_front_matter(lines: List[str]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, str]]:
    to_remove: set[int] = set()
    blocks: List[Dict[str, Any]] = []
    meta: Dict[str, str] = {}
    authors: List[str] = []

    seen_title = False
    collecting_authors = False
    max_scan = min(len(lines), FRONT_MATTER_MAX_SCAN)

    idx = 0
    while idx < max_scan:
        raw_line = lines[idx]
        stripped = raw_line.strip()
        lower = stripped.lower()

        if not stripped:
            if seen_title or collecting_authors:
                to_remove.add(idx)
            if collecting_authors and authors:
                break
            idx += 1
            continue

        if not seen_title and any(keyword in lower for keyword in FRONT_MATTER_SKIP_KEYWORDS):
            to_remove.add(idx)
            idx += 1
            continue

        if not seen_title and _looks_like_title_line(stripped):
            normalized_title = _normalize_capitalization(stripped)
            meta.setdefault("title", normalized_title)
            blocks.append({"type": "h1", "text": normalized_title})
            to_remove.add(idx)

            # Remove immediate duplicates of the same line
            upper_ref = stripped.upper()
            j = idx + 1
            while j < max_scan:
                nxt = lines[j].strip()
                if nxt and nxt.upper() == upper_ref:
                    to_remove.add(j)
                    j += 1
                else:
                    break
            idx = j
            seen_title = True
            continue

        if seen_title and not collecting_authors:
            if lower in {"by", "por"}:
                collecting_authors = True
                to_remove.add(idx)
                idx += 1
                continue
            if lower.startswith("by "):
                collecting_authors = True
                author_candidate = stripped[3:].strip()
                if author_candidate:
                    authors.append(_normalize_capitalization(author_candidate))
                to_remove.add(idx)
                idx += 1
                continue

        if seen_title and collecting_authors and _looks_like_author_line(stripped):
            authors.append(_normalize_capitalization(stripped))
            to_remove.add(idx)
            idx += 1
            continue

        if seen_title and _looks_like_author_line(stripped) and not authors:
            authors.append(_normalize_capitalization(stripped))
            to_remove.add(idx)
            idx += 1
            continue

        if seen_title and ("collaboration" in lower or "translated by" in lower or "illustrated by" in lower):
            blocks.append({"type": "subtitle", "text": _normalize_capitalization(stripped)})
            to_remove.add(idx)
            idx += 1
            continue

        if seen_title:
            break

        idx += 1

    if authors:
        unique_authors: List[str] = []
        seen_keys: set[str] = set()
        for name in authors:
            key = name.lower()
            if key not in seen_keys:
                seen_keys.add(key)
                unique_authors.append(name)
        meta["author"] = " & ".join(unique_authors)
        for name in unique_authors:
            blocks.append({"type": "subtitle", "text": name})

    cleaned_lines = [line for i, line in enumerate(lines) if i not in to_remove]
    return cleaned_lines, blocks, meta


def _is_poem_block(raw_lines: List[str]) -> bool:
    stripped = [line.rstrip() for line in raw_lines if line.strip()]
    if len(stripped) < 3:
        return False
    lengths = [len(line) for line in stripped]
    short_ratio = sum(1 for length in lengths if length <= 60) / len(lengths)
    if short_ratio < 0.8:
        return False
    avg_words = sum(len(line.split()) for line in stripped) / len(stripped)
    if avg_words > 12:
        return False
    sentence_like = sum(1 for line in stripped if line.endswith(('.', '!', '?')))
    if sentence_like / len(stripped) > 0.6:
        return False
    return True


def _is_letter_block(raw_lines: List[str]) -> bool:
    stripped = [line.strip() for line in raw_lines if line.strip()]
    if len(stripped) < 2:
        return False
    first = stripped[0].lower()
    if not any(first.startswith(prefix) for prefix in LETTER_SALUTATIONS):
        return False
    tail_candidates = stripped[-4:] if len(stripped) >= 4 else stripped
    for line in tail_candidates:
        clean = line.lower().strip(" ,.;:-")
        if any(clean.startswith(close) for close in LETTER_CLOSINGS):
            return True
    return False


def _join_block_lines(raw_lines: List[str]) -> str:
    return "\n".join(line.rstrip() for line in raw_lines).strip("\n")


def _detect_language_for_punctuation(text: str) -> str:
    lowercase = text.lower()
    spanish_indicators = ("¿", "¡", "ñ", "á", "é", "í", "ó", "ú")
    if any(ch in text for ch in spanish_indicators):
        return LANGUAGE_ES
    if any(word in lowercase for word in (" señor", " señora", "capítulo")):
        return LANGUAGE_ES
    return LANGUAGE_EN_US


def _build_punctuation_settings(
    text: str, overrides: Optional[NormalizerSettings]
) -> NormalizerSettings:
    settings: NormalizerSettings = NormalizerSettings()
    if overrides:
        settings.update({k: v for k, v in overrides.items() if v is not None})
    language = settings.get("language") or _detect_language_for_punctuation(text)
    settings["language"] = language
    if not settings.get("genre"):
        settings["genre"] = overrides.get("genre") if overrides else GENRE_NARRATIVE
    if not settings.get("genre"):
        settings["genre"] = GENRE_NARRATIVE

    if language == LANGUAGE_ES:
        settings.setdefault("quote_preference", "angular")
        settings.setdefault("dialogue_policy", "raya_dialogo")
        settings.setdefault("dash_policy", "raya_parentetica_espaciada")
        settings.setdefault("nbspace_policy", "nbspace_dialogue")
    else:
        settings.setdefault("quote_preference", "double_primary")
        settings.setdefault("dialogue_policy", "quotes_dialogue")
        settings.setdefault("dash_policy", "em_dash_unspaced")
        settings.setdefault("nbspace_policy", "standard_spacing")

    settings.setdefault("range_dash_policy", "en_dash_for_ranges")
    settings.setdefault("ellipsis_policy", "unicode_ellipsis")
    settings.setdefault("decimal_grouping_policy", "keep_source")
    return settings


def _serialize_punctuation_result(
    result: NormalizationResult, settings: NormalizerSettings
) -> Dict[str, Any]:
    serialized_changes = []
    for change in result.changes:
        serialized_changes.append(
            {
                "rule_id": change.rule_id,
                "description": change.description,
                "before": change.before,
                "after": change.after,
                "start_idx": change.start_idx,
                "end_idx": change.end_idx,
                "severity": change.severity,
                "notes": change.notes,
            }
        )
    return {
        "settings": dict(settings),
        "stats": dict(result.stats),
        "change_count": len(result.changes),
        "changes": serialized_changes,
    }

def process_text(
    text: str,
    *,
    title_hint: str = None,
    clean_gutenberg: bool = True,
    use_ai_supervision: bool = False,
    interactive_mode: bool = False,
    use_punctuation_module: bool = False,
    punctuation_settings: Optional[NormalizerSettings] = None,
    style_cfgs: Optional[Dict[str, Any]] = None,
    ai_model: str = "gpt-5-mini",
    token_callback: Optional[Callable[..., None]] = None,
    supervision_callback: Optional[Callable[[str, str, float, Optional[str], Optional[str]], Optional[str]]] = None,
) -> Dict[str, Any]:
    if clean_gutenberg:
        text = clean_text(text)
    text = strip_bom(normalize_text(text))
    punctuation_payload: Optional[Dict[str, Any]] = None
    if use_punctuation_module:
        effective_settings = _build_punctuation_settings(text, punctuation_settings)
        punctuation_result = normalize_punctuation(text, effective_settings)
        punctuation_payload = _serialize_punctuation_result(punctuation_result, effective_settings)
        text = punctuation_result.normalized_text
    lines = [clean_line(l) for l in text.split("\n")]
    lines = _merge_heading_wrapped_lines(lines)
    lines, front_blocks, front_meta = _extract_front_matter(lines)
    blocks = _segment_into_blocks(lines)

    # Pre-procesamiento con IndexFinder
    index_finder = IndexFinder(interactive_mode=interactive_mode)
    blocks = index_finder.process_blocks(blocks)
    if style_cfgs is None:
        style_cfgs = copy.deepcopy(DEFAULT_STYLESET)
    else:
        style_cfgs = copy.deepcopy(style_cfgs)


    global_cfg = style_cfgs.setdefault("_global", {})
    if front_meta.get("title") and not global_cfg.get("header_left_text"):
        global_cfg["header_left_text"] = front_meta["title"]
    if front_meta.get("author") and not global_cfg.get("header_right_text"):
        global_cfg["header_right_text"] = front_meta["author"]

    document: List[Dict[str, Any]] = []
    analysis: List[Dict[str, Any]] = []

    if punctuation_payload:
        analysis.append(
            {
                "line": 0,
                "decision": "punctuation_normalization",
                "info": {
                    "change_count": punctuation_payload["change_count"],
                    "stats": punctuation_payload["stats"],
                },
            }
        )

    stack: List[str] = []
    prev_block_text: Optional[str] = None
    prev_block_type: Optional[str] = None

    idx = 0
    while idx < len(blocks):
        block = blocks[idx]
        if block.get("_consumed_as_subtitle"):
            idx += 1
            continue

        block_lines = block["lines"]

        # **NUEVO PASO: Detectar Indentación Primero**
        indentation_type = detect_indentation_pattern(block)
        if indentation_type:
            # Se encontró un patrón de indentación. Clasificamos y continuamos.
            # Quitamos los espacios extra del inicio de cada línea antes de unirlas.
            clean_lines = [line.lstrip(' ') for line in block["lines"]]
            block_text = "\n".join(clean_lines)
            document.append({"type": indentation_type, "text": block_text})
            analysis.append({
                "line": block["start"],
                "decision": indentation_type,
                "info": {"len": len(block_text), "block_len": len(block_lines)},
            })
            prev_block_type = indentation_type
            prev_block_text = block_text
            idx += 1
            continue

        block_text = collapse_soft_linebreaks(block_lines).strip()

        if not block_text:
            idx += 1
            continue

        if _is_poem_block(block_lines):
            poem_text = _join_block_lines(block_lines)
            document.append({"type": "poem", "text": poem_text})
            analysis.append({
                "line": block["start"],
                "decision": "poem",
                "info": {"len": len(poem_text), "block_len": len(block_lines)},
            })
            prev_block_type = "poem"
            prev_block_text = poem_text
            idx += 1
            continue

        if _is_letter_block(block_lines):
            letter_text = _join_block_lines(block_lines)
            document.append({"type": "letter", "text": letter_text})
            analysis.append({
                "line": block["start"],
                "decision": "letter",
                "info": {"len": len(letter_text), "block_len": len(block_lines)},
            })
            prev_block_type = "letter"
            prev_block_text = letter_text
            idx += 1
            continue

        # Comprobar si el bloque fue marcado por IndexFinder
        block_type = block.get("type")
        if block_type in ("TOC_PROCESSED", "END_PROCESSED"):
            analysis.append({
                "line": block["start"],
                "decision": "block_skipped",
                "info": block.get("analysis", {}),
            })
            idx += 1
            continue

        next_block = blocks[idx + 1] if idx + 1 < len(blocks) else None
        next_first_line = next_block["lines"][0] if next_block and next_block.get("lines") else None

        context_prev_text = prev_block_text if prev_block_type == "p" else None

        processed = False

        match_full = match_patterns(
            block_text,
            raw_lines=block_lines,
            prev_line=context_prev_text,
            next_line=next_first_line,
            prev_block_type=prev_block_type,
        )

        if match_full:
            level, title, info, confidence = match_full

            # --- Human-in-the-loop ---
            if (interactive_mode and
                supervision_callback and
                HUMAN_SUPERVISION_THRESHOLD_LOW < confidence < HUMAN_SUPERVISION_THRESHOLD_HIGH):

                next_block_text = collapse_soft_linebreaks(next_block["lines"]).strip() if next_block else ""
                user_decision = supervision_callback(
                    block_text, level, confidence, prev_block_text, next_block_text
                )

                if user_decision:
                    info["human_supervision"] = {"original": level, "final": user_decision}
                    if user_decision != level:
                        # El usuario corrigió la sugerencia, ¡aprendamos de esto!
                        learner = PatternLearner()
                        learner.learn_from_correction(block_text, user_decision)
                        # Forzar la limpieza del caché para que el nuevo patrón se cargue la próxima vez.
                        from .rules import _template_cache
                        _template_cache.cache_clear()

                    if level != user_decision:
                        level = user_decision
                        if level == 'p':
                            match_full = None  # El usuario dice que es un párrafo
                        else:
                            # Reconstruir match_full con el nuevo nivel
                            _, title, info, confidence = match_full
                            match_full = (level, title, info, confidence)

            # --- AI Supervision (si no hubo intervención humana) ---
            elif use_ai_supervision and confidence < AI_SUPERVISION_THRESHOLD:
                next_block_text = collapse_soft_linebreaks(next_block["lines"]).strip() if next_block else ""
                ai_decision = supervise_heading(
                    candidate_heading=title,
                    previous_block=context_prev_text,
                    next_block=next_block_text,
                    current_decision=level,
                    confidence=confidence,
                    model=ai_model,
                    token_callback=token_callback,
                )
                info["ai_supervision"] = ai_decision
                if not ai_decision.get("is_heading"):
                    match_full = None  # AI rejects heading, treat as paragraph
                    level = "p"

        if match_full:
            level, title, info, confidence = match_full
            role = _get_marker_role(title)

            if role == "chapter_marker":
                # Nueva lógica unificada para subtítulos
                stack, eff_level = enforce(stack, level)

                # Dividir título y subtítulo si están en la misma línea
                parts = re.match(r'^((?:CHAPTER|CAPÍTULO)\s+[IVXLC\d]+\.?)\s*(?:\.\s+)?(.*)', title, re.IGNORECASE)
                main_title = title
                subtitle_text = ""
                if parts:
                    main_title = parts.group(1).strip()
                    subtitle_text = parts.group(2).strip()

                document.append({"type": eff_level, "text": main_title})
                prev_block_type = eff_level
                prev_block_text = main_title

                # Añadir subtítulo si se encontró en la misma línea
                if subtitle_text:
                    document.append({"type": "subtitle", "text": subtitle_text})
                    prev_block_type = "subtitle"
                    prev_block_text = subtitle_text

                    next_b = blocks[idx + 1] if idx + 1 < len(blocks) else None
                    if next_b:
                        cand = collapse_soft_linebreaks(next_b["lines"]).strip()
                        if cand and cand.lower() == subtitle_text.lower():
                            next_b["_consumed_as_subtitle"] = True
                # O buscar en la siguiente línea si no estaba en la misma
                # y si no se ha consumido ya uno del título.
                elif not subtitle_text:
                    next_b = blocks[idx + 1] if idx + 1 < len(blocks) else None
                    if next_b and not next_b.get("_consumed_as_subtitle"):
                        cand = collapse_soft_linebreaks(next_b["lines"]).strip()
                        if cand and detect_subtitle(cand):
                            document.append({"type": "subtitle", "text": cand})
                            prev_block_type = "subtitle"
                            prev_block_text = cand
                            next_b["_consumed_as_subtitle"] = True

                processed = True

            elif role == "container_marker":
                # Simplified logic: container markers are always standalone headings.
                # Merging logic is removed as it was causing inconsistencies.
                stack, eff_level = enforce(stack, level)
                document.append({"type": eff_level, "text": title})
                processed = True

            if not processed:
                # Fallback for when attach is not ok or no special role
                if level == "hr":
                    document.append({"type": "hr", "text": "---"})
                    analysis.append({"line": block["start"], "decision": "hr", "info": info})
                    prev_block_type = "hr"
                    prev_block_text = "---"
                else:
                    stack, eff_level = enforce(stack, level)
                    document.append({"type": eff_level, "text": title})
                    info.setdefault("block_len", len(block_lines))
                    analysis.append({"line": block["start"], "decision": eff_level, "info": info})
                    prev_block_type = eff_level
                    prev_block_text = title
                processed = True
        else:
            first_line = block_lines[0]
            first_line_text = collapse_soft_linebreaks([first_line]).strip()
            remainder_lines = block_lines[1:]
            next_for_first_line = remainder_lines[0] if remainder_lines else next_first_line

            heading_match = match_patterns(
                first_line_text,
                raw_lines=[first_line],
                prev_line=context_prev_text,
                next_line=next_for_first_line,
                prev_block_type=prev_block_type,
            )

            if heading_match:
                level, title, info, confidence = heading_match

                # --- Human-in-the-loop ---
                if (interactive_mode and
                    supervision_callback and
                    HUMAN_SUPERVISION_THRESHOLD_LOW < confidence < HUMAN_SUPERVISION_THRESHOLD_HIGH):

                    next_context = collapse_soft_linebreaks(remainder_lines).strip() if remainder_lines else (next_first_line or "")
                    user_decision = supervision_callback(
                        first_line_text, level, confidence, context_prev_text, next_context
                    )
                    if user_decision:
                        info["human_supervision"] = {"original": level, "final": user_decision}
                        if user_decision != level:
                            learner = PatternLearner()
                            learner.learn_from_correction(first_line_text, user_decision)
                            from .rules import _template_cache
                            _template_cache.cache_clear()

                        if level != user_decision:
                            level = user_decision
                            if level == 'p':
                                heading_match = None
                            else:
                                # Reconstruir heading_match con el nuevo nivel
                                _, title, info, confidence = heading_match
                                heading_match = (level, title, info, confidence)

                # --- AI Supervision (si no hubo intervención humana) ---
                elif use_ai_supervision and confidence < AI_SUPERVISION_THRESHOLD:
                    next_context = collapse_soft_linebreaks(remainder_lines).strip() if remainder_lines else (next_first_line or "")
                    ai_decision = supervise_heading(
                        candidate_heading=title,
                        previous_block=context_prev_text,
                        next_block=next_context,
                        current_decision=level,
                        confidence=confidence,
                        model=ai_model,
                        token_callback=token_callback,
                    )
                    info["ai_supervision"] = ai_decision
                    if not ai_decision.get("is_heading"):
                        heading_match = None

            if heading_match:
                level, title, info, confidence = heading_match
                heading_lines_used = 1
                if remainder_lines and _heading_remainder_attaches(remainder_lines):
                    title = collapse_soft_linebreaks(block_lines).strip()
                    heading_lines_used = len(block_lines)
                    remainder_lines = []

                stack, eff_level = enforce(stack, level)
                document.append({"type": eff_level, "text": title})
                info.setdefault("block_len", heading_lines_used)
                info.setdefault("raw_lines", heading_lines_used)
                analysis.append({"line": block["start"], "decision": eff_level, "info": info})
                prev_block_type = eff_level
                prev_block_text = title
                if remainder_lines:
                    blocks.insert(
                        idx + 1,
                        {"lines": remainder_lines, "start": block["start"] + heading_lines_used},
                    )
                processed = True
            elif len(block_lines) > 1:
                for offset in range(1, len(block_lines)):
                    candidate_line = block_lines[offset]
                    candidate_text = collapse_soft_linebreaks([candidate_line]).strip()
                    remainder_lines = block_lines[offset + 1 :]
                    candidate_next = (
                        remainder_lines[0]
                        if remainder_lines
                        else next_first_line
                    )
                    preceding_lines = block_lines[:offset]
                    preceding_text = collapse_soft_linebreaks(preceding_lines).strip() if preceding_lines else None

                    candidate_match = match_patterns(
                        candidate_text,
                        raw_lines=[candidate_line],
                        prev_line=preceding_text or context_prev_text,
                        next_line=candidate_next,
                        prev_block_type="p" if preceding_text else prev_block_type,
                    )

                    if candidate_match:
                        level, title, info, confidence = candidate_match
                        if use_ai_supervision and confidence < AI_SUPERVISION_THRESHOLD:
                            next_context = collapse_soft_linebreaks(remainder_lines).strip() if remainder_lines else (candidate_next or "")
                            ai_decision = supervise_heading(
                                candidate_heading=title,
                                previous_block=preceding_text or context_prev_text,
                                next_block=next_context,
                                current_decision=level,
                                confidence=confidence,
                                model=ai_model,
                                token_callback=token_callback,
                            )
                            info["ai_supervision"] = ai_decision
                            if not ai_decision.get("is_heading"):
                                candidate_match = None # AI says no, continue searching
                                continue

                    if candidate_match:
                        if preceding_text:
                            document.append({"type": "p", "text": preceding_text})
                            analysis.append({
                                "line": block["start"],
                                "decision": "p",
                                "info": {"len": len(preceding_text), "block_len": offset},
                            })
                            prev_block_type = "p"
                            prev_block_text = preceding_text

                        level, title, info, confidence = candidate_match
                        stack, eff_level = enforce(stack, level)
                        document.append({"type": eff_level, "text": title})
                        info.setdefault("block_len", 1)
                        info.setdefault("raw_lines", 1)
                        analysis.append({
                            "line": block["start"] + offset,
                            "decision": eff_level,
                            "info": info,
                        })
                        prev_block_type = eff_level
                        prev_block_text = title
                        if remainder_lines:
                            blocks.insert(
                                idx + 1,
                                {
                                    "lines": remainder_lines,
                                    "start": block["start"] + offset + 1,
                                },
                            )
                        processed = True
                        break

        if not processed:
            document.append({"type": "p", "text": block_text})
            analysis.append({
                "line": block["start"],
                "decision": "p",
                "info": {"len": len(block_text), "block_len": len(block_lines)},
            })
            prev_block_type = "p"
            prev_block_text = block_text

        idx += 1

    if front_blocks:
        document = front_blocks + document
        if front_meta:
            analysis.insert(0, {"line": 0, "decision": "front_matter", "info": front_meta})

    meta = {
        "title": title_hint or front_meta.get("title") or _guess_title(document),
        "generated_at": datetime.datetime.now().isoformat(timespec='seconds'),
        "engine": "txt2md_mvp 0.1.0",
        "confidence": 0.8,  # placeholder MVP
    }
    if punctuation_payload:
        meta.setdefault("punctuation_stats", punctuation_payload["stats"])
    author_guess = front_meta.get("author") or _guess_author(document)
    if author_guess:
        meta.setdefault("author", author_guess)
    if front_meta:
        meta.setdefault("front_matter", front_meta)
    toc_requested = bool(style_cfgs.get("_global", {}).get("generate_toc", False))
    md = render(document, meta, add_toc=toc_requested)
    return {
        "md": md,
        "analysis": analysis,
        "document": document,
        "meta": meta,
        "punctuation": punctuation_payload,
    }

def _guess_author(document: List[Dict[str, Any]]) -> str:
    for idx, blk in enumerate(document):
        text = blk.get("text", "").strip()
        lower = text.lower()
        if lower.startswith("by "):
            candidate = text[3:].strip()
            if candidate:
                return candidate
        if lower == "by" and idx + 1 < len(document):
            candidate = document[idx + 1].get("text", "").strip()
            if candidate:
                return candidate
    return ""

def _guess_title(document):
    for blk in document:
        if blk["type"] == "h1":
            return blk["text"]
    # fallback: first heading or empty
    for blk in document:
        if blk["type"].startswith("h"):
            return blk["text"]
    return ""

def _convert_to_docx_blocks(document: List[Dict[str, Any]]) -> List[Block]:
    """Converts the pipeline's document structure to md2docx's Block structure."""
    docx_blocks: List[Block] = []
    type_map = {
        "p": "paragraph",
        "h1": "heading1",
        "h2": "heading2",
        "h3": "heading3",
        "subtitle": "subtitle",
        "poem": "poem",
        "letter": "letter",
    }
    for item in document:
        doc_type = type_map.get(item["type"], item["type"])
        docx_blocks.append(Block(type=doc_type, content=item["text"]))
    return docx_blocks

def process_file(in_path: str, out_dir: str, clean_gutenberg: bool = True, use_ai_supervision: bool = False, interactive_mode: bool = False, session_file: Optional[str] = None):
    if session_file:
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        text = session_data['text']
        # Potentially load other session data like glossary here
    else:
        text, _encoding = _read_file_with_fallback(in_path)

    base = os.path.splitext(os.path.basename(in_path))[0]
    session_path = os.path.join(out_dir, f"{base}.session.json")

    # This is where the main processing happens
    title_hint = re.sub(r"[_\-]+", " ", base).strip()
    style_cfgs = copy.deepcopy(DEFAULT_STYLESET)
    result = process_text(
        text, title_hint=title_hint or None, clean_gutenberg=clean_gutenberg, use_ai_supervision=use_ai_supervision, interactive_mode=interactive_mode, style_cfgs=style_cfgs
    )

    # Save session data after processing
    session_data = {
        'text': text,
        'in_path': str(Path(in_path).resolve()),
        'document': result['document'],
        'punctuation': result.get('punctuation'),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(session_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    paragraph_cfg = style_cfgs.get("paragraph", DEFAULT_STYLESET["paragraph"])
    global_cfg = style_cfgs.setdefault("_global", {})

    title_text = ""
    author_text = ""
    if isinstance(meta, dict):
        title_text = (meta.get("title") or "").strip()
        author_text = (meta.get("author") or "").strip()

    if not author_text:
        author_hint = Path(in_path).parent.name.strip()
        if author_hint and (not title_text or author_hint.lower() != title_text.lower()):
            author_text = author_hint
            if isinstance(meta, dict) and not meta.get("author"):
                meta["author"] = author_text

    if not global_cfg.get("header_left_text") and title_text:
        global_cfg["header_left_text"] = title_text
    if not global_cfg.get("header_right_text") and author_text:
        global_cfg["header_right_text"] = author_text

    global_cfg.setdefault("header_font_name", paragraph_cfg.get("font_name"))
    try:
        para_size = float(paragraph_cfg.get("font_size_pt", 12))
    except (TypeError, ValueError):
        para_size = 12.0
    global_cfg.setdefault("header_font_size_pt", max(para_size - 1, 6))
    if global_cfg.get("header_font_size_pt") is None:
        global_cfg["header_font_size_pt"] = max(para_size - 1, 6)
    global_cfg.setdefault("header_italic", True)
    global_cfg.setdefault("header_distance_in", DEFAULT_STYLESET["_global"]["header_distance_in"])
    global_cfg.setdefault("footer_distance_in", DEFAULT_STYLESET["_global"]["footer_distance_in"])
    global_cfg.setdefault("header_space_after_pt", DEFAULT_STYLESET["_global"]["header_space_after_pt"])
    global_cfg.setdefault("footer_space_before_pt", DEFAULT_STYLESET["_global"]["footer_space_before_pt"])
    global_cfg.setdefault("header_alignment", DEFAULT_STYLESET["_global"]["header_alignment"])
    global_cfg.setdefault("chapter_padding_top_lines", DEFAULT_STYLESET["_global"]["chapter_padding_top_lines"])
    global_cfg.setdefault("chapter_padding_bottom_lines", DEFAULT_STYLESET["_global"]["chapter_padding_bottom_lines"])
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, base + ".md")
    an_path = os.path.join(out_dir, base + ".analysis.json")
    rep_path = os.path.join(out_dir, base + ".report.md")
    docx_path = os.path.join(out_dir, base + ".docx")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(result["md"])
    with open(an_path, "w", encoding="utf-8") as f:
        json.dump(result["analysis"], f, ensure_ascii=False, indent=2)

    # Generate DOCX from the pipeline's document structure
    docx_blocks = _convert_to_docx_blocks(result["document"])
    doc = build_document(
        blocks=docx_blocks,
        style_cfgs=style_cfgs,
        page_cfg=DEFAULT_PAGE.copy(),
    )
    configure_headers(doc, style_cfgs)
    add_page_numbers(doc, style_cfgs)
    doc.save(docx_path)
    _auto_refresh_docx_fields(docx_path)

    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("# Reporte de maquetado (MVP)\\n\\n")
        f.write(f"- Entradas: {os.path.basename(in_path)}\\n")
        headings_total = sum(1 for d in result['document'] if d['type'].startswith('h'))
        paragraphs_total = sum(1 for d in result['document'] if d['type'] == 'p')
        counts_by_level = {
            level: sum(1 for d in result['document'] if d['type'] == level)
            for level in ('h1', 'h2', 'h3', 'blockquote')
        }
        f.write(
            "- Bloques detectados: "
            f"{headings_total} encabezados (h1: {counts_by_level.get('h1', 0)}, h2: {counts_by_level.get('h2', 0)}, h3: {counts_by_level.get('h3', 0)}), "
            f"{paragraphs_total} párrafos, {counts_by_level.get('blockquote', 0)} citas\\n"
        )

        template_usage = {}
        heuristic_usage = {}
        for entry in result['analysis']:
            info = entry.get('info', {})
            template = info.get('template')
            if template:
                template_usage[template] = template_usage.get(template, 0) + 1
            heuristic = info.get('heuristic')
            if heuristic:
                heuristic_usage[heuristic] = heuristic_usage.get(heuristic, 0) + 1

        tpl_summary = (
            ', '.join(f"{k}: {v}" for k, v in sorted(template_usage.items(), key=lambda kv: kv[1], reverse=True))
            if template_usage else 'N/A'
        )
        heur_summary = (
            ', '.join(f"{k}: {v}" for k, v in sorted(heuristic_usage.items(), key=lambda kv: kv[1], reverse=True))
            if heuristic_usage else 'Ninguna'
        )
        f.write(f"- Plantillas aplicadas: {tpl_summary}\\n")
        f.write(f"- Heurísticas aplicadas: {heur_summary}\\n")

        alerts = []
        if counts_by_level.get('h1', 0) == 0:
            alerts.append("No se detectaron encabezados h1; revisa las reglas o la calidad del texto.")
        if heuristic_usage:
            alerts.append("Se usaron heurísticas; validar manually que los encabezados sean correctos.")
        if counts_by_level.get('h1', 0) > 0 and counts_by_level.get('h2', 0) == 0:
            alerts.append("Solo se detectaron encabezados h1; considera revisar niveles secundarios.")

        if alerts:
            f.write("\\n## Alertas\\n")
            for alert in alerts:
                f.write(f"- {alert}\\n")

        if result.get("punctuation"):
            f.write("\\n## Normalización de Puntuación\\n")
            stats = result["punctuation"]["stats"]
            f.write(f"- Cambios propuestos: {stats.get('proposed_changes', 0)}\\n")
            f.write(f"- Cambios aplicados: {stats.get('applied_changes', 0)}\\n")
            f.write("### Desglose de cambios aplicados:\\n")
            for key, value in stats.items():
                if key not in ["proposed_changes", "applied_changes"]:
                    f.write(f"- {key}: {value} cambios\\n")

        f.write("\\n- Notas: Motor MVP basado en reglas y heurísticas.\\n")
    return md_path, an_path, rep_path, docx_path


import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .utils import (
        get_indentation_level, is_all_caps_title, should_skip_titlecase_candidate,
        titlecase_hint
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        get_indentation_level, is_all_caps_title, should_skip_titlecase_candidate,
        titlecase_hint
    )

MIN_INDENTATION_SPACES = 2  # Mínimo de espacios para considerarlo una indentación especial


def detect_indentation_pattern(block: Dict) -> Optional[str]:
    """
    Analiza un bloque de texto y devuelve su tipo estructural basado en la indentación.
    Devuelve 'blockquote', 'poem', etc., o None si es un párrafo normal.
    """
    raw_lines = block.get("lines", [])
    if not raw_lines or len(raw_lines) == 0:
        return None

    # 1. Calcular los niveles de indentación de todas las líneas del bloque
    indent_levels = [get_indentation_level(line) for line in raw_lines if line.strip()]
    if not indent_levels:  # Si el bloque solo tiene líneas vacías
        return None

    # 2. Heurística para BLOCKQUOTE (Cita de Bloque)
    # Condición: Casi todas las líneas tienen una indentación consistente y significativa.
    first_indent = indent_levels[0]
    if first_indent >= MIN_INDENTATION_SPACES:
        # Contamos cuántas líneas cumplen con esa indentación inicial
        lines_with_same_indent = sum(1 for indent in indent_levels if indent >= first_indent)

        # Si más del 80% de las líneas están indentadas de forma similar, es un blockquote
        if (lines_with_same_indent / len(indent_levels)) > 0.8:
            return "blockquote"

    # 3. Heurística para POEM (Poesía) - (Más avanzada, para futuro)
    # Condición: Múltiples líneas cortas con indentación variable.
    # ... (Esta lógica puede ser más compleja, analizando longitud de líneas, etc.) ...
    # Por ahora, nos enfocamos en blockquote.

    # Si no se cumple ninguna regla, es un párrafo normal
    return None


DEFAULT_PATTERNS = {
    "h1": [
        r"^\s*(?:parte|libro)\s+(?P<num>([MDCLXVI]+|\d+))\b[:.\-–—]?\s*(?P<title>.*)$",
        r"^\s*(?:t[ií]tulo)\s+(?P<num>\d+|[MDCLXVI]+)\b[:.\-–—]?\s*(?P<title>.*)$",
    ],
    "h2": [
        r"^\s*(?:cap[ií]tulo)\s+(?P<num>(\d+|[MDCLXVI]+))\b[:.\-–—]?\s*(?P<title>.*)$",
        r"^\s*(?:secci[oó]n)\s+(?P<num>(\d+(\.\d+)*))\b[:.\-–—]?\s*(?P<title>.*)$",
        r"^\s*(?P<num>\d+)\.\s+(?P<title>.+)$",
        r"^\s*(?P<num>\d+(\.\d+)+)\s+(?P<title>.+)$",
    ],
    "h3": [
        r"^\s*(?P<num>\d+(\.\d+){1,})\s+(?P<title>.+)$",
    ],
}


def _parse_scalar(token: str):
    if token.lower() in {"true", "false"}:
        return token.lower() == "true"
    if token.isdigit():
        try:
            return int(token)
        except ValueError:
            pass
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    return token


def _parse_template(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_key = None
    current_subkey = None
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if indent == 0:
                current_key = None
                current_subkey = None
                if line.endswith(":"):
                    key = line[:-1].strip()
                    data[key] = {}
                    current_key = key
                else:
                    key, value = line.split(":", 1)
                    data[key.strip()] = _parse_scalar(value.strip())
            elif indent == 2 and current_key:
                container = data[current_key]
                if line.endswith(":"):
                    key = line[:-1].strip()
                    if isinstance(container, dict):
                        container[key] = []
                        current_subkey = key
                elif line.startswith("- "):
                    if isinstance(container, list):
                        container.append(_parse_scalar(line[2:].strip()))
                else:
                    key, value = line.split(":", 1)
                    if isinstance(container, dict):
                        container[key.strip()] = _parse_scalar(value.strip())
                        current_subkey = None
            elif indent == 4 and current_key and current_subkey:
                container = data[current_key][current_subkey]
                if isinstance(container, list) and line.startswith("- "):
                    container.append(_parse_scalar(line[2:].strip()))
    return data


def _compile_patterns(pattern_dict: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    compiled: Dict[str, List[re.Pattern]] = {}
    for lvl, pats in pattern_dict.items():
        compiled[lvl] = [re.compile(pat, flags=re.I) for pat in pats]
    return compiled


def _default_template() -> Dict[str, Any]:
    tpl = {
        "id": "builtin_default",
        "priority": -1,
        "patterns": DEFAULT_PATTERNS,
        "signals": {
            "allow_all_caps_titles": True,
            "titlecase_hint": True,
        },
    }
    tpl["_compiled"] = _compile_patterns(DEFAULT_PATTERNS)
    return tpl


def _load_templates() -> List[Dict[str, Any]]:
    """Carga plantillas desde el directorio principal y el de patrones aprendidos."""
    templates: List[Dict[str, Any]] = []

    # Lista de directorios de donde cargar plantillas
    template_dirs = [
        Path(__file__).resolve().parent / "templates",
        Path(__file__).resolve().parent / "templates" / "learned"
    ]

    for template_dir in template_dirs:
        if template_dir.exists():
            for path in sorted(template_dir.glob("*.yml")):
                try:
                    data = _parse_template(path)
                    data.setdefault("id", path.stem)

                    # Los patrones aprendidos tienen una prioridad más alta
                    if "learned" in str(path):
                        data.setdefault("priority", 10) # Mayor prioridad para patrones aprendidos
                    else:
                        data.setdefault("priority", 0)

                    patterns = data.get("patterns", {})

                    # Si hay un solo regex_pattern, lo metemos en la estructura de patterns
                    if "regex_pattern" in data and "type" in data:
                        level = data["type"]
                        if level not in patterns:
                            patterns[level] = []
                        patterns[level].append(data["regex_pattern"])

                    compiled = _compile_patterns(patterns)
                    data["_compiled"] = compiled

                    if "signals" not in data:
                        data["signals"] = {}
                    templates.append(data)
                except Exception:
                    continue

    templates.append(_default_template())
    templates.sort(key=lambda item: item.get("priority", 0), reverse=True)
    return templates


@lru_cache(maxsize=1)
def _template_cache() -> List[Dict[str, Any]]:
    return _load_templates()


def _looks_like_numbered_list(title: str) -> bool:
    text = title.strip()
    if not text:
        return True
    words = text.split()
    if len(words) >= 12:
        return True
    if any(sep in text for sep in ('. ', '? ', '! ')):
        return True
    return False


ALL_CAPS_HEADING_KEYWORDS = {
    "CHAPTER",
    "CAPITULO",
    "CAPÍTULO",
    "PART",
    "BOOK",
    "LIBRO",
    "SECTION",
    "SECCION",
    "SECCIÓN",
    "VOLUME",
    "ACT",
}

ROMAN_NUMERAL_RE = re.compile(
    r"^(?=[MDCLXVI])M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
ROMAN_PREFIX_RE = re.compile(r"^[IVXLCDM]{1,4}(?:[\.\-: ]|$)")


def _has_positive_all_caps_signal(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    upper = stripped.upper()

    if ROMAN_PREFIX_RE.match(upper):
        return True

    tokens = [tok for tok in re.split(r"[^\wÁÉÍÓÚÜÑ]+", upper) if tok]
    if not tokens:
        return False

    first = tokens[0]
    if first in ALL_CAPS_HEADING_KEYWORDS:
        if len(tokens) == 1:
            return True
        second = tokens[1]
        if second.isdigit() or ROMAN_NUMERAL_RE.match(second):
            return True
        return True

    return False



SCENE_SEPARATOR_RE = re.compile(
    r"^(?:(?:\s*[*]{3,}\s*)|"
    r"(?:\s*[*](?:\s+[*]){2,}\s*)|"
    r"(?:\s*-{3,}\s*)|(?:\s*-\s+-\s+-\s*)|"
    r"(?:\s*_{3,}\s*)|(?:\s*_\s+_\s+_\s*)|"
    r"(?:\s*~{3,}\s*)|(?:\s*~\s+~\s+~\s*)|"
    r"(?:\s*o\s*O\s*o\s*)|"
    r"(?:\s*\u2022{3,}\s*)|"
    r"(?:\s*\u00B7{3,}\s*)|"
    r"(?:\s*#[#\s]{2,}#?\s*)|"
    r"(?:\s*=\s*=\s*=\s*)|"
    r"(?:\s*\.+\s*)|"
    r"(?:\s*\u2219\s*\u2219\s*\u2219\s*)|"
    r"(?:\s*\u25CF\s*\u25CF\s*\u25CF\s*)|"
    r"(?:\s*\u25CB\s*\u25CB\s*\u25CB\s*))$",
    re.UNICODE,
)


def match_patterns(
    text: str,
    *,
    raw_lines: Optional[List[str]] = None,
    in_paragraph: bool = False,
    prev_line: Optional[str] = None,
    next_line: Optional[str] = None,
    prev_block_type: Optional[str] = None,
):
    stripped = text.strip()
    if not stripped:
        return None

    if SCENE_SEPARATOR_RE.match(stripped):
        confidence = 1.0
        info = {"heuristic": "scene_separator", "confidence": confidence}
        return "hr", "---", info, confidence

    templates = _template_cache()
    for tpl in templates:
        compiled = tpl.get("_compiled", {})
        for level in ("h1", "h2", "h3"):
            for regex in compiled.get(level, []):
                m = regex.match(stripped)
                if m:
                    title = m.groupdict().get("title") or stripped
                    groups = m.groupdict()
                    num_token = groups.get("num")
                    if num_token and num_token.isdigit() and _looks_like_numbered_list(title):
                        continue

                    # Usar el confidence_boost del template si existe, si no, 0.95 por defecto.
                    base_confidence = tpl.get("confidence_boost", 95) / 100.0
                    confidence = min(base_confidence, 1.0) # Asegurar que no pase de 1.0

                    info = {
                        "pattern": regex.pattern,
                        "groups": m.groupdict(),
                        "template": tpl.get("id"),
                        "confidence": confidence,
                    }
                    return level, title.strip(), info, confidence

    for tpl in templates:
        signals = tpl.get("signals", {})
        tpl_id = tpl.get("id")
        if (
            not in_paragraph
            and signals.get("allow_all_caps_titles")
            and is_all_caps_title(stripped)
        ):
            if not _has_positive_all_caps_signal(stripped):
                continue
            confidence = 0.6
            info: Dict[str, Any] = {
                "heuristic": "all_caps_short",
                "template": tpl_id,
                "signals": signals,
                "confidence": confidence,
            }
            if raw_lines is not None:
                info["raw_lines"] = len(raw_lines)
            return "h2", stripped, info, confidence

        allow_titlecase = signals.get("titlecase_hint")
        if allow_titlecase or (allow_titlecase is None and tpl_id == "builtin_default"):
            if (
                len(stripped) <= 72
                and not stripped.endswith(('.', ';', ','))
                and not in_paragraph
                and not should_skip_titlecase_candidate(
                    stripped,
                    prev_line=prev_line,
                    next_line=next_line,
                    raw_lines=raw_lines,
                    prev_block_type=prev_block_type,
                )
                and titlecase_hint(stripped)
            ):
                confidence = 0.4
                info = {
                    "heuristic": "titlecase_short",
                    "template": tpl_id,
                    "signals": signals,
                    "confidence": confidence,
                }
                if raw_lines is not None:
                    info["raw_lines"] = len(raw_lines)
                return "h3", stripped, info, confidence
    return None

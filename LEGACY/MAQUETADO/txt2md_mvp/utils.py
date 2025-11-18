import re
from typing import List, Optional


def get_indentation_level(line: str) -> int:
    """Calcula el número de espacios al inicio de una línea."""
    return len(line) - len(line.lstrip(' '))


DIALOGUE_OPENERS = ('"', "'", '“', '‘', '«', '„', '—')
DIALOGUE_CLOSERS = ('"', "'", '”', '’', '»', '“')
SPEECH_VERBS = (
    'dijo', 'respondió', 'respondio', 'exclamó', 'exclamo', 'preguntó', 'pregunto',
    'susurró', 'susurro', 'murmuró', 'murmuro', 'gritó', 'grito', 'contestó', 'contesto',
    'añadió', 'anadio', 'añadio', 'replicó', 'replico', 'observó', 'observo', 'indicó', 'indico',
    'said', 'replied', 'asked', 'whispered', 'shouted', 'murmured', 'cried', 'answered', 'called'
)
PERSONAL_PRONOUNS = {
    'yo', 'tú', 'vos', 'usted', 'ustedes', 'él', 'ella', 'ello', 'nosotros', 'nosotras',
    'vosotros', 'vosotras', 'ellos', 'ellas', 'i', 'you', 'he', 'she', 'we', 'they'
}

PUNCT_FOLLOWERS = ",.;:?!)]}»”'\""

WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’\-]*")

ROMAN_RE = re.compile(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.I)
ROMAN_LIST_RE = re.compile(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.\s', re.I)


def is_uppercase_segment(value: str) -> bool:
    letters = [c for c in value if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def is_roman(s: str) -> bool:
    s = s.strip().upper()
    return bool(ROMAN_RE.match(s))

def is_all_caps_title(line: str) -> bool:
    txt = line.strip()
    if not txt or len(txt) > 90:
        return False
    # all caps with spaces/punct, no trailing full stop
    alpha = [c for c in txt if c.isalpha()]
    if not alpha:
        return False

    # Exclude if it ends with sentence-ending punctuation, even inside a quote.
    temp_txt = txt
    if txt.endswith(DIALOGUE_CLOSERS):
        # Temporarily remove final quote to check what's before it
        temp_txt = txt[:-1].rstrip()

    if temp_txt.endswith(('.', ';', ':', '!', '?')):
        return False

    return all(c.isupper() for c in alpha)

def titlecase_hint(line: str) -> bool:
    w = [t for t in line.strip().split() if t.isalpha()]
    if not w:
        return False
    # at least half capitalized first letter
    caps = sum(1 for t in w if t[:1].isupper())
    return caps >= max(1, len(w)//2)


def should_skip_titlecase_candidate(
    line: str,
    prev_line: Optional[str] = None,
    next_line: Optional[str] = None,
    raw_lines: Optional[List[str]] = None,
    prev_block_type: Optional[str] = None,
) -> bool:
    stripped = line.strip()
    if not stripped:
        return True

    # Exclude short lines ending in a period or quote, likely signatures or dialogue.
    words = WORD_RE.findall(stripped)
    if len(words) <= 4 and (stripped.endswith(('.', '!', '?')) or stripped.endswith(DIALOGUE_CLOSERS)):
        return True

    lowered = stripped.lower()

    if raw_lines and len(raw_lines) > 1:
        uppercase_lines = 0
        total_lines = 0
        for raw in raw_lines:
            raw_stripped = raw.strip()
            if not raw_stripped:
                continue
            total_lines += 1
            if is_uppercase_segment(raw_stripped):
                uppercase_lines += 1
        if total_lines and uppercase_lines < total_lines:
            return True

    # Lines enclosed in quotes or starting with typical dialogue cues
    if stripped[0] in DIALOGUE_OPENERS:
        return True
    if stripped.endswith(DIALOGUE_CLOSERS):
        return True
    if any(q in stripped for q in ('"', '“', '”', '«', '»')):
        # Multiple quotes inside the line are a strong dialogue signal
        quote_count = sum(stripped.count(q) for q in ('"', '“', '”', '«', '»'))
        if quote_count >= 2:
            return True

    if any(ch in stripped for ch in ('.', '?', '!')):
        if not re.match(r"^(([IVXLCDM]+|[A-Z])\.|\d+\.)\s", stripped):
            return True

    words = WORD_RE.findall(stripped)
    if len(words) >= 3:
        caps = sum(1 for w in words if w[0].isupper())
        if caps < max(2, int(len(words) * 0.6)):
            return True

    # Speech verbs usually indicate narrative dialogue
    for verb in SPEECH_VERBS:
        if re.search(rf"\b{verb}\b", lowered):
            return True

    # Short pronoun-led sentences are often dialogue fragments
    tokens = [re.sub(r"[^\wáéíóúüñ'-]", '', t.lower()) for t in stripped.split()]
    tokens = [t for t in tokens if t]
    if tokens and tokens[0] in PERSONAL_PRONOUNS and len(stripped) <= 60:
        return True

    # Commas or semicolons at the end signal continuation of a sentence
    if stripped.endswith((',', ';')):
        return True

    # Lines ending with colon introducing dialogue or exposition
    if stripped.endswith(':'):
        if next_line and next_line.lstrip().startswith(DIALOGUE_OPENERS):
            return True
        if next_line and next_line.strip().lower().startswith(tuple(PERSONAL_PRONOUNS)):
            return True

    # If previous non-empty line did not end a paragraph, avoid promoting continuation
    if prev_block_type == "p" and prev_line:
        prev = prev_line.rstrip()
        if prev and not prev.endswith(('.', '!', '?', '"', '”', '’', '»')):
            return True

    return False

def clean_line(line: str) -> str:
    return line.rstrip("\n").replace("\t", "    ")


def _should_preserve_line_structure(line: str) -> bool:
    stripped = line.lstrip()
    if not stripped:
        return False
    if stripped.startswith(('> ', '- ', '* ', '+ ', '```', '~~~')):
        return True
    if re.match(r"\d+\.\s", stripped):
        return True
    if ROMAN_LIST_RE.match(stripped):
        return True
    if '|' in stripped:
        return True
    return False


def collapse_soft_linebreaks(lines: List[str]) -> str:
    """
    Joins lines within a block of text that are part of the same paragraph.
    A "block" is assumed to be a single paragraph.
    """
    filtered = [ln.rstrip() for ln in lines if ln.strip()]
    if not filtered:
        return ""
    if any(_should_preserve_line_structure(ln) for ln in filtered):
        return "\n".join(filtered).strip()

    # Join lines, handling hyphenated words at the end of a line.
    rebuilt_text = " ".join(filtered)
    # Remove space after a hyphen at the end of a virtual line.
    rebuilt_text = re.sub(r'(-\s|–\s|—\s)', '', rebuilt_text)

    return rebuilt_text


def normalize_text(text: str) -> str:
    # basic normalization
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse 3+ blank lines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def strip_bom(s: str) -> str:
    return s.lstrip('\ufeff')


def is_title_like_for_merge(text: str) -> bool:
    """
    Determines if a text is suitable for merging as a title.
    This is a separate check from detect_subtitle.
    """
    s = text.strip()
    if not s:
        return False
    if len(s.split()) > 8:
        return False
    if s.endswith(('.', '!', '?')):
        return False
    return True


STOPWORDS_EN = {
    'a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from',
    'by', 'of', 'in', 'out', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'
}
STOPWORDS_ES = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por',
    'un', 'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero',
    'sus', 'le', 'ya', 'o', 'este', 'ha', 'sí', 'porque', 'esta', 'cuando',
    'muy', 'sin', 'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien',
    'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra',
    'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes',
    'algunos', 'qué', 'entre', 'ser', 'es', 'son', 'fue', 'era', 'eres',
    'somos', 'soy', 'está', 'están', 'estoy', 'estamos', 'estuvo', 'estuvimos',
    'estuvieron', 'había', 'habíamos', 'habían', 'he', 'has', 'han', 'hemos'
}


def stopword_ratio(tokens: List[str], lang_hint: Optional[str] = None) -> float:
    """Calculates the ratio of stopwords in a list of tokens."""
    if not tokens:
        return 0.0
    sw = STOPWORDS_ES if lang_hint == "es" else STOPWORDS_EN
    hits = sum(1 for w in tokens if w.lower() in sw)
    return hits / len(tokens)


def detect_subtitle(text: str, lang_hint: Optional[str] = None) -> bool:
    """
    Detects if a given text is likely a subtitle based on heuristics.
    """
    s = text.strip()
    if not s:
        return False

    # 1. No sentence-ending punctuation
    if any(p in s for p in ".!?:;"):
        return False

    words = s.split()

    # 2. Word count limit
    if len(words) > 8:
        return False

    # 3. Low stopword ratio
    if stopword_ratio(words, lang_hint) > 0.35:
        return False

    # 4. High ratio of uppercase or title-cased words
    # (A simple proxy for this is to check capitalization)
    upperish = sum(1 for w in words if w.isupper() and len(w) > 1)
    titleish = sum(1 for w in words if w[0].isupper())

    # It's a subtitle if it's mostly uppercase or title-cased
    is_case_dominant = upperish >= max(1, len(words) // 2) or titleish >= len(words) - 2

    return is_case_dominant


def is_title_like_for_merge(text: str) -> bool:
    """
    Determines if a text is suitable for merging as a title.
    This is a separate check from detect_subtitle.
    """
    s = text.strip()
    if not s:
        return False
    if len(s.split()) > 8:
        return False
    if s.endswith(('.', '!', '?')):
        return False
    return True
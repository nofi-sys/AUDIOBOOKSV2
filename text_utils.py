"""Utilities for text normalization, paragraph rebuilding, and anchor helpers."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re
import unicodedata
from collections import Counter

import unidecode
import pdfplumber
from rapidfuzz import fuzz

# stopwords set and digit names reused across modules
STOP = {
    "de",
    "la", "el", "y", "que", "en", "a", "los", "se", "del", "por", "con", "las", "un", "para", "una",
    "su", "al", "lo", "como", "mas", "o", "pero", "sus", "le", "ya", "fue", "punto", "puntos", "coma",
    "es", "era", "eres", "soy", "somos", "son", "ser", "haber", "hay", "habia",
    "esto", "esta", "este", "estas", "estos",
    "que", "si", "ni", "no",
}

# stopwords prohibidas para anclas (articulos/pronombres vacios de contenido)
ANCHOR_STOPWORDS = STOP.union({
    "las", "los", "lo", "al", "del",
    "su", "sus", "mi", "mis", "tu", "tus", "nuestro", "nuestra", "nuestros", "nuestras",
    "uno", "una", "unos", "unas", "algunos", "algunas",
    "le", "les", "se", "me", "te", "nos",
    "que", "como", "cuando", "donde",
})

# weight for mismatching stop words in DTW
STOP_WEIGHT = 0.2

DIGIT_NAMES = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco",
    "6": "seis", "7": "siete", "8": "ocho", "9": "nueve", "10": "diez", "11": "once",
    "12": "doce", "13": "trece", "14": "catorce", "15": "quince", "16": "dieciseis",
    "17": "diecisiete", "18": "dieciocho", "19": "diecinueve", "20": "veinte",
}

# frequency threshold for trigram anchors (fraction of total, keep very rare).
# Tip: si el recorte propone zonas ridiculas, bajar aun mas este valor.
ANCHOR_MAX_FREQ = 0.015

TITLE_MAX_WORDS = 8
SENTENCE_END_RE = re.compile(r"[.!?][\"')»”]*\s*$")
ROMAN_RE = re.compile(r'^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', re.I)


def _looks_like_heading(line: str, next_line: Optional[str] = None) -> bool:
    """Heuristic to detect headings/titles without sentence-ending punctuation."""
    stripped = line.strip()
    if not stripped:
        return False

    # Keep explicit sentence endings as paragraph continuation, not titles
    if SENTENCE_END_RE.search(stripped):
        return False

    words = stripped.split()
    if len(words) > TITLE_MAX_WORDS:
        return False

    # Chapter markers such as I., II, CAPITULO, etc.
    if ROMAN_RE.match(words[0].rstrip(".").upper()):
        return True
    if stripped.isupper():
        return True

    caps = sum(1 for w in words if w[:1].isupper())
    if caps >= max(1, len(words) // 2):
        return True

    if next_line and next_line[:1].isupper():
        return True

    return False


def _collapse_soft_breaks(lines: List[str]) -> str:
    """Joins line-wrapped text while keeping explicit breaks already handled by the caller."""
    filtered = [ln.strip() for ln in lines if ln.strip()]
    if not filtered:
        return ""

    joined: List[str] = []
    for idx, ln in enumerate(filtered):
        if idx > 0 and joined and joined[-1].endswith("-"):
            joined[-1] = joined[-1][:-1] + ln  # de-hyphenate word breaks
        else:
            joined.append(ln)
    return " ".join(joined)


def prepare_paragraphs(text: str) -> List[str]:
    """
    Rebuild paragraphs using heuristics tailored for scanned TXT books:
    - Split on blank lines.
    - If a non-empty line ends with sentence-ending punctuation, force a paragraph break.
    - Short lines (<= TITLE_MAX_WORDS) without punctuation are treated as headings/titles.
    - Remaining single newlines are treated as soft wraps and collapsed.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")
    paragraphs: List[str] = []
    buffer: List[str] = []

    for idx, raw in enumerate(raw_lines):
        line = raw.strip()
        next_line = raw_lines[idx + 1].strip() if idx + 1 < len(raw_lines) else ""

        if not line:
            if buffer:
                para = _collapse_soft_breaks(buffer)
                if para:
                    paragraphs.append(para)
                buffer = []
            continue

        buffer.append(line)

        if SENTENCE_END_RE.search(line):
            para = _collapse_soft_breaks(buffer)
            if para:
                paragraphs.append(para)
            buffer = []
            continue

        if _looks_like_heading(line, next_line):
            para = _collapse_soft_breaks(buffer)
            if para:
                paragraphs.append(para)
            buffer = []
            continue

    if buffer:
        para = _collapse_soft_breaks(buffer)
        if para:
            paragraphs.append(para)

    return paragraphs


def paragraphs_to_markdown(paragraphs: List[str]) -> str:
    """Builds a markdown rendition of the reconstructed paragraphs."""
    md_lines: List[str] = []
    for para in paragraphs:
        if not para:
            continue
        if len(para.split()) <= TITLE_MAX_WORDS and not SENTENCE_END_RE.search(para):
            md_lines.append(f"## {para}")
        else:
            md_lines.append(para)
        md_lines.append("")  # blank line between paragraphs
    return "\n".join(md_lines).rstrip() + "\n"


def normalize(text: str, strip_punct: bool = True) -> str:
    """Lowercase, remove accents and optionally strip punctuation."""
    text = text.replace("“", ' " ').replace("”", ' " ')
    text = text.replace("«", ' " ').replace("»", ' " ')
    text = re.sub(r'([.,;?!])', r' \1 ', text)
    text = unidecode.unidecode(text.lower())
    text = re.sub(r"\b([a-z])\.\b", r"\1", text)
    abbr = {
        "dr": "doctor", "dra": "doctora", "sr": "senor", "sra": "senora",
        "srta": "senorita", "esq": "escribano", "ing": "ingeniero",
        "arq": "arquitecto", "lic": "licenciado", "licda": "licenciada",
        "ud": "usted", "uds": "ustedes",
    }
    for short, full in abbr.items():
        text = re.sub(rf"\b{short}\.", full, text)
    if not strip_punct:
        text = re.sub(r"\bpunto y coma\b", ";", text)
        text = re.sub(r"\bpunto\b", ".", text)
        text = re.sub(r"\bpuntos\b", ".", text)
        text = re.sub(r"\bcoma\b", ",", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_script(path: str) -> str:
    """Return raw text from a PDF or TXT file without normalization."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        with pdfplumber.open(p) as pdf:
            pages = [pg.extract_text() or "" for pg in pdf.pages]
        raw = "\n".join(pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa un TXT.")
        return raw
    for enc in ("utf-8", "latin-1"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            pass
    try:
        import chardet
        data = p.read_bytes()
        enc = chardet.detect(data)["encoding"] or "latin-1"
        return data.decode(enc, errors="replace")
    except Exception as exc:
        raise RuntimeError(f"No se pudo determinar la codificación: {exc}")


def normalize_paragraph_breaks(text: str) -> str:
    """
    Normalizes paragraph breaks in the input text.
    Rebuilds paragraphs using heuristics and returns a double-newline separated text.
    """
    return "\n\n".join(prepare_paragraphs(text))


def split_text_into_paragraphs(text: str) -> List[str]:
    """
    Splits the input text into a list of paragraphs.
    Uses paragraph heuristics to collapse soft line breaks and detect headings.
    """
    return prepare_paragraphs(text)


def token_equal(a: str, b: str) -> bool:
    """Return True if tokens are very similar or digit equivalents."""
    from rapidfuzz.distance import Levenshtein

    digit_name_to_num = {unidecode.unidecode(v): k for k, v in DIGIT_NAMES.items()}

    def _base(t: str) -> str:
        t = unicodedata.normalize("NFKD", t)
        t = t.encode("ascii", "ignore").decode("ascii")
        return t.casefold()

    a_base = _base(a)
    b_base = _base(b)
    if a_base == b_base:
        return True

    if (len(a) == 2 and a[1] == "." and a[0].isalpha() and len(b) == 1 and b == a[0]) or \
       (len(b) == 2 and b[1] == "." and b[0].isalpha() and len(a) == 1 and a == b[0]):
        return True

    punct_map = {".": {"punto", "puntos"}, ",": {"coma"}, ";": {"punto y coma"}}
    for sym, words in punct_map.items():
        if (a == sym and b in words) or (b == sym and a in words):
            return True

    def _num_value(tok: str) -> str | None:
        if tok.isdigit():
            return tok.lstrip("0") or "0"
        if tok in digit_name_to_num:
            return digit_name_to_num[tok]
        return None

    a_num = _num_value(a_base)
    b_num = _num_value(b_base)
    if a_num or b_num:
        if a_num and b_num:
            return a_num == b_num
        # No mezclar cifras con palabras genericas
        return False

    if Levenshtein.normalized_distance(a_base, b_base) <= 0.2:
        return True
    return False


def find_anchor_trigrams(ref_tok: List[str], hyp_tok: List[str]) -> List[Tuple[int, int]]:
    """Locate monotonic trigram anchors present at low frequency."""
    def _collect_trigrams(tokens: List[str], relaxed: bool = False, allow_stopwords: bool = False) -> List[Tuple[str, str, str]]:
        trigs: List[Tuple[str, str, str]] = []
        for i in range(len(tokens) - 2):
            tri = (tokens[i], tokens[i + 1], tokens[i + 2])
            if not allow_stopwords and any(t in ANCHOR_STOPWORDS for t in tri):
                continue
            if not relaxed:
                content = sum(1 for t in tri if t.isdigit() or len(t) >= 4)
                if content < 2:
                    continue
            trigs.append(tri)
        return trigs

    ref_trigs = _collect_trigrams(ref_tok)
    hyp_trigs = _collect_trigrams(hyp_tok)

    def _freq_limit(total: int) -> int:
        if total <= 0:
            return 0
        if ANCHOR_MAX_FREQ < 1:
            return max(1, int(total * ANCHOR_MAX_FREQ))
        return int(ANCHOR_MAX_FREQ)

    freq_ref = Counter(ref_trigs)
    freq_hyp = Counter(hyp_trigs)
    lowfreq_ref = {tri for tri, count in freq_ref.items() if count <= _freq_limit(len(ref_trigs))}
    lowfreq_hyp = {tri for tri, count in freq_hyp.items() if count <= _freq_limit(len(hyp_trigs))}
    candidate_trigs = lowfreq_ref.intersection(lowfreq_hyp)
    # Si no hay candidatos, relajar filtros para textos muy cortos
    if not candidate_trigs:
        ref_trigs = _collect_trigrams(ref_tok, relaxed=True)
        hyp_trigs = _collect_trigrams(hyp_tok, relaxed=True)
        freq_ref = Counter(ref_trigs)
        freq_hyp = Counter(hyp_trigs)
        lowfreq_ref = {tri for tri, count in freq_ref.items() if count <= _freq_limit(len(ref_trigs))}
        lowfreq_hyp = {tri for tri, count in freq_hyp.items() if count <= _freq_limit(len(hyp_trigs))}
        candidate_trigs = lowfreq_ref.intersection(lowfreq_hyp)
    anchors: List[Tuple[int, int]] = []
    j_last = 0
    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i + 1], ref_tok[i + 2])
        if tri not in candidate_trigs:
            continue
        for j in range(j_last, len(hyp_tok) - 2):
            if (hyp_tok[j], hyp_tok[j + 1], hyp_tok[j + 2]) == tri:
                anchors.append((i, j))
                j_last = j + 3
                break
    filtered: List[Tuple[int, int]] = []
    last_i, last_j = -1, -1
    for i, j in anchors:
        if i > last_i and j > last_j:
            filtered.append((i, j))
            last_i, last_j = i, j
    return filtered


COMMON_THRESHOLD = 0.05


def extract_word_list(text: str, max_words: int = 50) -> List[str]:
    """Return frequent and noteworthy words from ``text`` for ASR prompting."""
    tokens = normalize(text).split()
    counts = Counter(t for t in tokens if t not in STOP and len(t) > 3)
    if len(tokens) >= 100:
        max_common = len(tokens) * COMMON_THRESHOLD
        counts = Counter({w: c for w, c in counts.items() if c <= max_common})
    first_pos: Dict[str, int] = {tok: i for i, tok in reversed(list(enumerate(tokens)))}
    ordered = sorted(counts.items(), key=lambda x: (-x[1], first_pos.get(x[0], 0)))
    proper_raw = re.findall(r"\b[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ']*\b", text)
    proper_tokens: List[str] = list(dict.fromkeys(
        normalize(tok) for tok in proper_raw if len(normalize(tok)) > 3 and normalize(tok) not in STOP
    ))
    result = proper_tokens[:]
    for tok, _ in ordered:
        if tok not in result:
            result.append(tok)
        if len(result) >= max_words:
            break
    return result[:max_words]


def find_repeated_sequences(
    text: str,
    min_length: int = 5,
    max_length: int = 50,
    similarity_threshold: float = 85.0,
) -> List[str]:
    """Finds consecutive repeated or near-repeated sequences of words in a text."""
    original_words = text.split()
    lower_words = text.lower().split()
    n = len(lower_words)
    if n < min_length * 2:
        return []

    potential_matches = []
    # Step 1: Find the best match for each possible starting position `i`
    for i in range(n - min_length * 2 + 1):
        best_match_for_i = None
        for length in range(min_length, min(max_length, n - i) + 1):
            seq1_lower = " ".join(lower_words[i: i + length])
            search_start = i + length
            search_end = min(n, search_start + 5)

            for j in range(search_start, search_end - length + 2):
                if j + length > n:
                    continue
                seq2_lower = " ".join(lower_words[j: j + length])
                score = fuzz.ratio(seq1_lower, seq2_lower)

                if score >= similarity_threshold:
                    if (best_match_for_i is None or
                        score > best_match_for_i["score"] or
                        (score == best_match_for_i["score"] and length > best_match_for_i["length"])):
                        best_match_for_i = {"i": i, "j": j, "length": length, "score": score}

        if best_match_for_i:
            potential_matches.append(best_match_for_i)

    # Step 2: Resolve overlaps by prioritizing the best matches
    potential_matches.sort(key=lambda m: (m["score"], m["length"]), reverse=True)

    results = []
    used_indices = set()
    for match in potential_matches:
        i, j, length = match["i"], match["j"], match["length"]
        is_overlapping = any(k in used_indices for k in range(i, i + length)) or \
                         any(k in used_indices for k in range(j, j + length))

        if not is_overlapping:
            seq1_orig = " ".join(original_words[i: i + length])
            seq2_orig = " ".join(original_words[j: j + length])
            results.append({"text": f"{seq1_orig} ... {seq2_orig}", "i": i})
            for k in range(i, i + length):
                used_indices.add(k)
            for k in range(j, j + length):
                used_indices.add(k)

    # Step 3: Sort final results based on their appearance order
    results.sort(key=lambda r: r["i"])
    return [r["text"] for r in results]

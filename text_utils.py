"""Utilities for text normalization and trigram anchors."""

from pathlib import Path
from typing import List, Tuple, Dict
import re
import unicodedata
from collections import Counter

import unidecode
import pdfplumber
from rapidfuzz import fuzz, process

# stopwords set and digit names reused across modules
STOP = {
    "de",
    "la", "el", "y", "que", "en", "a", "los", "se", "del", "por", "con", "las", "un", "para", "una",
    "su", "al", "lo", "como", "más", "o", "pero", "sus", "le", "ya", "fue", "punto", "puntos", "coma",
}

# weight for mismatching stop words in DTW
STOP_WEIGHT = 0.2

DIGIT_NAMES = {
    "0": "cero", "1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco",
    "6": "seis", "7": "siete", "8": "ocho", "9": "nueve", "10": "diez", "11": "once",
    "12": "doce", "13": "trece", "14": "catorce", "15": "quince", "16": "dieciseis",
    "17": "diecisiete", "18": "dieciocho", "19": "diecinueve", "20": "veinte",
}

# frequency threshold for trigram anchors
ANCHOR_MAX_FREQ = 2

def normalize(text: str, strip_punct: bool = True) -> str:
    """Lowercase, remove accents and optionally strip punctuation."""
    text = text.replace('“', ' " ').replace('”', ' " ')
    text = re.sub(r'([.,;?!])', r' \1 ', text)
    text = unidecode.unidecode(text.lower())
    text = re.sub(r"\b([a-z])\.\b", r"\1", text)
    abbr = {
        "dr": "doctor", "dra": "doctora", "sr": "senor", "sra": "senora",
        "srta": "senorita", "esq": "escribano",
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

def token_equal(a: str, b: str) -> bool:
    """Return True if tokens are very similar or digit equivalents."""
    from rapidfuzz.distance import Levenshtein
    def _base(t: str) -> str:
        t = unicodedata.normalize("NFKD", t)
        t = t.encode("ascii", "ignore").decode("ascii")
        return t.casefold()
    if _base(a) == _base(b):
        return True
    if (len(a) == 2 and a[1] == "." and a[0].isalpha() and len(b) == 1 and b == a[0]) or \
       (len(b) == 2 and b[1] == "." and b[0].isalpha() and len(a) == 1 and a == b[0]):
        return True
    punct_map = {".": {"punto", "puntos"}, ",": {"coma"}, ";": {"punto y coma"}}
    for sym, words in punct_map.items():
        if (a == sym and b in words) or (b == sym and a in words):
            return True
    if Levenshtein.normalized_distance(a, b) <= 0.2:
        return True
    if a.isdigit() and DIGIT_NAMES.get(a) == b:
        return True
    if b.isdigit() and DIGIT_NAMES.get(b) == a:
        return True
    return False

def find_anchor_trigrams(ref_tok: List[str], hyp_tok: List[str]) -> List[Tuple[int, int]]:
    """Locate monotonic trigram anchors present at low frequency."""
    ref_trigs: List[Tuple[str, str, str]] = []
    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i + 1], ref_tok[i + 2])
        if any(t in STOP for t in tri): continue
        ref_trigs.append(tri)
    hyp_trigs: List[Tuple[str, str, str]] = []
    for j in range(len(hyp_tok) - 2):
        tri = (hyp_tok[j], hyp_tok[j + 1], hyp_tok[j + 2])
        if any(t in STOP for t in tri): continue
        hyp_trigs.append(tri)
    freq_ref = Counter(ref_trigs)
    freq_hyp = Counter(hyp_trigs)
    lowfreq_ref = {tri for tri, count in freq_ref.items() if count <= ANCHOR_MAX_FREQ}
    lowfreq_hyp = {tri for tri, count in freq_hyp.items() if count <= ANCHOR_MAX_FREQ}
    candidate_trigs = lowfreq_ref.intersection(lowfreq_hyp)
    anchors: List[Tuple[int, int]] = []
    j_last = 0
    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i + 1], ref_tok[i + 2])
        if tri not in candidate_trigs: continue
        for j in range(j_last, len(hyp_tok) - 2):
            if (hyp_tok[j], hyp_tok[j+1], hyp_tok[j+2]) == tri:
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
    proper_raw = re.findall(r"\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]*\b", text)
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
            seq1_lower = " ".join(lower_words[i : i + length])
            search_start = i + length
            search_end = min(n, search_start + 5)

            for j in range(search_start, search_end - length + 2):
                if j + length > n:
                    continue
                seq2_lower = " ".join(lower_words[j : j + length])
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
            seq1_orig = " ".join(original_words[i : i + length])
            seq2_orig = " ".join(original_words[j : j + length])
            results.append({"text": f"{seq1_orig} ... {seq2_orig}", "i": i})
            for k in range(i, i + length):
                used_indices.add(k)
            for k in range(j, j + length):
                used_indices.add(k)

    # Step 3: Sort final results based on their appearance order
    results.sort(key=lambda r: r["i"])
    return [r["text"] for r in results]

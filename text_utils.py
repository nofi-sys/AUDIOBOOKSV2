"""Utilities for text normalization and trigram anchors."""

from pathlib import Path
from typing import List, Tuple, Dict
import re

import unidecode
import pdfplumber

# stopwords set and digit names reused across modules
STOP = {
    "de",
    "la",
    "el",
    "y",
    "que",
    "en",
    "a",
    "los",
    "se",
    "del",
    "por",
    "con",
    "las",
    "un",
    "para",
    "una",
    "su",
    "al",
    "lo",
    "como",
    "más",
    "o",
    "pero",
    "sus",
    "le",
    "ya",
    "fue",
    "punto",
    "puntos",
    "coma",
}

# weight for mismatching stop words in DTW
STOP_WEIGHT = 0.2

DIGIT_NAMES = {
    "0": "cero",
    "1": "uno",
    "2": "dos",
    "3": "tres",
    "4": "cuatro",
    "5": "cinco",
    "6": "seis",
    "7": "siete",
    "8": "ocho",
    "9": "nueve",
    "10": "diez",
    "11": "once",
    "12": "doce",
    "13": "trece",
    "14": "catorce",
    "15": "quince",
    "16": "dieciseis",
    "17": "diecisiete",
    "18": "dieciocho",
    "19": "diecinueve",
    "20": "veinte",
}

# frequency threshold for trigram anchors
ANCHOR_MAX_FREQ = 2


def normalize(text: str, strip_punct: bool = True) -> str:
    """Lowercase, remove accents and optionally strip punctuation."""

    text = unidecode.unidecode(text.lower())
    if not strip_punct:
        # unify spelled punctuation with symbols for easier matching
        text = re.sub(r"\bpunto y coma\b", ";", text)
        text = re.sub(r"\bpunto\b", ".", text)
        text = re.sub(r"\bpuntos\b", ".", text)
        text = re.sub(r"\bcoma\b", ",", text)
    if strip_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def read_script(path: str) -> str:
    """Read text from a PDF or TXT file."""

    p = Path(path)
    if p.suffix.lower() == ".pdf":
        with pdfplumber.open(p) as pdf:
            pages = [pg.extract_text() or "" for pg in pdf.pages]
        raw = "\n".join(pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa un TXT.")
        return raw
    return p.read_text(encoding="utf8", errors="ignore")


def token_equal(a: str, b: str) -> bool:
    """Return True if tokens are very similar or digit equivalents."""

    from rapidfuzz.distance import Levenshtein

    if a == b:
        return True
    # consider punctuation words equivalent to symbols
    punct_map = {
        ".": {"punto", "puntos"},
        ",": {"coma"},
        ";": {"punto y coma"},
    }
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


def find_anchor_trigrams(
    ref_tok: List[str], hyp_tok: List[str]
) -> List[Tuple[int, int]]:
    """Locate monotonic trigram anchors present at low frequency."""

    ref_trigs: List[Tuple[str, str, str]] = []
    hyp_trigs: List[Tuple[str, str, str]] = []

    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i + 1], ref_tok[i + 2])
        if tri[0] in STOP or tri[1] in STOP or tri[2] in STOP:
            continue
        ref_trigs.append(tri)

    for j in range(len(hyp_tok) - 2):
        tri = (hyp_tok[j], hyp_tok[j + 1], hyp_tok[j + 2])
        if tri[0] in STOP or tri[1] in STOP or tri[2] in STOP:
            continue
        hyp_trigs.append(tri)

    freq_ref: Dict[Tuple[str, str, str], int] = {}
    freq_hyp: Dict[Tuple[str, str, str], int] = {}
    for tri in ref_trigs:
        freq_ref[tri] = freq_ref.get(tri, 0) + 1
    for tri in hyp_trigs:
        freq_hyp[tri] = freq_hyp.get(tri, 0) + 1

    lowfreq_ref = {tri for tri, count in freq_ref.items() if count <= ANCHOR_MAX_FREQ}
    lowfreq_hyp = {tri for tri, count in freq_hyp.items() if count <= ANCHOR_MAX_FREQ}
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

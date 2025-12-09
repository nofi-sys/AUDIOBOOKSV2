"""Build a glossary of proper names and entities from a script using spaCy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from text_utils import read_script
from phonetic_utils import strip_accents


PREFERRED_MODELS = ("es_dep_news_trf", "es_core_news_lg")


def _normalize_token(token: str) -> str:
    """Lowercase, strip accents, remove punctuation."""
    t = strip_accents(token.lower())
    t = re.sub(r"[^\w]+", "", t)
    return t


def _category_for_ent(label: str) -> str:
    if label.upper() in {"PERSON", "PER"}:
        return "person"
    if label.upper() == "ORG":
        return "org"
    if label.upper() in {"GPE", "LOC"}:
        return "place"
    return "person"


def _category_for_propn(tokens_norm: Sequence[str]) -> str:
    heuristics_place = {"rio", "cerro", "monte", "villa", "ciudad", "puerto", "san", "santa"}
    heuristics_org = {"sociedad", "fundacion", "asociacion", "universidad", "club"}
    if any(tok in heuristics_place for tok in tokens_norm):
        return "place"
    if any(tok in heuristics_org for tok in tokens_norm):
        return "org"
    return "person"


def load_spacy_model(preferred: Sequence[str] | None = None):
    """Try to load the preferred Spanish model, falling back gracefully."""
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("spaCy is required for glossary extraction. Install with `pip install spacy`.") from exc

    errors: Dict[str, str] = {}
    for name in preferred or PREFERRED_MODELS:
        try:
            return spacy.load(name)
        except Exception as exc:  # pragma: no cover - best effort loader
            errors[name] = str(exc)
    msgs = "; ".join(f"{name}: {err}" for name, err in errors.items())
    raise RuntimeError(
        "No se pudo cargar un modelo de spaCy. Instala es_dep_news_trf o es_core_news_lg. "
        f"Errores: {msgs}"
    )


def build_glossary(
    text: str,
    *,
    nlp=None,
    priority: float = 1.0,
    max_propn_len: int = 4,
) -> List[dict]:
    """
    Extract entities and PROPN runs from ``text`` and return glossary entries.

    Output format:
        {
            "tokens_raw": [...],
            "tokens_norm": [...],
            "category": "person|place|org|technical_term",
            "priority": float
        }
    """
    model = nlp or load_spacy_model()
    doc = model(text)

    entries: Dict[tuple[str, ...], dict] = {}
    covered: set[int] = set()

    def _add_entry(tokens_raw: Iterable[str], tokens_norm: List[str], category: str, pri: float, rank: int) -> None:
        key = tuple(tokens_norm)
        if not key:
            return
        existing = entries.get(key)
        if existing and (existing.get("_rank", 99) < rank or (existing.get("_rank", 99) == rank and existing["priority"] >= pri)):
            return
        entries[key] = {
            "tokens_raw": list(tokens_raw),
            "tokens_norm": tokens_norm,
            "category": category,
            "priority": float(pri),
            "_rank": rank,
        }

    for ent in doc.ents:
        if ent.label_.upper() not in {"PERSON", "PER", "ORG", "GPE", "LOC"}:
            continue
        tokens_raw = [t.text for t in ent]
        tokens_norm = [_normalize_token(t.text) for t in ent if _normalize_token(t.text)]
        _add_entry(tokens_raw, tokens_norm, _category_for_ent(ent.label_), priority, rank=0)
        covered.update(t.i for t in ent)

    i = 0
    while i < len(doc):
        tok = doc[i]
        if tok.i in covered or tok.pos_ != "PROPN":
            i += 1
            continue
        run: List = []
        while (
            i < len(doc)
            and doc[i].pos_ == "PROPN"
            and doc[i].i not in covered
            and len(run) < max_propn_len
        ):
            run.append(doc[i])
            i += 1
        tokens_raw = [t.text for t in run]
        tokens_norm = [_normalize_token(t.text) for t in run if _normalize_token(t.text)]
        if not tokens_norm:
            continue
        category = _category_for_propn(tokens_norm)
        _add_entry(tokens_raw, tokens_norm, category, priority, rank=1)

    glossary = []
    for entry in entries.values():
        entry.pop("_rank", None)
        glossary.append(entry)
    glossary.sort(key=lambda e: (e["category"], e["tokens_norm"]))
    return glossary


def save_glossary(glossary: Sequence[dict], output_path: str | Path) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI helper
    parser = argparse.ArgumentParser(description="Construir glosario desde un guion usando spaCy.")
    parser.add_argument("input", help="Ruta al guion (TXT o PDF).")
    parser.add_argument("-o", "--output", help="Ruta de salida para glossary.json.")
    parser.add_argument("--model", help="Nombre de modelo spaCy; por defecto intenta es_dep_news_trf y es_core_news_lg.")
    parser.add_argument("--priority", type=float, default=1.0, help="Prioridad base para las entradas.")
    parser.add_argument("--max-propn-len", type=int, default=4, help="Maximo de tokens en secuencias PROPN.")
    args = parser.parse_args()

    text = read_script(args.input)
    preferred = (args.model,) if args.model else PREFERRED_MODELS
    nlp = load_spacy_model(preferred)
    glossary = build_glossary(text, nlp=nlp, priority=args.priority, max_propn_len=args.max_propn_len)
    out_path = Path(args.output) if args.output else Path(args.input).with_name(
        f"{Path(args.input).stem}_glossary.json"
    )
    save_glossary(glossary, out_path)
    print(f"Glosario generado: {len(glossary)} entradas -> {out_path}")


if __name__ == "__main__":
    main()

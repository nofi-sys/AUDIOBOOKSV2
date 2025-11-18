from __future__ import annotations

import json
import copy
import os
import random
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Set

from .jerga_pipeline import JergaPipelineCoordinator, JergaReport

MIN_TERM_LENGTH = 3
MIN_TERM_FREQUENCY = 3
MAX_CONTEXT_SNIPPETS = 3
CONTEXT_WINDOW = 120

DEFAULT_TRAP_WORDS = [
    "cheap shot",
    "shot",
    "deal",
    "hell",
    "goddamn",
    "goddamned",
    "mob",
    "bagman",
    "horsewire",
    "alky",
    "callgirl",
    "syndic",
]

TRANSLATION_LOG_SEPARATOR = "=" * 20
AGGREGATE_TOKEN = "<<<SEGMENT_BREAK>>>"
AGGREGATE_SEPARATOR = f"\n{AGGREGATE_TOKEN}\n"

TRANSLATABLE_BLOCK_TYPES = {"p", "blockquote", "subtitle", "poem", "letter"}
REVIEW_MAX_COMPLETION_TOKENS = 10000  # Mantener amplio para evitar recortes de modelo.
REVIEW_VERDICT_ONLY_MAX_TOKENS = 10000  # Evitar limites reducidos que corten dictamenes.
# Define paths relative to the project root for robustness
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEW_PROMPTS_DIR = _PROJECT_ROOT / "prompts" / "translation_review"
DEFAULT_REVIEW_PROMPT = REVIEW_PROMPTS_DIR / "es_literary_core.md"
REVIEW_OBSERVATIONS_MODULE = REVIEW_PROMPTS_DIR / "es_observaciones_module.md"

PUNCTUATION_MODULE_FILENAME = "punctuation_and_graphic_signs_normalizer.json"
PUNCTUATION_MODULE_PATH = _PROJECT_ROOT / "prompts" / PUNCTUATION_MODULE_FILENAME
PUNCTUATION_MODULE_NAME = "PunctuationAndGraphicSignsNormalizer"

GLOSSARY_MODULE_FILENAME = "glossary_curation_module.json"
GLOSSARY_MODULE_PATH = _PROJECT_ROOT / "prompts" / GLOSSARY_MODULE_FILENAME
GLOSSARY_MODULE_NAME = "GlossaryCurationModule"


def _parse_translation_log(log_path: Path) -> List[Tuple[str, str]]:
    if not log_path or not log_path.exists():
        return []
    try:
        content = log_path.read_text(encoding="utf-8")
    except Exception:
        return []

    pattern = re.compile(
        r"--- ORIGINAL ---\s*(.*?)\s*--- TRADUCIDO ---\s*(.*?)\s*"
        + re.escape(TRANSLATION_LOG_SEPARATOR)
        + r"\s*",
        re.S,
    )
    entries: List[Tuple[str, str]] = []
    for match in pattern.finditer(content):
        original = match.group(1).strip("\n")
        translated = match.group(2).strip("\n")
        entries.append((original, translated))
    return entries


def _normalize_for_resume(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def _is_incomplete_translation(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped.startswith("[[TRADUCTION_ERROR")


_API_TRACE_LISTENERS: List[Callable[[str], None]] = []

def register_api_trace_listener(listener: Callable[[str], None]) -> None:
    """Register a callable that receives a message whenever an API call is announced."""
    if listener not in _API_TRACE_LISTENERS:
        _API_TRACE_LISTENERS.append(listener)

def unregister_api_trace_listener(listener: Callable[[str], None]) -> None:
    """Remove a previously registered API trace listener."""
    if listener in _API_TRACE_LISTENERS:
        _API_TRACE_LISTENERS.remove(listener)


def _announce_api_call(model: str, purpose: str, *, extra: Optional[str] = None) -> None:
    """Emit a lightweight trace whenever we call the language model."""
    details = f"[API CALL] model={model} purpose={purpose}"
    if extra:
        details += f" | {extra}"
    print(details)
    for listener in list(_API_TRACE_LISTENERS):
        try:
            listener(details)
        except Exception:
            continue


def _record_token_usage(
    token_callback: Optional[Callable[..., None]],
    usage: Any,
    *,
    model: str,
    purpose: str,
) -> None:
    """
    Normalise and forward token accounting information to the registered callback.
    This helper tolerates the different field names returned by various OpenAI APIs.
    """
    if not token_callback or not usage:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    cached_tokens = (
        getattr(usage, "cached_prompt_tokens", None)
        or getattr(usage, "cached_input_tokens", None)
        or getattr(usage, "cached_tokens", None)
    )
    total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens:
        token_callback(prompt_tokens, model=model, type="input", purpose=purpose)
    if completion_tokens:
        token_callback(completion_tokens, model=model, type="output", purpose=purpose)
    if cached_tokens:
        token_callback(cached_tokens, model=model, type="cached_input", purpose=purpose)
    if not (prompt_tokens or completion_tokens) and total_tokens:
        # Old-style responses may only provide a grand total.
        token_callback(total_tokens, model=model, type="output", purpose=purpose)


def _load_review_prompt_base() -> str:
    """Loads the base prompt from the filesystem, with robust error handling."""
    fallback_prompt = (
        "Actua como un traductor y editor literario profesional hispanohablante. "
        "Evalua fidelidad al texto original, naturalidad en espanol y cumplimiento del glosario. "
        "Revisa concordancias de genero y numero, corrige calcos idiomaticos, evita repeticiones y rimas no deseadas, "
        "y conserva jerga o registros especiales detectados en el original. "
        "Devuelve un objeto JSON con las claves: estado (ok|dudoso|mal), revision (texto final en espanol) "
        "y observaciones (solo cuando el estado no sea ok, maximo 30 palabras). "
        "Si el estado es ok, la revision debe ser la traduccion original sin cambios."
    )
    if not DEFAULT_REVIEW_PROMPT.exists():
        print(f"ERROR: Review prompt not found at '{DEFAULT_REVIEW_PROMPT}'. Using fallback.")
        return fallback_prompt

    try:
        return DEFAULT_REVIEW_PROMPT.read_text(encoding="utf-8").strip()
    except OSError as e:
        print(f"ERROR: Failed to read review prompt '{DEFAULT_REVIEW_PROMPT}': {e}. Using fallback.")
        return fallback_prompt


REVIEW_PROMPT_BASE = _load_review_prompt_base()
def _load_optional_prompt(path: Path) -> str:
    """Load an optional prompt module, returning an empty string if missing."""
    if not path.exists():
        print(f"WARNING: Review prompt module not found at '{path}'.")
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"WARNING: Failed to read review prompt module '{path}': {exc}")
        return ""


REVIEW_OBSERVATIONS_PROMPT = _load_optional_prompt(REVIEW_OBSERVATIONS_MODULE)
STYLE_PROFILES: Dict[str, Dict[str, str]] = {
    "literary_noir": {
        "label": "Literario",
        "tone": (
            "El tono de esta obra es ciencia ficcion noir: duro, cinico y con un trasfondo criminal. "
            "Mantiene un registro directo y natural, evitando la rigidez academica."
        ),
        "idioms": (
            "No traduzcas literalmente los modismos ni la jerga. Busca equivalentes idiomaticos en espanol "
            "que transmitan la misma intencion, aunque las palabras cambien por completo. Ejemplo: \"cheap shot\" -> \"oportunidad facil\"."
        ),
        "syntax": (
            "Respeta el ritmo del autor. Conserva frases cortas y punzantes cuando aparezcan, "
            "y no simplifiques las estructuras complejas si son parte del estilo original."
        ),
    },
    "technical_formal": {
        "label": "Tecnico",
        "tone": (
            "El tono de esta obra es tecnico y formal. Utiliza terminologia precisa y un registro profesional."
        ),
        "idioms": (
            "Evita expresiones coloquiales. Prefiere explicaciones claras y exactas para cualquier modismo o jerga."
        ),
        "syntax": (
            "Prioriza la claridad y la estructura logica. Divide frases excesivamente largas si mejoran la legibilidad."
        ),
    },
    "essay_academic": {
        "label": "Ensayo",
        "tone": (
            "El tono de esta obra es de ensayo academico. Mantiene un registro formal, con precision terminologica y rigor conceptual."
        ),
        "idioms": (
            "Traduce los modismos a expresiones cultas o neutraliza su tono para preservar la claridad intelectual."
        ),
        "syntax": (
            "Respeta la estructura argumentativa y conserva los conectores logicos que articulan el discurso."
        ),
    },
}
try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Doc = None  # type: ignore

try:
    from .api_logger import log_interaction, log_simplified_translation
except ImportError:
    from api_logger import log_interaction, log_simplified_translation  # type: ignore

try:
    import openai
except ImportError:
    openai = None

class GlossaryBuilder:
    """Builds a glossary of key terms from the source text for translation consistency."""

    def __init__(
        self,
        client: Optional['openai.OpenAI'] = None,
        token_callback: Optional[Callable[..., None]] = None,
        nlp_model: str = "en_core_web_trf",
    ) -> None:
        self.nlp = self._load_spacy_model(nlp_model)
        self.client = client
        self.token_callback = token_callback
        self.term_policies: Dict[str, Dict[str, str]] = {}
        self.term_metadata: Dict[str, Dict[str, Any]] = {}
        self.curated_contexts: Dict[str, List[str]] = {}
        self._glossary_module_cache: Optional[Dict[str, Any]] = None
        self.glossary_policy: Dict[str, Any] = {}

    def _load_spacy_model(self, model: str) -> Optional['spacy.Language']:
        if not spacy:
            return None
        try:
            return spacy.load(model)
        except OSError:
            print(f"Spacy model '{model}' not found. Downloading...")
            try:
                spacy.cli.download(model)
                return spacy.load(model)
            except Exception as exc:
                print(f"Failed to download and load spacy model '{model}': {exc}")
                return None

    def _load_glossary_module(self) -> Optional[Dict[str, Any]]:
        if self._glossary_module_cache is not None:
            return self._glossary_module_cache
        if not GLOSSARY_MODULE_PATH.exists():
            print(f"WARNING: Glossary module not found at '{GLOSSARY_MODULE_PATH}'.")
            self._glossary_module_cache = None
            return None
        try:
            with GLOSSARY_MODULE_PATH.open("r", encoding="utf-8") as fh:
                self._glossary_module_cache = json.load(fh)
        except Exception as exc:
            print(f"WARNING: Failed to load glossary module '{GLOSSARY_MODULE_PATH}': {exc}")
            self._glossary_module_cache = None
        return self._glossary_module_cache

    def _default_glossary_policy(self) -> Dict[str, Any]:
        module = self._load_glossary_module() or {}
        template = module.get("policy_template", {})
        try:
            return copy.deepcopy(template)
        except Exception:
            return dict(template)

    @staticmethod
    def _normalize_key(term: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", term.lower())

    def _clean_term(self, term: str) -> Optional[str]:
        if not term:
            return None
        cleaned = re.sub(r"\s+", " ", term.strip())
        cleaned = re.sub(r"[-\u2013\u2014]+$", "", cleaned).strip()
        cleaned = re.sub(r"^[-\u2013\u2014]+", "", cleaned).strip()
        if len(cleaned) < MIN_TERM_LENGTH:
            return None
        if not re.search(r"[a-zA-Z]", cleaned):
            return None
        if cleaned.isdigit():
            return None
        return cleaned

    @staticmethod
    def _collect_message_text(response: Any) -> str:
        """Extract the textual content from a chat completion response."""
        try:
            message = response.choices[0].message  # type: ignore[index]
        except (AttributeError, IndexError):
            return ""
        content = getattr(message, "content", "")
        if isinstance(content, list):
            fragments: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    fragments.append(str(part.get("text", "")))
                else:
                    fragments.append(str(part))
            content = "".join(fragments)
        return (content or "").strip()

    def _add_term(self, store: Dict[str, str], term: str) -> Optional[str]:
        cleaned = self._clean_term(term)
        if not cleaned:
            return None
        key = self._normalize_key(cleaned)
        if not key:
            return None
        existing = store.get(key)
        if not existing or len(cleaned) > len(existing):
            store[key] = cleaned
            existing = cleaned
        return existing

    def _build_term_contexts(self, text: str, terms: List[str]) -> Dict[str, List[str]]:
        contexts: Dict[str, List[str]] = {}
        for term in terms:
            snippets: List[str] = []
            if not term:
                contexts[term] = snippets
                continue
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            for match in pattern.finditer(text):
                start = max(match.start() - CONTEXT_WINDOW, 0)
                end = min(match.end() + CONTEXT_WINDOW, len(text))
                snippet = text[start:end].replace("\n", " ").strip()
                if snippet:
                    snippets.append(snippet)
                if len(snippets) >= MAX_CONTEXT_SNIPPETS:
                    break
            contexts[term] = snippets
        return contexts

    @staticmethod
    def _chunk_terms(terms: List[str], size: int) -> List[List[str]]:
        if size <= 0:
            size = 25
        return [terms[i : i + size] for i in range(0, len(terms), size)]
    def _fallback_extract_terms(self, text: str, top_n: int = 50) -> Dict[str, str]:
        cap_seq = re.findall(r"(?:[A-Z][a-z]+(?:['â€™][A-Za-z]+)?(?:\s+[A-Z][a-z]+(?:['â€™][A-Za-z]+)?)+)", text)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
        counter: Counter[str] = Counter()
        for raw in cap_seq + acronyms:
            cleaned = self._clean_term(raw)
            if cleaned:
                counter[cleaned] += 1
        items = [(term, freq) for term, freq in counter.items() if freq >= 2]
        items.sort(key=lambda x: (-x[1], x[0]))
        selected = [term for term, _freq in items[:top_n]]
        return {term: "" for term in selected}

    def extract_terms(self, text: str, top_n: int = 50) -> Dict[str, str]:
        if not self.nlp:
            self.term_metadata = {}
            print("spaCy model not available. Falling back to regex-based glossary extraction.")
            return self._fallback_extract_terms(text, top_n=top_n)

        doc: Doc = self.nlp(text)
        candidate_terms: Dict[str, str] = {}
        self.term_metadata = {}

        approx_tokens = max(1, len(text) // 6)
        dynamic_min_freq = 3 if approx_tokens > 8000 else 2

        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                canonical = self._add_term(candidate_terms, ent.text)
                if not canonical:
                    continue
                meta = self.term_metadata.setdefault(
                    canonical,
                    {"sources": set(), "entity_labels": set(), "pos_tags": set(), "frequency": 0},
                )
                meta["sources"].add("entity")
                meta["entity_labels"].add(ent.label_)
                meta["frequency"] = meta.get("frequency", 0) + 1

        stopwords = self.nlp.Defaults.stop_words
        noun_counter: Counter[str] = Counter()
        noun_pos_map: Dict[str, set] = {}

        for chunk in getattr(doc, "noun_chunks", []):
            phrase = self._clean_term(chunk.text)
            if not phrase:
                continue
            tokens = [tok for tok in chunk if tok.is_alpha]
            if not tokens:
                continue
            sw_ratio = sum(1 for tok in tokens if tok.is_stop or tok.text.lower() in stopwords) / len(tokens)
            if sw_ratio >= 0.5:
                continue
            noun_counter[phrase] += 1
            noun_pos_map.setdefault(phrase, set()).add("NOUN_CHUNK")

        for token in doc:
            if token.is_stop or token.text.lower() in stopwords:
                continue
            if not token.is_alpha:
                continue
            if token.pos_ == "PROPN":
                cleaned = self._clean_term(token.text)
                if cleaned:
                    noun_counter[cleaned] += 1
                    noun_pos_map.setdefault(cleaned, set()).add("PROPN")

        candidates = [(term, freq) for term, freq in noun_counter.items() if freq >= dynamic_min_freq]
        candidates.sort(key=lambda x: (-x[1], x[0]))

        for term, freq in candidates[:top_n]:
            canonical = self._add_term(candidate_terms, term)
            if not canonical:
                continue
            meta = self.term_metadata.setdefault(
                canonical,
                {"sources": set(), "entity_labels": set(), "pos_tags": set(), "frequency": 0},
            )
            meta["sources"].add("noun")
            meta["pos_tags"].update(noun_pos_map.get(term, set()))
            meta["frequency"] = max(meta.get("frequency", 0), freq)

        return {term: "" for term in sorted(candidate_terms.values())}
    def curate_terms(self, terms: Dict[str, str], text: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
        policy = self._default_glossary_policy()
        if not terms:
            self.glossary_policy = policy
            self.curated_contexts = {}
            self.term_policies = {}
            return {"policy": policy, "entries": []}

        ordered_terms = list(dict.fromkeys(sorted(terms.keys())))
        contexts = self._build_term_contexts(text, ordered_terms)
        contexts_by_clean: Dict[str, List[str]] = {}
        for term, snippets in contexts.items():
            clean = self._clean_term(term)
            if clean:
                contexts_by_clean[clean] = snippets
        self.curated_contexts = contexts_by_clean

        module = self._load_glossary_module() or {}
        taxonomy = module.get("output_requirements", {}).get("taxonomy", [])
        taxonomy_set = {item.lower() for item in taxonomy}
        prompt_sections = module.get("prompt_sections", {})
        role_message = module.get("role", "")
        instructions_block = prompt_sections.get("instructions", "")
        curation_rules = module.get("curation_rules", [])
        entry_schema = module.get("output_requirements", {}).get("entry_schema", {})

        def _fallback_entries() -> Dict[str, Any]:
            fallback_entries: List[Dict[str, Any]] = []
            for term in ordered_terms:
                translation = term
                cleaned = self._clean_term(term) or term
                fallback_entries.append(
                    {
                        "lemma": cleaned,
                        "translation": translation,
                        "action": "keep",
                        "category": "concept",
                        "rationale": "fallback_keep_no_ai",
                        "surface_variants": [{"source": term, "target": translation}],
                        "context_rules": "",
                        "constraints": "",
                    }
                )
            self.glossary_policy = policy
            self.term_policies = {entry["lemma"]: entry for entry in fallback_entries}
            return {"policy": policy, "entries": fallback_entries}

        if not self.client:
            print("Warning: OpenAI client not configured. Using fallback glossary curation.")
            return _fallback_entries()

        print(f"--- Glossary Curator: Using model {model} ---")
        aggregated_entries: Dict[str, Dict[str, Any]] = {}

        for batch in self._chunk_terms(ordered_terms, 8):
            payload = []
            for term in batch:
                meta = self.term_metadata.get(term, {})
                payload.append(
                    {
                        "term": term,
                        "contexts": contexts.get(term, []),
                        "occurrences": len(contexts.get(term, [])),
                        "metadata": {
                            "sources": sorted(list(meta.get("sources", []))) if meta else [],
                            "entity_labels": sorted(list(meta.get("entity_labels", []))) if meta else [],
                            "pos_tags": sorted(list(meta.get("pos_tags", []))) if meta else [],
                            "frequency": meta.get("frequency", 0) if meta else 0,
                            "is_all_caps": term.isupper(),
                            "has_space": " " in term,
                        },
                    }
                )

            policy_json = json.dumps(policy, ensure_ascii=False, indent=2)
            taxonomy_line = ", ".join(taxonomy)
            schema_json = json.dumps(entry_schema, ensure_ascii=False, indent=2)
            rules_block = "\n".join(f"- {rule}" for rule in curation_rules)
            candidates_json = json.dumps(payload, ensure_ascii=False, indent=2)

            prompt_parts = [
                instructions_block.strip(),
                "",
                f"Taxonomía permitida: {taxonomy_line}",
                "",
                "Estructura de cada entrada:",
                schema_json,
                "",
                "Política editorial de referencia (reprodúcela sin cambios en la clave \"policy\" del JSON final):",
                policy_json,
                "",
                "Reglas adicionales:",
                rules_block,
                "",
                "Candidatos con contexto:",
                candidates_json,
                "",
                "Devuelve un único objeto JSON con las claves \"policy\" y \"entries\". No incluyas ningún texto adicional.",
                "Si descartas un candidato, simplemente no lo menciones.",
            ]
            prompt = "\n".join(part for part in prompt_parts if part)

            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": role_message or "You generate JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 8000,
                "response_format": {"type": "json_object"},
            }

            purpose = "glossary_curation"
            try:
                _announce_api_call(model, purpose, extra=f"batch={len(batch)}")
                response = self.client.chat.completions.create(**params)
                log_interaction(model, prompt, params, response=response)
                _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)
                content = self._collect_message_text(response)
                if not content:
                    raise ValueError("Empty response from curator")
                parsed = json.loads(content)
                entries = parsed.get("entries")
                if not isinstance(entries, list):
                    raise ValueError("Glossary curator must return a list in 'entries'.")
                for raw_entry in entries:
                    sanitized = self._sanitize_glossary_entry(raw_entry, taxonomy_set)
                    if not sanitized:
                        continue
                    aggregated_entries[sanitized["lemma"]] = sanitized
            except Exception as exc:
                print(f"Error during glossary curation batch: {exc}")
                log_interaction(model, prompt, params, error=exc)
                # Fallback to keep entries for this batch
                for term in batch:
                    cleaned = self._clean_term(term) or term
                    aggregated_entries.setdefault(
                        cleaned,
                        {
                            "lemma": cleaned,
                            "translation": cleaned,
                            "action": "keep",
                            "category": "concept",
                            "rationale": "fallback_batch_error",
                            "surface_variants": [{"source": term, "target": cleaned}],
                            "context_rules": "",
                            "constraints": "",
                        },
                    )

        entries_list = sorted(aggregated_entries.values(), key=lambda item: item["lemma"])
        self.glossary_policy = policy
        self.term_policies = {entry["lemma"]: entry for entry in entries_list}
        return {"policy": policy, "entries": entries_list}

    def _sanitize_glossary_entry(self, entry: Any, taxonomy: Set[str]) -> Optional[Dict[str, Any]]:
        if not isinstance(entry, dict):
            return None
        lemma = entry.get("lemma")
        if not isinstance(lemma, str) or not lemma.strip():
            return None
        lemma = lemma.strip()
        action = str(entry.get("action", "keep")).lower()
        if action not in {"keep", "translate", "normalize"}:
            action = "keep"
        category = str(entry.get("category", "concept")).lower().strip()
        if category not in taxonomy:
            category = "concept"
        translation = entry.get("translation")
        if not isinstance(translation, str) or not translation.strip():
            translation = lemma if action != "translate" else lemma
        translation = translation.strip()

        surface_variants_raw = entry.get("surface_variants") or []
        surface_variants: List[Dict[str, str]] = []
        if isinstance(surface_variants_raw, list):
            for variant in surface_variants_raw:
                if not isinstance(variant, dict):
                    continue
                source = variant.get("source")
                target = variant.get("target")
                if isinstance(source, str) and source.strip():
                    surface_variants.append(
                        {
                            "source": source.strip(),
                            "target": target.strip() if isinstance(target, str) and target.strip() else translation,
                        }
                    )
        if not surface_variants:
            surface_variants.append({"source": lemma, "target": translation})

        context_rules = entry.get("context_rules", "")
        constraints = entry.get("constraints", "")

        return {
            "lemma": lemma,
            "translation": translation,
            "action": action,
            "category": category,
            "rationale": (entry.get("rationale") or "").strip(),
            "surface_variants": surface_variants,
            "context_rules": context_rules.strip() if isinstance(context_rules, str) else "",
            "constraints": constraints.strip() if isinstance(constraints, str) else "",
        }

    @staticmethod
    def entries_to_translation_map(entries: List[Dict[str, Any]]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in entries:
            lemma = entry.get("lemma")
            translation = entry.get("translation", "")
            if isinstance(lemma, str) and isinstance(translation, str) and lemma and translation:
                mapping.setdefault(lemma, translation)
            variants = entry.get("surface_variants") or []
            if isinstance(variants, list):
                for variant in variants:
                    if not isinstance(variant, dict):
                        continue
                    source = variant.get("source")
                    target = variant.get("target", translation)
                    if isinstance(source, str) and isinstance(target, str) and source and target:
                        mapping.setdefault(source, target)
        return mapping

    def translate_glossary(
        self,
        terms: Dict[str, Dict[str, Any]],
        text: str,
        model: str = "gpt-5-mini",
    ) -> Dict[str, Dict[str, Any]]:
        if not self.client:
            print("Warning: OpenAI client not configured. Skipping glossary translation.")
            return terms

        print(f"--- Glossary Translator: Using model {model} ---")
        if not terms:
            return {}

        ordered_terms = list(dict.fromkeys(sorted(terms.keys())))
        contexts = self._build_term_contexts(text, ordered_terms)
        policies = self.term_policies if self.term_policies else {}
        final_results: Dict[str, Dict[str, Any]] = {}
        translation_payload: Dict[str, Dict[str, Any]] = {}

        for term in ordered_terms:
            metadata = terms.get(term, {}) or {}
            policy = policies.get(
                term,
                {
                    "action": metadata.get("action", "translate"),
                    "category": metadata.get("category", "other"),
                    "rationale": metadata.get("rationale", ""),
                    "raw_term": term,
                },
            )
            action = policy.get("action", metadata.get("action", "translate"))
            base_entry = {
                "translation": metadata.get("translation", ""),
                "action": action,
                "category": policy.get("category", metadata.get("category", "other")),
                "rationale": policy.get("rationale", metadata.get("rationale", "")),
                "frequency": metadata.get("frequency", len(self.curated_contexts.get(term, []))),
                "examples": metadata.get(
                    "examples",
                    (self.curated_contexts.get(term, []) or contexts.get(term, []))[:MAX_CONTEXT_SNIPPETS],
                ),
            }
            if action == "ignore":
                print(f"Skipping term '{term}' per curation policy (ignore).")
                continue
            if action == "keep":
                translation = base_entry.get("translation") or term
                base_entry["translation"] = translation
                final_results[term] = base_entry
                print(f"Policy keep: retaining original term '{term}'")
                continue
            translation_payload[term] = {
                "contexts": (contexts.get(term, []) or self.curated_contexts.get(term, []))[:MAX_CONTEXT_SNIPPETS],
                "category": base_entry.get("category", "other"),
                "rationale": base_entry.get("rationale", ""),
            }
            final_results[term] = base_entry

        if not translation_payload:
            return final_results

        for batch in self._chunk_terms(list(translation_payload.keys()), 10):
            batch_payload = {term: translation_payload[term] for term in batch}
            prompt_lines = [
                "Eres un traductor y editor literario. Debes proponer traducciones consistentes para nombres propios, lugares y terminos clave de una novela.",
                "Aplica estas reglas:",
                "1. Nombres propios de personas y apellidos: mantenlos en ingles salvo que el glosario existente especifique otra cosa.",
                "2. Lugares y organizaciones: traduce solo si existe una forma ampliamente aceptada en espanol; en caso contrario, conserva el original.",
                "3. Terminos de jerga o conceptos: ofrece una traduccion precisa que preserve el tono y el sentido.",
                "Devuelve unicamente un objeto JSON con la forma:",
                "{",
                '  "terminos": [',
                "    {",
                '      "term": "<texto original>",',
                '      "translation": "<traduccion al espanol>"',
                "    }",
                "  ]",
                "}",
                "No agregues comentarios ni texto adicional.",
                "",
                "TERMINOS Y CONTEXTO:",
                json.dumps(batch_payload, ensure_ascii=False),
            ]
            prompt = "\n".join(prompt_lines)

            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a translation editor who responds with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "max_completion_tokens": 10000,
                "response_format": {"type": "json_object"},
            }

            purpose = "glossary_translation"
            try:
                _announce_api_call(model, purpose, extra=f"batch={len(batch)}")
                response = self.client.chat.completions.create(**params)
                log_interaction(model, prompt, params, response=response)
                finish_reason = response.choices[0].finish_reason
                content = self._collect_message_text(response) if finish_reason == "stop" else ""
                _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)
                if not content:
                    raise ValueError("Empty response from glossary translator")
                payload = json.loads(content)
                entries = payload.get("terminos") if isinstance(payload, dict) else None
                if not isinstance(entries, list):
                    raise ValueError("Glossary translation response must include una lista en la clave 'terminos'.")
                translations: Dict[str, str] = {}
                for entry in entries:
                    if isinstance(entry, dict):
                        term = entry.get("term")
                        translation = entry.get("translation")
                        if isinstance(term, str) and isinstance(translation, str):
                            translations[term] = translation.strip()
                for term in batch:
                    candidate = (translations.get(term, "") or "").strip()
                    if not candidate:
                        print(f"Glossary translation empty for '{term}', removing term from glossary.")
                        final_results.pop(term, None)
                        continue
                    final_results[term]["translation"] = candidate
                    print(f"Translated '{term}' -> '{candidate}'")
            except Exception as exc:
                print(f"Error translating glossary batch: {exc}")
                log_interaction(model, prompt, params, error=exc)
                for term in batch:
                    final_results.pop(term, None)

        return final_results

class SummaryGenerator:
    """Generates a global summary of the book to provide context to the translator."""

    def __init__(self, client: Optional['openai.OpenAI'] = None, token_callback: Optional[Callable[..., None]] = None):
        self.client = client
        if self.client is None and openai and os.getenv("OPENAI_API_KEY"):
            self.client = openai.OpenAI()
        self.token_callback = token_callback

    def generate(self, text: str, max_words: int = 5000, model: str = "gpt-5-mini") -> str:
        """Generates a summary of the text using an AI model."""
        if not self.client:
            return "AI client not available. Summary generation skipped."

        print(f"--- Summary Generator: Using model {model} ---")
        sample_text = text[:max_words]
        prompt = (
            "Eres un asistente literario. A continuacion te proporciono el inicio de una novela. "
            "Genera un resumen de no mas de 200 palabras que describa el genero (ciencia ficcion, fantasia, drama), "
            "el tono (oscuro, humoristico, epico), el estilo de escritura (simple, barroco) y la trama principal. "
            "Este resumen se usara como contexto para un traductor."
            f"\n\n--- TEXTO ---\n{sample_text}\n\n--- RESUMEN ---"
        )

        try:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 300,
            }
            purpose = "summary_generation"
            _announce_api_call(model, purpose, extra=f"max_tokens={params['max_completion_tokens']}")
            response = self.client.chat.completions.create(**params)
            log_interaction(model, prompt, params, response=response)

            _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log_interaction(model, prompt, params, error=exc)
            return f"Error generating summary: {exc}"

class TranslationCaller:
    """Manages the process of chunking text and calling the translation API."""

    def __init__(self, client: Optional['openai.OpenAI'] = None, token_callback: Optional[Callable[..., None]] = None):
        self.client = client
        if self.client is None and openai and os.getenv("OPENAI_API_KEY"):
            self.client = openai.OpenAI()
        self.token_callback = token_callback

    def _build_dynamic_prompt(
        self,
        chunk: str,
        glossary_subset: Any,
        prev_context: str,
        next_context: str,
        static_prompt_prefix: str,
        segments_count: int = 1,
        segment_token: Optional[str] = None,
    ) -> str:
        """Builds the full prompt by combining the static prefix with dynamic content."""
        if isinstance(glossary_subset, dict):
            glossary_payload = glossary_subset
        elif isinstance(glossary_subset, list):
            glossary_payload = {"entries": glossary_subset}
        else:
            glossary_payload = {"entries": []}
        glossary_subset_str = json.dumps(glossary_payload, ensure_ascii=False, separators=(",", ":"))

        parts = [
            static_prompt_prefix,
            "Glosario específico para este fragmento (subconjunto del glosario principal):\n",
            f"{glossary_subset_str}\n",
            "Contexto inmediato (no traducir, solo referencia):\n",
            f"- Anterior: {prev_context or '(sin texto)'}\n",
            f"- Posterior: {next_context or '(sin texto)'}\n\n",
            "Texto a traducir:\n",
            f"{chunk}\n\n",
            "Instrucciones clave:\n",
            "- Devuelve solo el texto traducido, sin comentarios.\n",
            "- Evita anglicismos y expresiones no naturales en español.\n",
            "- Asegúrate de que todas las palabras y frases estén traducidas.\n",
            "- Cumple estrictamente ambos glosarios (general y específico del fragmento).\n",
        ]
        if segments_count > 1 and segment_token:
            parts.append(
                f"- Mantén exactamente {segments_count} bloques en el mismo orden. "
                f"Separa cada bloque con la cadena '{segment_token}'.\n"
            )
        return "".join(parts)

    def translate_chunk(
        self,
        chunk: str,
        glossary_subset: Any,
        prev_context: str,
        next_context: str,
        static_prompt_prefix: str,
        model: str = "gpt-5-mini",
        simplified_log_path: Optional[Path] = None,
        segments_count: int = 1,
        segment_token: Optional[str] = None,
        prompt_cache_key: Optional[str] = None,
    ) -> str:
        if not self.client:
            raise ConnectionError("OpenAI client is not initialized.")

        print(f"--- Translation Caller: Using model {model} ---")
        prompt = self._build_dynamic_prompt(
            chunk,
            glossary_subset,
            prev_context,
            next_context,
            static_prompt_prefix,
            segments_count=segments_count,
            segment_token=segment_token,
        )

        try:
            params = {
                "model": model,
                "messages": [{"role": "system", "content": prompt}],
                "prompt_cache_key": prompt_cache_key,
            }
            purpose = "translation_chunk"
            _announce_api_call(model, purpose)
            response = self.client.chat.completions.create(**params)
            log_interaction(model, prompt, params, response=response)

            _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)

            translated_text = response.choices[0].message.content.strip()

            return translated_text
        except Exception as exc:
            print(f"Error during translation API call: {exc}")
            log_interaction(model, prompt, params, error=exc)
            return f"[[TRADUCTION_ERROR: {exc}]]"

class ConsistencyChecker:
    """Verifies that the translated text respects the provided glossary."""

    @staticmethod
    def _normalize_glossary(glossary: Any) -> Dict[str, str]:
        if isinstance(glossary, dict) and "entries" in glossary:
            entries = glossary.get("entries")
            if isinstance(entries, list):
                return GlossaryBuilder.entries_to_translation_map(entries)
            return {}
        if isinstance(glossary, list):
            entries = [entry for entry in glossary if isinstance(entry, dict)]
            return GlossaryBuilder.entries_to_translation_map(entries)
        if isinstance(glossary, dict):
            return {str(k): str(v) for k, v in glossary.items()}
        return {}

    def check(self, translated_text: str, glossary: Any) -> List[str]:
        alerts: List[str] = []
        mapping = self._normalize_glossary(glossary)
        for original, translation in mapping.items():
            if not translation or original.lower() == translation.lower():
                continue
            if original in translated_text:
                alerts.append(
                    f"Inconsistency found: The original term '{original}' was found in the translated text. "
                    f"It should haber sido traducido a '{translation}'."
                )
        return alerts

    def check_block(self, translated_text: str, glossary: Any) -> List[str]:
        alerts: List[str] = []
        if not translated_text:
            return alerts
        mapping = self._normalize_glossary(glossary)
        lower_text = translated_text.lower()
        for original, translation in mapping.items():
            if not translation or original.lower() == translation.lower():
                continue
            if original.lower() in lower_text and translation.lower() not in lower_text:
                alerts.append(
                    f"Term '{original}' appears without the expected translation '{translation}'."
                )
        return alerts

class TranslationQA:
    """Performs automated quality checks on translated content."""

    def __init__(
        self,
        client: Optional['openai.OpenAI'] = None,
        token_callback: Optional[Callable[..., None]] = None,
        trap_words: Optional[List[str]] = None,
    ) -> None:
        self.client = client
        if self.client is None and openai and os.getenv("OPENAI_API_KEY"):
            self.client = openai.OpenAI()
        self.token_callback = token_callback
        self.trap_words = trap_words or DEFAULT_TRAP_WORDS

    @staticmethod
    def _normalize_glossary(glossary: Any) -> Dict[str, str]:
        if isinstance(glossary, dict) and "entries" in glossary:
            entries = glossary.get("entries")
            if isinstance(entries, list):
                return GlossaryBuilder.entries_to_translation_map(entries)
            return {}
        if isinstance(glossary, list):
            entries = [entry for entry in glossary if isinstance(entry, dict)]
            return GlossaryBuilder.entries_to_translation_map(entries)
        if isinstance(glossary, dict):
            return {str(k): str(v) for k, v in glossary.items()}
        return {}

    def detect_calques(self, translated_text: str) -> List[Dict[str, Any]]:
        findings: List[Dict[str, Any]] = []
        if not translated_text:
            return findings
        lowered = translated_text.lower()
        for trap in self.trap_words:
            pattern = re.escape(trap.lower())
            matches = list(re.finditer(pattern, lowered))
            if not matches:
                continue
            snippets: List[str] = []
            for match in matches:
                start = max(match.start() - 40, 0)
                end = min(match.end() + 40, len(translated_text))
                snippets.append(translated_text[start:end].replace("\n", " ").strip())
            findings.append({"term": trap, "count": len(matches), "snippets": snippets})
        return findings

    def _run_qa_prompt(
        self,
        original_text: str,
        translated_text: str,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        prompt_lines = [
            "Eres un auditor de calidad para traducciones literarias. A continuacion se muestra un texto original y su traduccion al espanol.",
            f"TEXTO ORIGINAL:\n\"{original_text}\"",
            f"TRADUCCION:\n\"{translated_text}\"",
            "Evalua la traduccion en dos areas:",
            "1. Fidelidad de tono.",
            "2. Manejo de modismos o jerga.",
            "Responde unicamente con un objeto JSON con el formato:",
            "{",
            '  "fidelidad_tono": "Buena|Regular|Mala",',
            '  "calcos_detectados": [',
            '    {"original": "", "traduccion_literal": "", "sugerencia_mejora": ""}',
            '  ]',
            "}",
            "Usa una lista vacia si no detectas calcos.",
        ]
        prompt = "\n".join(prompt_lines)

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a QA auditor who returns JSON only."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": 250,
        }

        purpose = "qa_prompt"
        try:
            _announce_api_call(model, purpose, extra=f"max_tokens={params.get('max_completion_tokens')}")
            response = self.client.chat.completions.create(**params)
            log_interaction(model, prompt, params, response=response)
            _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)
            content = (response.choices[0].message.content or "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("QA response must be a JSON object.")
            return parsed
        except Exception as exc:
            log_interaction(model, prompt, params, error=exc)
            print(f"QA sampling failed: {exc}")
            return None

    def review_block_prompt_template(
        self,
        original_text: str,
        translated_text: str,
        glossary: Any,
        *,
        include_observations: bool = True,
        verdict_only: bool = False,
    ) -> str:
        relevant_terms: Dict[str, str] = {}
        mapping = self._normalize_glossary(glossary)
        lower_original = original_text.lower()
        lower_translated = translated_text.lower()
        for term, target in mapping.items():
            if not target:
                continue
            if term.lower() in lower_original or term.lower() in lower_translated or target.lower() in lower_translated:
                relevant_terms[term] = target
            if len(relevant_terms) >= 20:
                break

        glossary_json = json.dumps(relevant_terms, ensure_ascii=False, indent=2)
        max_tokens_hint = REVIEW_VERDICT_ONLY_MAX_TOKENS if verdict_only else REVIEW_MAX_COMPLETION_TOKENS

        prompt_lines = [
            REVIEW_PROMPT_BASE,
            "",
            (
                "Objetivo de salida: entrega un JSON valido sin superar "
                f"{max_tokens_hint} tokens. Ajusta la respuesta al esquema solicitado."
            ),
            "",
            f"Glosario obligatorio:\n{glossary_json}",
            "",
            "Texto original:",
            original_text,
            "",
            "Traduccion a revisar:",
            translated_text,
        ]

        prompt_lines.append("")
        if verdict_only:
            prompt_lines.extend(
                [
                    "### Dictamen rapido (solo estado)",
                    "- Devuelve un objeto JSON que contenga exclusivamente la clave `estado` con uno de los valores permitidos: `ok`, `dudoso` o `mal`.",
                    "- No agregues otras claves salvo que las instrucciones adicionales indiquen que debes devolver `revision` (por ejemplo, para ajustar un encabezado).",
                    "- Si anades `revision`, omite `observaciones` y limita los cambios al ajuste tipografico requerido.",
                ]
            )
            include_observations = False
        elif include_observations:
            if REVIEW_OBSERVATIONS_PROMPT:
                prompt_lines.extend(["", REVIEW_OBSERVATIONS_PROMPT])
            else:
                prompt_lines.extend(
                    [
                        "### Observaciones breves",
                        "- Utiliza `observaciones` solo para describir la correccion o justificar una duda concreta (maximo 30 palabras).",
                        "- Prefiere dejarlas vacias si la modificacion es evidente por si misma.",
                    ]
                )
        else:
            prompt_lines.extend(
                [
                    "### Observaciones desactivadas",
                    "- No incluyas la clave `observaciones`. Si el esquema JSON la exige, devuelvela como cadena vacia.",
                    "- Registra los ajustes necesarios exclusivamente en `revision`.",
                ]
            )

        if not verdict_only:
            prompt_lines.extend(
                [
                    "",
                    "### Reglas para `revision`",
                    "- Si `estado` es `ok`, devuelve `revision` como cadena vacia para indicar que no hubo cambios.",
                    "- Si detectas un error, corrige solo los segmentos necesarios dentro de `revision`, respetando el tono y el contenido del original.",
                ]
            )

        return "\n".join(prompt_lines)


    def review_block(
        self,
        original_text: str,
        translated_text: str,
        glossary: Any,
        model: Optional[str] = None,
        extra_guidance: Optional[str] = None,
        include_observations: bool = True,
        verdict_only: bool = False,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not self.client:
            return None, "Cliente no inicializado"
        active_model = model or "gpt-5-mini"
        normalized_glossary = self._normalize_glossary(glossary)
        prompt = self.review_block_prompt_template(
            original_text,
            translated_text,
            normalized_glossary,
            include_observations=include_observations,
            verdict_only=verdict_only,
        )
        if extra_guidance:
            prompt = f"{prompt}\n\nInstrucciones adicionales de formato:\n{extra_guidance.strip()}"

        # No fijamos temperature explÃ­cita: el modelo usa defaults mÃ¡s estables para revisiÃ³n.
        max_tokens = REVIEW_VERDICT_ONLY_MAX_TOKENS if verdict_only else REVIEW_MAX_COMPLETION_TOKENS
        params = {
            "model": active_model,
            "messages": [
                {"role": "system", "content": "Eres un editor literario que responde unicamente con un objeto JSON valido."},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        content = ""
        purpose = "translation_review"
        try:
            print("--- PROMPT PARA REVISION DE IA ---")
            print(prompt)
            print("---------------------------------")
            _announce_api_call(active_model, purpose, extra=f"max_tokens={max_tokens}")
            response = self.client.chat.completions.create(**params)
            log_interaction(active_model, prompt, params, response=response)
            _record_token_usage(self.token_callback, response.usage, model=active_model, purpose=purpose)

            content = (response.choices[0].message.content or "").strip()
            print(f"--- RESPUESTA CRUDA DE IA ---\n{content}\n-----------------------------")
            if not content:
                raise ValueError("Respuesta vacia del modelo")
            data = json.loads(content)

            status = str(data.get("estado", "dudoso")).lower()
            if status not in {"ok", "dudoso", "mal"}:
                status = "dudoso"

            observaciones = str(data.get("observaciones", "")).strip()
            revision_value = data.get("revision", "")
            if not isinstance(revision_value, str):
                revision_value = ""

            revision_text = revision_value.strip()
            translated_baseline = translated_text.strip()

            if not include_observations:
                observaciones = ""

            if verdict_only:
                if not revision_text:
                    revision_text = ""
            elif status == "ok":
                observaciones = ""
                revision_text = ""
            else:
                if not revision_text:
                    revision_text = translated_baseline
                if revision_text.strip() == translated_baseline and status != "mal":
                    status = "ok"
                    observaciones = ""
                    revision_text = ""

            result = {"status": status, "observaciones": observaciones, "revision": revision_text}
            return result, content

        except Exception as exc:
            log_interaction(active_model, prompt, params, error=exc)
            print(f"AI review failed: {exc}")
            raise Exception(f"Contenido recibido antes del error: {content}") from exc

    def sample_reviews(
        self,
        originals: List[str],
        translations: List[str],
        model: Optional[str],
        sample_ratio: float = 0.05,
        max_samples: int = 8,
    ) -> List[Dict[str, Any]]:
        if not self.client or not model:
            return []
        pair_count = min(len(originals), len(translations))
        if pair_count == 0:
            return []
        sample_size = max(1, int(pair_count * sample_ratio))
        sample_size = min(sample_size, max_samples, pair_count)
        rng = random.Random(42)
        indices = sorted(rng.sample(range(pair_count), sample_size))
        reviews: List[Dict[str, Any]] = []
        for idx in indices:
            original = originals[idx]
            translated = translations[idx]
            result = self._run_qa_prompt(original, translated, model=model)
            if result is None:
                continue
            reviews.append(
                {
                    "index": idx,
                    "fidelidad_tono": result.get("fidelidad_tono"),
                    "calcos_detectados": result.get("calcos_detectados", []),
                }
            )
        return reviews

    def evaluate(
        self,
        original_blocks: List[str],
        translated_blocks: List[str],
        qa_model: Optional[str] = None,
        sample_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        translated_text = "\n\n".join(translated_blocks)
        trap_hits = self.detect_calques(translated_text)
        reviews = self.sample_reviews(original_blocks, translated_blocks, model=qa_model, sample_ratio=sample_ratio)
        return {
            "trap_hits": trap_hits,
            "qa_reviews": reviews,
            "qa_model": qa_model if reviews else None,
            "sample_ratio": sample_ratio,
            "trap_word_count": sum(hit["count"] for hit in trap_hits),
        }

class TranslationEngine:
    """Orchestrates the entire translation process from preparation to execution."""

    def __init__(self, api_key: Optional[str] = None, token_callback: Optional[Callable[..., None]] = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not set. AI features will be disabled.")
            self.client = None
        else:
            self.client = openai.OpenAI(api_key=api_key) if openai else None

        self.token_callback = token_callback
        self.glossary_builder = GlossaryBuilder(client=self.client, token_callback=self.token_callback)
        self.summary_generator = SummaryGenerator(client=self.client, token_callback=self.token_callback)
        self.translation_caller = TranslationCaller(client=self.client, token_callback=self.token_callback)
        self.consistency_checker = ConsistencyChecker()
        self.translation_qa = TranslationQA(client=self.client, token_callback=self.token_callback)
        self.jerga_pipeline = JergaPipelineCoordinator(client=self.client, token_callback=self.token_callback)
        self._last_translation_cancelled = False
        self._last_translation_failure_reason: Optional[str] = None
        try:
            attempts = int(os.getenv("TXT2MD_TRANSLATION_MAX_ATTEMPTS", "2"))
        except ValueError:
            attempts = 2
        self._translation_max_attempts = max(1, attempts)
        try:
            delay_value = float(os.getenv("TXT2MD_TRANSLATION_RETRY_DELAY_SECONDS", "60"))
        except ValueError:
            delay_value = 60.0
        self._translation_retry_delay = max(0.0, delay_value)
        self._punctuation_module_cache: Optional[Dict[str, Any]] = None
        self.last_punctuation_module_used: Optional[str] = None
        self.last_jerga_report: Optional[Dict[str, Any]] = None
        self._last_jerga_report_obj: Optional[JergaReport] = None

    def _load_punctuation_module(self) -> Optional[Dict[str, Any]]:
        """Load and cache the optional punctuation normalization module."""
        if self._punctuation_module_cache is not None:
            return self._punctuation_module_cache

        if not PUNCTUATION_MODULE_PATH.exists():
            print(f"WARNING: Punctuation module not found at '{PUNCTUATION_MODULE_PATH}'.")
            self._punctuation_module_cache = None
            return None

        try:
            with PUNCTUATION_MODULE_PATH.open("r", encoding="utf-8") as fh:
                self._punctuation_module_cache = json.load(fh)
        except Exception as exc:
            print(f"WARNING: Failed to load punctuation module '{PUNCTUATION_MODULE_PATH}': {exc}")
            self._punctuation_module_cache = None
        return self._punctuation_module_cache

    def _build_static_prompt_prefix(
        self,
        summary: str,
        glossary: Any,
        style_profile: str,
        style_notes: str = "",
        punctuation_guidance: Optional[str] = None,
        jerga_report: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Builds the static part of the prompt for translation."""
        if isinstance(glossary, dict):
            glossary_payload = glossary
        elif isinstance(glossary, list):
            glossary_payload = {"entries": glossary}
        else:
            glossary_payload = {"entries": []}
        glossary_str = json.dumps(glossary_payload, ensure_ascii=False, separators=(",", ":"))

        style = STYLE_PROFILES.get(style_profile, STYLE_PROFILES["literary_noir"])
        style_summary = f"Tono: {style['tone']} | Idiomas/Jerga: {style['idioms']} | Sintaxis: {style['syntax']}"

        punctuation_section = ""
        if punctuation_guidance:
            cleaned_guidance = punctuation_guidance.strip()
            if cleaned_guidance:
                punctuation_section = f"Modulo puntuacion: {cleaned_guidance}\n"

        user_notes = f"Notas usuario: {style_notes.strip()}\n" if style_notes else ""

        jerga_section = ""
        if jerga_report:
            jerga_str = json.dumps(jerga_report, ensure_ascii=False, separators=(",", ":"))
            jerga_section = f"Informe de jerga: {jerga_str}\n"

        parts = [
            "Actua como traductor literario profesional en espanol. Mantén fidelidad de tono y ritmo.\n",
            f"Resumen global:\n{summary}\n",
            f"Guia de estilo ({style['label']}): {style_summary}\n",
            user_notes,
            punctuation_section,
            jerga_section,
            "Glosario obligatorio (usa exactamente estas traducciones; si esta vacio no hay terminos forzados):\n",
            f"{glossary_str}\n",
        ]

        inflexible_rules = []
        if isinstance(glossary, dict):
            policy = glossary.get("policy", {})
            if isinstance(policy, dict):
                formatting_rules = policy.get("formatting")
                if isinstance(formatting_rules, str) and "cursiva" in formatting_rules.lower():
                    inflexible_rules.append(
                        "- REGLA DE CURSIVA: Aplica cursiva a nombres de vehículos y títulos de obras, según la política del glosario."
                    )

        inflexible_rules.append(
            "- REGLA DE DIÁLOGO: Usa SIEMPRE la raya de diálogo (—) para todos los diálogos. NUNCA uses comillas inglesas (\") ni angulares («»)."
        )

        if inflexible_rules:
            parts.append("\nReglas Inflexibles (prioridad máxima):\n")
            parts.extend(f"{rule}\n" for rule in inflexible_rules)

        return "".join(parts)

    def _describe_retry_reason(original_text: str, translated_text: str) -> str:
        if _is_incomplete_translation(translated_text):
            return "la API devolvio un marcador de error"
        if not translated_text.strip():
            return "la API devolvio una cadena vacia"
        norm_original = _normalize_for_resume(original_text).lower()
        norm_translated = _normalize_for_resume(translated_text).lower()
        original_words = self._tokenize_words(norm_original)
        translated_words = self._tokenize_words(norm_translated)
        if len(original_words) > 10 and len(translated_words) < len(original_words) * 0.5:
            return "la traduccion es sospechosamente corta"
        if norm_original == norm_translated:
            return "la respuesta es identica al original"
        return "la respuesta mantiene la mayor parte del texto original"

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        return re.findall(r"[A-Za-zÁÉÍÓÚáéíóúñÑüÜ]+", text.lower())

    @staticmethod
    def _contains_url_or_email(text: str) -> bool:
        lowered = text.lower()
        if any(token in lowered for token in ("http://", "https://", "www.", "ftp://")):
            return True
        if "@" in text and re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
            return True
        return False

    def _needs_retry(self, original_text: str, translated_text: str) -> bool:
        if _is_incomplete_translation(translated_text):
            return True
        norm_original = _normalize_for_resume(original_text)
        norm_translated = _normalize_for_resume(translated_text)
        if not norm_translated:
            return True
        if not norm_original:
            return False
        if self._contains_url_or_email(norm_original):
            return False
        original_words = self._tokenize_words(norm_original)
        if len(original_words) <= 3:
            return False
        translated_words = self._tokenize_words(norm_translated)
        if not translated_words:
            return True
        # Ratio de palabras de 0.5 a 0.4
        if len(original_words) > 12 and len(translated_words) < len(original_words) * 0.4:
            return True
        if norm_original.lower() == norm_translated.lower():
            return True
        # Aumentar el umbral de superposición de palabras de 0.7 a 0.8
        if len(translated_words) >= 5:
            original_set = set(original_words)
            overlap = sum(1 for word in translated_words if word in original_set)
            if overlap / len(translated_words) > 0.8:
                return True
        return False

    def _wait_before_retry(self) -> None:
        if self._translation_retry_delay <= 0:
            return
        delay = self._translation_retry_delay
        rounded = int(delay)
        if rounded == delay:
            human_delay = f"{rounded} segundo{'s' if rounded != 1 else ''}"
        else:
            human_delay = f"{delay:.1f} segundos"
        print(f"Esperando {human_delay} antes de reintentar...")
        time.sleep(delay)

    def _register_translation_failure(self, message: str) -> None:
        self._last_translation_failure_reason = message
        print(f"ERROR: {message}")

    def _translate_block_with_retry(
        self,
        original_text: str,
        *,
        glossary_payload: Dict[str, Any],
        prev_context: str,
        next_context: str,
        static_prompt_prefix: str,
        model: str,
        prompt_cache_key: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        attempts = 0
        last_output: Optional[str] = None
        while attempts < self._translation_max_attempts:
            translated_text = self.translation_caller.translate_chunk(
                chunk=original_text,
                glossary_subset=glossary_payload,
                prev_context=prev_context,
                next_context=next_context,
                static_prompt_prefix=static_prompt_prefix,
                model=model,
                simplified_log_path=None,
                prompt_cache_key=prompt_cache_key,
            )
            last_output = translated_text
            if not self._needs_retry(original_text, translated_text):
                return translated_text, None
            attempts += 1
            reason = self._describe_retry_reason(original_text, translated_text)
            print(
                f"Advertencia: traduccion incompleta detectada ({reason}). "
                f"Intento {attempts} de {self._translation_max_attempts}."
            )
            if attempts < self._translation_max_attempts:
                self._wait_before_retry()
        final_reason = self._describe_retry_reason(original_text, last_output or "")
        return None, final_reason

    def review_block(
        self,
        original_text: str,
        translated_text: str,
        glossary: Any,
        model: Optional[str] = None,
        extra_guidance: Optional[str] = None,
        include_observations: bool = True,
        verdict_only: bool = False,
    ) -> Optional[Dict[str, Any]]:
        return self.translation_qa.review_block(
            original_text,
            translated_text,
            glossary,
            model=model,
            extra_guidance=extra_guidance,
            include_observations=include_observations,
            verdict_only=verdict_only,
        )

    def run_preparation(
        self,
        document: List[Dict[str, Any]],
        output_dir: Path,
        input_filename: str,
        model: str = "gpt-5-mini",
        glossary_curation_model: Optional[str] = None,
        jerga_config: Optional[Dict[str, Any]] = None,
        reuse_glossary_path: Optional[Path] = None,
    ) -> Dict[str, Path]:
        text = "\n\n".join([block.get("text", "") for block in document])
        base_filename = Path(input_filename).stem
        artifact_paths: Dict[str, Path] = {}

        # 1. Glossary Curation
        glossary_path = output_dir / f"{base_filename}_glossary.json"

        if reuse_glossary_path and reuse_glossary_path.exists():
            print(f"Reusing existing glossary from: {reuse_glossary_path}")
            glossary_path = reuse_glossary_path
            artifact_paths["glossary"] = glossary_path
        else:
            if not self.glossary_builder.nlp:
                print("spaCy model not available. Using fallback glossary extraction.")
            suggestions = self.glossary_builder.extract_terms(text)
            if not suggestions:
                print("No glossary terms were extracted.")
            curated_model = glossary_curation_model or model
            glossary_structure = (
                self.glossary_builder.curate_terms(suggestions, text, model=curated_model)
                if suggestions
                else {"policy": self.glossary_builder._default_glossary_policy(), "entries": []}
            )
            if not isinstance(glossary_structure, dict):
                glossary_structure = {"policy": self.glossary_builder._default_glossary_policy(), "entries": []}

            try:
                glossary_path.write_text(json.dumps(glossary_structure, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Glossary suggestions saved to {glossary_path}")
                artifact_paths["glossary"] = glossary_path
            except Exception as exc:
                print(f"Error saving glossary suggestions: {exc}")

        # 2. Summary Generation
        print("Generando resumen del libro...")
        summary = self.summary_generator.generate(text, model=model)
        summary_path = output_dir / f"{base_filename}_summary.txt"
        try:
            summary_path.write_text((summary or "").strip() + "\n", encoding="utf-8")
            print(f"Resumen guardado en: {summary_path}")
            artifact_paths["summary"] = summary_path
        except Exception as exc:
            print(f"No se pudo guardar el resumen en archivo: {exc}")

        # 3. Jerga/Slang Report
        if jerga_config:
            try:
                print("Ejecutando pipeline de jerga...")
                jerga_report_obj = self.jerga_pipeline.run(document, config=jerga_config)
                self.last_jerga_report = jerga_report_obj.to_dict()
                jerga_report_path = output_dir / f"{base_filename}_jerga_report.json"
                jerga_report_path.write_text(json.dumps(self.last_jerga_report, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Informe de jerga guardado en: {jerga_report_path}")
                artifact_paths["jerga_report"] = jerga_report_path
            except Exception as exc:
                print(f"WARNING: slang detection failed during preparation: {exc}")

        return artifact_paths

    @staticmethod
    def _select_glossary_subset(text: str, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not text or not entries:
            return []
        lowered = text.lower()
        selected: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for entry in entries:
            lemma = entry.get("lemma")
            action = str(entry.get("action", "translate")).lower()
            include = False

            if action == "normalize":
                include = True
            if not include and isinstance(lemma, str):
                if "{" in lemma or "}" in lemma:
                    include = True
                elif lemma.lower() in lowered:
                    include = True

            if not include:
                variants = entry.get("surface_variants") or []
                if isinstance(variants, list):
                    for variant in variants:
                        if not isinstance(variant, dict):
                            continue
                        source = variant.get("source")
                        if isinstance(source, str) and source.strip():
                            if source.lower() in lowered:
                                include = True
                                break

            if include:
                key = lemma if isinstance(lemma, str) else None
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                selected.append(entry)

        return selected

    def run_translation(
        self,
        document: List[Dict[str, Any]],
        glossary: Any,
        summary: str,
        style_profile: str,
        style_notes: str = "",
        use_punctuation_module: bool = False,
        model: str = "gpt-5-mini",
        simplified_log_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        resume: bool = False,
        max_translated_blocks: Optional[int] = None,
        aggregate_mode: bool = True,
        aggregate_target_words: int = 2500,
        cancel_event: Optional[threading.Event] = None,
        detect_jerga: bool = False,
        jerga_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self._last_translation_cancelled = False
        self._last_translation_failure_reason = None
        punctuation_guidance = ""
        self.last_punctuation_module_used = None
        self.last_jerga_report = None
        self._last_jerga_report_obj = None
        if use_punctuation_module:
            module = self._load_punctuation_module()
            if module:
                module_name = module.get("module", PUNCTUATION_MODULE_NAME)
                module_version = module.get("version")
                module_goal = (module.get("goal") or "").strip()
                module_prompt = (module.get("prompt_for_operators") or "").strip()
                guidance_segments = []
                header_line = module_name
                if module_version:
                    header_line = f"{module_name} v{module_version}"
                guidance_segments.append(header_line)
                if module_goal:
                    guidance_segments.append(module_goal)
                if module_prompt:
                    guidance_segments.append(module_prompt)
                punctuation_guidance = "\n\n".join(segment for segment in guidance_segments if segment).strip()
                self.last_punctuation_module_used = header_line
            else:
                print("WARNING: Punctuation module requested but could not be loaded.")

        if detect_jerga and document:
            try:
                jerga_report = self.jerga_pipeline.run(document, config=jerga_config)
                self._last_jerga_report_obj = jerga_report
                self.last_jerga_report = jerga_report.to_dict()
            except Exception as exc:
                print(f"WARNING: slang detection failed: {exc}")

        glossary_entries: List[Dict[str, Any]] = []
        if isinstance(glossary, dict) and "entries" in glossary:
            raw_entries = glossary.get("entries") or []
            if isinstance(raw_entries, list):
                glossary_entries = [entry for entry in raw_entries if isinstance(entry, dict)]
        elif isinstance(glossary, list):
            glossary_entries = [entry for entry in glossary if isinstance(entry, dict)]

        if detect_jerga and self._last_jerga_report_obj:
            for item in self._last_jerga_report_obj.validated:
                translation = (item.translation or "").strip()
                if not translation:
                    continue
                lemma = item.text.strip()
                if not lemma:
                    continue
                glossary_entries.append({
                    "lemma": lemma, "translation": translation, "action": "translate", "category": "jerga",
                    "rationale": item.notes or "Detectada automáticamente como jerga.",
                    "surface_variants": [{"source": lemma, "target": translation}],
                })

        static_prompt_prefix = self._build_static_prompt_prefix(
            summary=summary,
            glossary=glossary,
            style_profile=style_profile,
            style_notes=style_notes,
            punctuation_guidance=punctuation_guidance,
            jerga_report=self.last_jerga_report,
        )
        prompt_cache_key = f"translation_session_{time.time()}"

        def _should_cancel() -> bool:
            return bool(cancel_event and cancel_event.is_set())

        resume_entries: List[Tuple[str, str]] = []
        resume_active = False
        resume_index = 0
        cancelled = False

        if _should_cancel():
            self._last_translation_cancelled = True
            return [block.copy() for block in document]

        def _is_translatable(block: Dict[str, Any]) -> bool:
            block_type = block.get("type", "")
            if block_type.startswith("h"):
                return True
            return block_type in TRANSLATABLE_BLOCK_TYPES

        if simplified_log_path and resume and simplified_log_path.exists():
            resume_entries = _parse_translation_log(simplified_log_path)
            resume_active = bool(resume_entries)
            if not resume_active:
                print("Resume solicitado pero no se encontraron traducciones previas.")
        elif simplified_log_path and simplified_log_path.exists():
            simplified_log_path.unlink()

        translatable_indices = [idx for idx, block in enumerate(document) if _is_translatable(block)]
        total_translatable = len(translatable_indices)
        effective_total = min(total_translatable, max_translated_blocks) if max_translated_blocks is not None else total_translatable
        if effective_total == 0:
            self._last_translation_cancelled = cancelled
            return [blk.copy() for blk in document]

        if not aggregate_mode:
            translated_document: List[Dict[str, Any]] = []
            processed_blocks = 0
            for i, block in enumerate(document):
                if _should_cancel():
                    cancelled = True
                    translated_document.extend(b.copy() for b in document[i:])
                    break
                if _is_translatable(block) and (max_translated_blocks is None or processed_blocks < max_translated_blocks):
                    prev_context = document[i - 1]["text"] if i > 0 else ""
                    next_context = document[i + 1]["text"] if i + 1 < len(document) else ""
                    original_text = block.get("text", "")

                    if resume_active and resume_index < len(resume_entries):
                        logged_original, logged_translation = resume_entries[resume_index]
                        if _normalize_for_resume(logged_original) == _normalize_for_resume(original_text) and not _is_incomplete_translation(logged_translation):
                            translated_text = logged_translation
                            resume_index += 1
                        else:
                            resume_active = False

                    if not resume_active or resume_index > len(resume_entries):
                        search_text = " ".join(filter(None, [prev_context, original_text, next_context]))
                        glossary_subset = self._select_glossary_subset(search_text, glossary_entries)
                        translated_text, failure_reason = self._translate_block_with_retry(
                            original_text,
                            glossary_payload={"entries": glossary_subset},
                            prev_context=prev_context,
                            next_context=next_context,
                            static_prompt_prefix=static_prompt_prefix,
                            model=model,
                            prompt_cache_key=prompt_cache_key,
                        )
                        if translated_text is None:
                            self._register_translation_failure(f"Bloque {i + 1}: se agotaron los reintentos ({failure_reason}).")
                            cancelled = True
                            translated_document.extend(b.copy() for b in document[i:])
                            break
                        if simplified_log_path:
                            log_simplified_translation(simplified_log_path, original_text, translated_text)

                    new_block = block.copy()
                    new_block["text"] = translated_text
                    translated_document.append(new_block)
                    processed_blocks += 1
                    if progress_callback:
                        progress_callback(processed_blocks, effective_total)
                else:
                    translated_document.append(block.copy())
            self._last_translation_cancelled = cancelled
            return translated_document

        # Aggregated translation mode
        aggregate_target_words = max(aggregate_target_words, 200)
        limited_indices = translatable_indices[:max_translated_blocks] if max_translated_blocks is not None else translatable_indices
        effective_total = len(limited_indices)
        if effective_total == 0:
            return [blk.copy() for blk in document]

        chunks: List[List[int]] = []
        ptr = 0
        while ptr < len(limited_indices):
            chunk_indices: List[int] = []
            chunk_words = 0
            while ptr < len(limited_indices):
                block_idx = limited_indices[ptr]
                word_count = max(1, len(re.findall(r"\w+", document[block_idx].get("text", ""))))
                if chunk_indices and (chunk_words + word_count) > aggregate_target_words:
                    break
                chunk_indices.append(block_idx)
                chunk_words += word_count
                ptr += 1
            if chunk_indices:
                chunks.append(chunk_indices)

        translations_map: Dict[int, str] = {}
        processed_blocks = 0

        for chunk_indices in chunks:
            if _should_cancel():
                cancelled = True
                break

            # Resume logic for chunks
            if resume_active:
                temp_index = resume_index
                chunk_resume_texts: List[str] = []
                all_match = True
                for block_idx in chunk_indices:
                    if temp_index >= len(resume_entries):
                        all_match = False
                        break
                    logged_original, logged_translation = resume_entries[temp_index]
                    if _normalize_for_resume(logged_original) == _normalize_for_resume(document[block_idx].get("text", "")) and not _is_incomplete_translation(logged_translation):
                        chunk_resume_texts.append(logged_translation)
                        temp_index += 1
                    else:
                        all_match = False
                        break
                if all_match:
                    resume_index = temp_index
                    for i, block_idx in enumerate(chunk_indices):
                        translations_map[block_idx] = chunk_resume_texts[i]
                    processed_blocks += len(chunk_indices)
                    if progress_callback:
                        progress_callback(processed_blocks, effective_total)
                    continue
                else:
                    resume_active = False

            chunk_texts = [document[idx]["text"] for idx in chunk_indices]
            chunk_payload = AGGREGATE_SEPARATOR.join(chunk_texts)
            prev_context = document[chunk_indices[0] - 1]["text"] if chunk_indices[0] > 0 else ""
            next_context = document[chunk_indices[-1] + 1]["text"] if chunk_indices[-1] + 1 < len(document) else ""

            combined_context = " ".join(filter(None, [prev_context, chunk_payload, next_context]))
            glossary_subset = self._select_glossary_subset(combined_context, glossary_entries)

            translated_chunk = self.translation_caller.translate_chunk(
                chunk=chunk_payload,
                glossary_subset={"entries": glossary_subset},
                prev_context=prev_context,
                next_context=next_context,
                static_prompt_prefix=static_prompt_prefix,
                model=model,
                segments_count=len(chunk_indices),
                segment_token=AGGREGATE_TOKEN,
                prompt_cache_key=prompt_cache_key,
            )

            segments = [seg.strip() for seg in re.split(rf"\s*{re.escape(AGGREGATE_TOKEN)}\s*", translated_chunk.strip())]
            if len(segments) != len(chunk_indices) or self._needs_retry(chunk_payload, translated_chunk):
                print(f"La respuesta agregada es inválida. Recurriendo a traducción individual.")
                for block_idx in chunk_indices:
                    if _should_cancel():
                        cancelled = True
                        break

                    prev_ctx = document[block_idx - 1]["text"] if block_idx > 0 else ""
                    next_ctx = document[block_idx + 1]["text"] if block_idx + 1 < len(document) else ""
                    block_context_text = " ".join(filter(None, [prev_ctx, document[block_idx]["text"], next_ctx]))
                    glossary_subset_block = self._select_glossary_subset(block_context_text, glossary_entries)

                    original_text = document[block_idx]["text"]
                    translated_text, failure_reason = self._translate_block_with_retry(
                        original_text,
                        glossary_payload={"entries": glossary_subset_block},
                        prev_context=prev_ctx,
                        next_context=next_ctx,
                        static_prompt_prefix=static_prompt_prefix,
                        model=model,
                        prompt_cache_key=prompt_cache_key,
                    )

                    if translated_text is None:
                        self._register_translation_failure(f"Bloque {block_idx + 1}: se agotaron los reintentos ({failure_reason}).")
                        cancelled = True
                        break

                    translations_map[block_idx] = translated_text
                    if simplified_log_path:
                        log_simplified_translation(simplified_log_path, original_text, translated_text)
                    processed_blocks += 1
                if cancelled: break
                continue

            for i, block_idx in enumerate(chunk_indices):
                translations_map[block_idx] = segments[i]
                if simplified_log_path:
                    log_simplified_translation(simplified_log_path, document[block_idx]["text"], segments[i])
            processed_blocks += len(chunk_indices)
            if progress_callback:
                progress_callback(processed_blocks, effective_total)

        if cancelled and progress_callback:
            progress_callback(processed_blocks, effective_total)

        if cancelled and progress_callback and effective_total:
            progress_callback(processed_blocks, effective_total)

        translated_document: List[Dict[str, Any]] = []
        translated_indices = set(translations_map.keys())
        limited_indices_set = set(limited_indices)

        for idx, block in enumerate(document):
            new_block = block.copy()
            if idx in translations_map:
                new_block["text"] = translations_map[idx]
            elif idx in limited_indices_set:
                # Block targeted for translation but not processed (should not happen)
                new_block["text"] = block.get("text", "")
            translated_document.append(new_block)

        self._last_translation_cancelled = cancelled
        return translated_document

    def was_last_translation_cancelled(self) -> bool:
        """Indicates whether the most recent translation run finished prematurely."""
        return self._last_translation_cancelled

    def last_translation_failure_reason(self) -> Optional[str]:
        """Provides context for the last aborted translation, if any."""
        return self._last_translation_failure_reason

    def generate_context_snippet(
        self,
        *,
        kind: str,
        base_text: str,
        model: str = "gpt-5-mini",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate editorial context (bio/resumen/contraportada) using the configured AI client."""
        if not self.client:
            raise RuntimeError("No hay cliente de IA disponible para generar contenido contextual.")
        snippet = (base_text or "").strip()
        if not snippet:
            raise ValueError("Se requiere texto base para generar contenido contextual.")

        kind_key = kind.lower()
        instructions_map = {
            "author_bio": (
                "Redacta una mini biografía del autor en tercera persona, con un tono profesional y cercano. "
                "Incluye logros relevantes, el género predominante y detalles que interesen a un lector nuevo. "
                "Extensión objetivo: 120-160 palabras. Evita inventar hechos no presentes en la información proporcionada."
            ),
            "book_summary": (
                "Escribe un resumen editorial completo de la obra. Describe género, tono, protagonistas y conflicto central. "
                "Extensión objetivo: 150-200 palabras. No reveles spoilers críticos del final."
            ),
            "back_cover": (
                "Genera un texto de contraportada atractivo en español neutro. Presenta la premisa, el conflicto y deja un gancho final. "
                "Extensión objetivo: 150-220 palabras. Evita revelar giros finales y mantén un tono comercial."
            ),
        }
        if kind_key not in instructions_map:
            raise ValueError(f"Tipo de contexto no soportado: {kind}")

        metadata = metadata or {}
        context_lines: List[str] = []
        for key, value in metadata.items():
            if not value:
                continue
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value if item)
            context_lines.append(f"{key.replace('_', ' ').title()}: {value}")
        context_block = "\n".join(context_lines)

        limited_snippet = snippet[:6000]
        user_sections = [
            "### Datos editoriales:",
            context_block or "(sin datos adicionales)",
            "",
            "### Información disponible:",
            limited_snippet,
            "",
            "### Tarea:",
            instructions_map[kind_key],
        ]
        user_prompt = "\n".join(section for section in user_sections if section is not None)
        system_prompt = (
            "Eres un redactor editorial profesional que escribe en español neutro. "
            "Entrega textos listos para publicación con ortografía y puntuación impecables."
        )

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 700,
        }
        if model != "gpt-5-mini":
            params["temperature"] = 0.7 if kind_key != "back_cover" else 0.8

        purpose = f"context_{kind_key}"
        _announce_api_call(model, purpose)
        try:
            response = self.client.chat.completions.create(**params)
            log_interaction(model, user_prompt, params, response=response)
            if response.usage:
                _record_token_usage(self.token_callback, response.usage, model=model, purpose=purpose)
            content = (response.choices[0].message.content or "").strip()
            return content
        except Exception as exc:
            log_interaction(model, user_prompt, params, error=exc)
            raise



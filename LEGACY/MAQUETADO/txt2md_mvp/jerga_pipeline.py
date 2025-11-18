from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
import re
from typing import Any, Callable, Dict, Iterable, List, Optional


JsonDict = Dict[str, Any]


@dataclass
class JergaCandidate:
    """Representa una frase candidata a jerga detectada en una pasada ligera."""

    text: str
    category: str = "unknown"
    comment: str = ""
    block_index: Optional[int] = None

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in ("", None)}


@dataclass
class JergaValidation:
    """Resultado de la validación de jerga con un modelo de mayor potencia."""

    text: str
    translation: Optional[str] = None
    status: str = "pending"  # "pending" | "translated" | "error"
    notes: str = ""

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in ("", None)}


@dataclass
class JergaReport:
    """Reporte agregado de detección y validación de jerga."""

    detected: List[JergaCandidate] = field(default_factory=list)
    validated: List[JergaValidation] = field(default_factory=list)
    discarded: List[JergaValidation] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "detected": [candidate.to_dict() for candidate in self.detected],
            "validated": [item.to_dict() for item in self.validated],
            "discarded": [item.to_dict() for item in self.discarded],
            "metadata": copy.deepcopy(self.metadata),
        }


class JergaDetector:
    """Escaneo inicial de jerga usando un modelo ligero."""

    def __init__(
        self,
        client: Optional[Any] = None,
        token_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        self.client = client
        self.token_callback = token_callback
        self._patterns: Dict[str, Dict[str, Any]] = {
            r"\bholy\s+cat\b": {"category": "exclamación pulp", "translation": "¡Caray!"},
            r"\bgreat\s+cat\b": {"category": "exclamación pulp", "translation": "¡Cielos!"},
            r"\bwell[, ]+i'?ll\b": {"category": "idioma", "translation": ""},
            r"\bzowie\b": {"category": "onomatopeya", "translation": "¡Rayos!"},
            r"\bpurr\s+purr\b": {"category": "onomatopeya", "translation": ""},
            r"\bdoctah\b": {"category": "anglicismo fonético", "translation": "doctor"},
            r"\bdistrict\b": {"category": "anglicismo", "translation": "distrito"},
        }
        self._compiled = [(re.compile(pattern, re.IGNORECASE), data) for pattern, data in self._patterns.items()]

    def analyse_blocks(
        self,
        blocks: Iterable[Dict[str, Any]],
        *,
        model: str = "gpt-5-nano",
        max_items: int = 128,
    ) -> List[JergaCandidate]:
        """
        Analiza los bloques para detectar jerga mediante heurísticas locales.
        """
        candidates: List[JergaCandidate] = []
        for idx, block in enumerate(blocks):
            if len(candidates) >= max_items:
                break
            text = (block.get("text") or "").strip()
            if not text or len(text) < 6:
                continue
            lowered = text.lower()
            for regex, info in self._compiled:
                match = regex.search(lowered)
                if not match:
                    continue
                phrase = match.group(0)
                candidates.append(
                    JergaCandidate(
                        text=phrase,
                        category=str(info.get("category", "jerga")),
                        comment="Detección heurística basada en patrones conocidos.",
                        block_index=idx,
                    )
                )
                break
        print(f"[JergaDetector] Heurísticas locales detectaron {len(candidates)} candidatos (modelo={model}).")
        return candidates


class JergaValidator:
    """Valida las entradas detectadas y propone traducciones."""

    def __init__(
        self,
        client: Optional[Any] = None,
        token_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        self.client = client
        self.token_callback = token_callback

    def validate(
        self,
        candidates: Iterable[JergaCandidate],
        *,
        model: str = "gpt-5-mini",
    ) -> List[JergaValidation]:
        """
        Valida cada candidato devolviendo traducción o error.
        """
        candidates = list(candidates)
        print(f"[JergaValidator] Validando {len(candidates)} candidatos con modelo={model}.")
        dictionary_map = {
            "holy cat": "¡Caray!",
            "great cat": "¡Cielos!",
            "zowie": "¡Rayos!",
            "doctah": "doctor",
            "district": "distrito",
        }
        results: List[JergaValidation] = []
        for candidate in candidates:
            phrase = candidate.text.lower().strip()
            translation = dictionary_map.get(phrase)
            if translation:
                results.append(
                    JergaValidation(
                        text=candidate.text,
                        translation=translation,
                        status="translated",
                        notes="Traducción sugerida por diccionario interno.",
                    )
                )
            else:
                results.append(
                    JergaValidation(
                        text=candidate.text,
                        translation=None,
                        status="pending",
                        notes="Revisión manual requerida.",
                    )
                )
        return results


class JergaPipelineCoordinator:
    """Coordina la detección y validación de jerga para nutrir el glosario."""

    def __init__(
        self,
        client: Optional[Any] = None,
        token_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        self.detector = JergaDetector(client=client, token_callback=token_callback)
        self.validator = JergaValidator(client=client, token_callback=token_callback)

    def run(
        self,
        document: List[Dict[str, Any]],
        *,
        config: Optional[JsonDict] = None,
    ) -> JergaReport:
        """
        Ejecuta el flujo completo de jerga.
        Config soporta claves:
            - detector_model
            - validator_model
            - max_items
        """
        config = config or {}
        start_ts = time.time()
        detector_model = config.get("detector_model", "gpt-5-nano")
        validator_model = config.get("validator_model", "gpt-5-mini")
        max_items = int(config.get("max_items", 128))

        detected = self.detector.analyse_blocks(document, model=detector_model, max_items=max_items)
        validated = self.validator.validate(detected, model=validator_model) if detected else []
        discarded = [item for item in validated if item.status == "error"]
        kept_validated = [item for item in validated if item.status != "error"]

        metadata = {
            "detector_model": detector_model,
            "validator_model": validator_model,
            "total_candidates": len(detected),
            "validated": len(kept_validated),
            "discarded": len(discarded),
            "elapsed_seconds": round(time.time() - start_ts, 3),
        }
        print(
            "[JergaPipeline] Resumen placeholder:",
            metadata,
        )

        return JergaReport(
            detected=detected,
            validated=kept_validated,
            discarded=discarded,
            metadata=metadata,
        )

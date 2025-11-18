# -*- coding: utf-8 -*-
import re
import os
import yaml
import time
from typing import List, Dict, Any, Optional

# Heurísticas para la detección de índices

# 1. Títulos de índice obvios
TOC_TITLES = [
    "contents",
    "index",
    "table of contents",
    "índice",
    "contenidos",
    "sumario",
    "tabla de materias",
]

# 2. Patrones de líneas de entrada de índice (regex)
# - Capítulos con números romanos: Chapter I, CHAPTER 1., etc.
# - Entradas numeradas: 1., 1 -, etc.
TOC_LINE_PATTERNS = [
    re.compile(r"^\s*(?:chapter|capítulo|part|parte|book|libro)?\s*[IVXLCDM]+[\s.\-].+", re.IGNORECASE),
    re.compile(r"^\s*\d+[\s.\-].+"),
]

# 3. Frases de finalización para excluir y detectar
END_PHRASES = [
    "the end",
    "the end.",
    "fin",
    "fin.",
    "el fin",
    "el fin.",
    "end of the project gutenberg ebook",
]


class IndexFinder:
    """
    Analiza bloques de texto para identificar si son un índice (Tabla de Contenidos).
    Utiliza un sistema de puntuación basado en heurísticas y puede aprender nuevos patrones.
    """

    def __init__(self, interactive_mode: bool = False, templates_dir: str = "txt2md_mvp/templates/indices"):
        self.interactive_mode = interactive_mode
        self.templates_dir = templates_dir
        self.learned_patterns = self._load_learned_patterns()

    def _load_learned_patterns(self) -> List[Dict[str, Any]]:
        """Carga los patrones de índice aprendidos desde la carpeta de plantillas."""
        patterns = []
        if not os.path.isdir(self.templates_dir):
            return patterns

        for filename in os.listdir(self.templates_dir):
            if filename.endswith((".yml", ".yaml")):
                filepath = os.path.join(self.templates_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        pattern_data = yaml.safe_load(f)
                        if "line_pattern" in pattern_data:
                            pattern_data["regex"] = re.compile(pattern_data["line_pattern"], re.IGNORECASE)
                            patterns.append(pattern_data)
                except Exception as e:
                    print(f"Warning: Could not load or parse pattern file {filename}: {e}")
        return patterns

    def _calculate_score(
        self, block_lines: List[str], block_position_ratio: float
    ) -> Dict[str, Any]:
        """
        Calcula una puntuación de confianza para un bloque de texto.
        """
        score = 0
        reasons = []

        text_block = "\n".join(block_lines)
        num_lines = len(block_lines)

        if not block_lines or num_lines == 0:
            return {"score": 0, "reasons": ["Empty block"]}

        # Heurística 1: Título del índice
        first_line = block_lines[0].strip().lower()
        if any(title in first_line for title in TOC_TITLES):
            score += 40
            reasons.append("Contains explicit TOC title")

        # Heurística 2: Patrones de línea (hard-coded y aprendidos)
        matching_lines = 0
        # Comprobar patrones aprendidos primero
        for pattern in self.learned_patterns:
            if any(pattern["regex"].match(line.strip()) for line in block_lines):
                score += pattern.get("confidence_boost", 35)
                reasons.append(f"Matches learned pattern: {pattern['name']}")

        for line in block_lines:
            if any(pattern.match(line.strip()) for pattern in TOC_LINE_PATTERNS):
                matching_lines += 1

        if num_lines > 1:
            line_match_ratio = matching_lines / num_lines
            if line_match_ratio > 0.7:
                score += 30
                reasons.append(f"High line pattern match ratio ({line_match_ratio:.2f})")

        # Heurística 3: Posición en el documento
        if block_position_ratio < 0.1:
            score += 20
            reasons.append("Located in the first 10% of the document")
        elif block_position_ratio > 0.9:
            score += 15 # Menos puntos si está al final
            reasons.append("Located in the last 10% of the document")

        # Heurística 4: Líneas cortas
        short_lines = sum(1 for line in block_lines if len(line.strip()) < 100)
        if (short_lines / num_lines) > 0.8:
            score += 15
            reasons.append("High ratio of short lines")

        # Heurística 5: Párrafos largos (penalización)
        if any(len(line) > 300 for line in block_lines):
            score -= 50
            reasons.append("Contains long paragraphs, unlikely to be a TOC")

        # Penalización por frases de finalización (debe ser una coincidencia exacta)
        text_block_stripped_lower = text_block.strip().lower()
        if text_block_stripped_lower in END_PHRASES:
            score = 0
            reasons.append("Block is an exact match for an end phrase.")

        return {"score": min(100, max(0, score)), "reasons": reasons}

    def _is_potential_toc_title(self, block_lines: List[str]) -> bool:
        """Comprueba si un bloque es probablemente un título de TOC (ej. 'CONTENTS')."""
        if not block_lines or len(block_lines) > 2:
            return False
        # Un título de TOC suele ser corto y una coincidencia exacta de palabra
        first_line = block_lines[0].strip().lower()
        return first_line in TOC_TITLES

    def _is_potential_toc_list(self, block_lines: List[str]) -> bool:
        """Comprueba si un bloque parece una lista de entradas de TOC."""
        if not block_lines or len(block_lines) < 2:
            return False

        matching_lines = 0
        for line in block_lines:
            # Comprueba patrones hard-coded y aprendidos
            if any(pattern.match(line.strip()) for pattern in TOC_LINE_PATTERNS) or \
               any(p["regex"].match(line.strip()) for p in self.learned_patterns):
                matching_lines += 1

        line_match_ratio = matching_lines / len(block_lines)
        # Si más del 70% de las líneas coinciden, es probablemente una lista de TOC
        return line_match_ratio > 0.7

    def process_blocks(
        self, blocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Procesa una lista de bloques, identifica el índice y lo marca.
        Incluye lógica para fusionar un título de TOC con su lista de capítulos.
        """
        total_blocks = len(blocks)
        if total_blocks == 0:
            return blocks

        toc_found = False
        i = 0
        while i < len(blocks):
            block = blocks[i]
            block_lines = block.get("lines", [])

            # Saltar bloques ya procesados por una fusión anterior
            if block.get("type"):
                i += 1
                continue

            # Comprobar frases de finalización
            block_text_stripped_lower = "\n".join(block_lines).strip().lower()
            if block_text_stripped_lower in END_PHRASES:
                block["type"] = "END_PROCESSED"
                block["analysis"] = {"decision": "END_PHRASE_DETECTED"}
                i += 1
                continue

            # Si ya se encontró un TOC, buscar partes subsecuentes
            if toc_found:
                score_info = self._calculate_score(block_lines, i / total_blocks)
                if score_info["score"] > 80:
                     block["type"] = "TOC_PROCESSED"
                     block["analysis"] = {"decision": "TOC_SUBSEQUENT", "score": score_info}
                i += 1
                continue

            # Lógica de fusión con lookahead
            is_title = self._is_potential_toc_title(block_lines)
            next_block_is_list = False
            if i + 1 < len(blocks):
                next_block_lines = blocks[i+1].get("lines", [])
                next_block_is_list = self._is_potential_toc_list(next_block_lines)

            if is_title and next_block_is_list:
                merged_lines_for_analysis = block_lines + next_block_lines
                merged_lines_for_prompt = block_lines + [""] + next_block_lines

                block_position_ratio = i / total_blocks if total_blocks > 0 else 0
                score_info = self._calculate_score(merged_lines_for_analysis, block_position_ratio)

                user_confirmed = False
                if score_info["score"] > 80:
                    user_confirmed = True
                elif self.interactive_mode and 50 <= score_info["score"] <= 80:
                    if self._request_user_feedback(merged_lines_for_prompt):
                        user_confirmed = True
                        self.learn_and_save_pattern(merged_lines_for_analysis)

                if user_confirmed:
                    block["type"] = "TOC_PROCESSED"
                    block["analysis"] = {"decision": "TOC_MERGED_MAIN", "score": score_info}
                    blocks[i+1]["type"] = "TOC_PROCESSED"
                    blocks[i+1]["analysis"] = {"decision": "TOC_MERGED_PART"}
                    toc_found = True
                    i += 2
                    continue

            # Fallback a la lógica original si no hubo fusión
            block_position_ratio = i / total_blocks if total_blocks > 0 else 0
            score_info = self._calculate_score(block_lines, block_position_ratio)

            if score_info["score"] > 80:
                block["type"] = "TOC_PROCESSED"
                block["analysis"] = {"decision": "TOC_MAIN", "score": score_info}
                toc_found = True
            elif self.interactive_mode and 50 <= score_info["score"] <= 80:
                if self._request_user_feedback(block_lines):
                    block["type"] = "TOC_PROCESSED"
                    block["analysis"] = {"decision": "TOC_USER_CONFIRMED", "score": score_info}
                    toc_found = True
                    self.learn_and_save_pattern(block_lines)

            i += 1

        return blocks

    def _abstract_line(self, line: str) -> str:
        """Genera un patrón de regex abstracto para una línea de texto."""
        line = re.sub(r"^\s+", "", line)
        line = re.sub(r"\s+$", "", line)

        # Reemplazar números romanos
        line = re.sub(r"^(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", r"[IVXLCDM]+", line, flags=re.IGNORECASE)
        # Reemplazar números arábigos
        line = re.sub(r"^\d+", r"\\d+", line)
        # Reemplazar palabras en mayúsculas (títulos)
        line = re.sub(r"\b[A-Z][A-Z0-9\s]+\b", r".+", line)
        # Escapar caracteres especiales de regex
        line = re.escape(line)
        # Generalizar espacios
        line = re.sub(r"\\ ", r"\\s+", line)

        return f"^{line}.*$"

    def learn_and_save_pattern(self, block_lines: List[str]):
        """Aprende un nuevo patrón de TOC y lo guarda en un archivo YAML."""
        if not block_lines:
            return

        # Usar la primera línea como base para el patrón
        base_pattern = self._abstract_line(block_lines[0])

        pattern_name = f"custom_toc_{int(time.time())}"
        pattern_data = {
            "name": f"Custom TOC Pattern ({time.strftime('%Y-%m-%d')})",
            "confidence_boost": 35,
            "line_pattern": base_pattern,
        }

        if not os.path.isdir(self.templates_dir):
            os.makedirs(self.templates_dir)

        filepath = os.path.join(self.templates_dir, f"{pattern_name}.yml")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(pattern_data, f, default_flow_style=False, allow_unicode=True)
            print(f"New TOC pattern learned and saved to {filepath}")
            # Recargar patrones para que esté disponible inmediatamente
            self.learned_patterns = self._load_learned_patterns()
        except Exception as e:
            print(f"Error saving new pattern: {e}")

    def _request_user_feedback(self, block_lines: List[str]) -> bool:
        """
        Muestra un pop-up para pedir al usuario que confirme si un bloque es un TOC.
        Se asume que ya existe una instancia de Tkinter en ejecución.
        """
        try:
            # No crear una nueva ventana raíz de Tk, ya que la GUI principal ya existe.
            # Se importa aquí para evitar una dependencia dura si no se usa el modo interactivo.
            from tkinter import messagebox

            text_to_show = "\n".join(block_lines[:10])
            if len(block_lines) > 10:
                text_to_show += "\n..."

            response = messagebox.askyesno(
                "Index Detection",
                f"A possible Table of Contents has been detected. Is this correct?\n\n---\n{text_to_show}\n---"
            )
            return response
        except ImportError:
            # Si tkinter no está disponible, no se puede pedir feedback.
            # Se podría registrar esto o simplemente continuar.
            print("Warning: tkinter not found. Cannot ask for user feedback on ambiguous TOC.")
            return False
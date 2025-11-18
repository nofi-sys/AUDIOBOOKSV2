import re
import datetime
import hashlib
from pathlib import Path
from typing import Dict, Any

# Directorio donde se guardarán los patrones aprendidos.
LEARNED_PATTERNS_DIR = Path(__file__).resolve().parent / "templates" / "learned"

class PatternLearner:
    """
    Gestiona el aprendizaje de nuevos patrones de encabezado a partir del feedback del usuario.
    """

    def __init__(self, templates_dir: Path = LEARNED_PATTERNS_DIR):
        self.templates_dir = templates_dir
        self.templates_dir.mkdir(parents=True, exist_ok=True)

    def _abstract_to_regex(self, text: str) -> str:
        """
        Convierte un texto de ejemplo en una expresión regular generalizada.
        """
        # Patrones para identificar partes estructurales, de más específico a más general.
        patterns = [
            # Keyword, Number, Separator, Title (e.g., "CHAPTER 1: The Story")
            r"^(?P<keyword>(?:Chapter|Cap[ií]tulo|Secci[oó]n|Section|Parte|Part|Libro|Book))\s+(?P<number>[\w\d\.]+)\s*(?P<separator>[:.\-–—])?\s*(?P<title>.*)$",
            # Number, Separator, Title (e.g., "1. The Story")
            r"^(?P<number>[\w\d\.]+)\s*(?P<separator>[.:])?\s*(?P<title>.*)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groupdict()
                regex_parts = ["^"]

                if 'keyword' in groups and groups['keyword']:
                    kw = groups['keyword'].lower()
                    # Generalize keyword
                    if kw.startswith("cap"): regex_parts.append(r"(?:Chapter|Cap[ií]tulo)")
                    elif kw.startswith("secci"): regex_parts.append(r"(?:Section|Secci[oó]n)")
                    elif kw.startswith("part"): regex_parts.append(r"(?:Part|Parte)")
                    elif kw.startswith("libr"): regex_parts.append(r"(?:Book|Libro)")
                    else: regex_parts.append(re.escape(groups['keyword'])) # Fallback
                    regex_parts.append(r"\s+")

                if 'number' in groups and groups['number']:
                    # Generalize number format
                    regex_parts.append(r"[\w\d\.]+")

                if 'separator' in groups and groups.get('separator'):
                    # Generalize separator
                    regex_parts.append(r"\s*[:.\-–—]?\s*")

                if 'title' in groups and groups['title']:
                    # Generalize the title part to match anything
                    regex_parts.append(r".*")

                regex_parts.append("$")
                return "".join(regex_parts)

        # Fallback: if no structural pattern matches, create a literal regex.
        # This is for unique headings that don't follow a common structure.
        return f"^{re.escape(text)}$"

    def learn_from_correction(self, original_text: str, correct_type: str) -> None:
        """
        Aprende un nuevo patrón a partir de la corrección de un usuario.

        Args:
            original_text (str): El texto original que fue clasificado.
            correct_type (str): El tipo correcto ('h1', 'h2', 'p', etc.) confirmado por el usuario.
        """
        if not original_text.strip() or not correct_type.startswith('h'):
            # No aprendemos de párrafos o texto vacío.
            return

        # Generar una expresión regular a partir del texto.
        regex_pattern = self._abstract_to_regex(original_text)

        # Crear un nombre de archivo único basado en el hash del patrón para evitar duplicados.
        pattern_hash = hashlib.md5(regex_pattern.encode()).hexdigest()
        file_path = self.templates_dir / f"learned_{pattern_hash[:8]}.yml"

        if file_path.exists():
            # Podríamos actualizar el contador de uso aquí en el futuro.
            return

        # Construir el contenido del archivo YAML.
        content = {
            "name": f"Learned from: '{original_text[:30]}...'",
            "type": correct_type,
            "confidence_boost": 50,  # Aumento de confianza para este patrón.
            "regex_pattern": regex_pattern,
            "source_text": original_text,
            "usage_count": 1,
            "last_used": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        # Escribir el archivo YAML.
        # Usamos un formato simple de clave: valor para evitar dependencias (PyYAML).
        yaml_lines = []
        for key, value in content.items():
            if isinstance(value, str):
                # Escapar comillas simples en el string para que sea un valor YAML válido.
                yaml_value = value.replace("'", "''")
                yaml_lines.append(f"{key}: '{yaml_value}'")
            else:
                yaml_lines.append(f"{key}: {value}")

        file_path.write_text("\n".join(yaml_lines), encoding="utf-8")

if __name__ == '__main__':
    # Ejemplo de uso para probar el módulo.
    learner = PatternLearner()

    # Simular una corrección del usuario.
    user_text_1 = "CAPÍTULO 1: El Despertar"
    user_correction_1 = "h2"
    learner.learn_from_correction(user_text_1, user_correction_1)

    user_text_2 = "Sección 3.A - Avances"
    user_correction_2 = "h3"
    learner.learn_from_correction(user_text_2, user_correction_2)

    print(f"Patrones guardados en: {LEARNED_PATTERNS_DIR}")
    for f in LEARNED_PATTERNS_DIR.glob("*.yml"):
        print(f"--- {f.name} ---")
        print(f.read_text(encoding="utf-8"))
        print("-" * (len(f.name) + 8))
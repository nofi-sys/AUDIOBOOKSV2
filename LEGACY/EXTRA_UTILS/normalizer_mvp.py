import json
import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from typing import Any, Dict, List, Tuple

from text_utils import read_script


# Ruta del diccionario base (junto a este archivo).
CORE_DICT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "normalization_core.json"
)


# =============== CARGA DE DICCIONARIOS ===============

def load_rules_from_json(path: str) -> List[Dict[str, Any]]:
    """Carga reglas desde un JSON.

    Acepta:
      - {"rules": [ ... ]}
      - [ ... ] directamente
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        rules = data.get("rules", [])
    elif isinstance(data, list):
        rules = data
    else:
        rules = []

    if not isinstance(rules, list):
        raise ValueError(f"Formato inválido en {path}: 'rules' no es una lista")

    return rules


def get_regex_flags(flag_str: str) -> int:
    """Convierte una cadena de flags de regex en el entero de flags de `re`."""
    flags = 0
    if not flag_str:
        return flags
    flag_str = flag_str.lower()
    if "i" in flag_str:
        flags |= re.IGNORECASE
    return flags


# =============== NORMALIZACIÓN DE TEXTO ===============

def find_replacements(text: str, rules: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """Aplica todas las reglas al texto ORIGINAL.

    Devuelve:
      - lista de reemplazos no solapados (ordenados por posición)
      - texto normalizado resultante

    Cada reemplazo:
      {
        "start": int,
        "end": int,
        "original": str,
        "replacement": str,
        "rule_id": str,
        "category": str,
      }
    """
    all_matches: List[Dict[str, Any]] = []

    for rule in rules:
        pattern = rule.get("pattern")
        if not pattern:
            continue

        flags = get_regex_flags(rule.get("flags", ""))
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            # Si una regla está mal armada, la salteamos pero avisamos en consola.
            print(
                f"[WARN] No se pudo compilar patrón {pattern!r} "
                f"(regla {rule.get('id', '?')}): {e}"
            )
            continue

        replacement = rule.get("replacement", "")
        rule_id = rule.get("id", "")
        category = rule.get("category", "")

        for m in regex.finditer(text):
            start, end = m.start(), m.end()
            original = m.group(0)
            all_matches.append(
                {
                    "start": start,
                    "end": end,
                    "original": original,
                    "replacement": replacement,
                    "rule_id": rule_id,
                    "category": category,
                }
            )

    # Resolver solapamientos: nos quedamos con matches disjuntos.
    # Ordenamos por inicio y, ante empates, por longitud (más largo primero).
    all_matches.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

    non_overlapping: List[Dict[str, Any]] = []
    current_end = -1
    for m in all_matches:
        if m["start"] >= current_end:
            non_overlapping.append(m)
            current_end = m["end"]

    # Construir texto normalizado a partir de los reemplazos.
    pieces: List[str] = []
    cursor = 0
    for m in non_overlapping:
        if m["start"] > cursor:
            pieces.append(text[cursor : m["start"]])
        pieces.append(m["replacement"])
        cursor = m["end"]
    pieces.append(text[cursor:])

    new_text = "".join(pieces)
    return non_overlapping, new_text


# =============== DETECCIÓN DE CANDIDATOS SOSPECHOSOS ===============

TOKEN_PATTERN = re.compile(r"\w+|\S")


def tokenize_for_candidates(text: str) -> List[Dict[str, Any]]:
    """Tokenización mínima: palabras y signos separados.

    Devuelve lista de dicts:
      - { "text": str, "start": int, "end": int }
    """
    tokens: List[Dict[str, Any]] = []
    for match in TOKEN_PATTERN.finditer(text):
        tokens.append(
            {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def normalize_letters(s: str) -> str:
    """Normaliza una forma para comparación aproximada basada solo en letras."""
    return "".join(ch.lower() for ch in s if ch.isalpha())


def is_potential_abbreviation(token: str) -> bool:
    """Abreviaturas típicas: terminan en punto y contienen letras."""
    t = token.strip()
    if len(t) <= 1:
        return False
    if t.endswith(".") and any(c.isalpha() for c in t):
        return True
    return False


def is_potential_sigla(token: str) -> bool:
    """Siglas: mayúsculas (con o sin puntos), 2-8 letras."""
    t = token.strip()
    if len(t) < 2 or len(t) > 8:
        return False

    cleaned = t.replace(".", "")
    if not cleaned:
        return False

    if cleaned.isupper() and cleaned.isalpha():
        return True
    return False


def is_potential_symbolic(token: str) -> bool:
    """Tokens con símbolos que suelen requerir normalización."""
    t = token.strip()
    if not t:
        return False

    # Símbolos frecuentes en abreviaturas/términos especiales.
    if any(ch in t for ch in {"/", "#", "°", "º", "ª"}):
        return True

    # Mezcla de letras y dígitos (ej. N1, IB3, etc.).
    has_alpha = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if has_alpha and has_digit:
        return True

    return False


def build_known_exact_set(rules: List[Dict[str, Any]]) -> set[str]:
    """Conjunto de ejemplos EXACTOS conocidos (minúsculas)."""
    known: set[str] = set()
    for rule in rules:
        for ex in rule.get("examples", []):
            norm = ex.strip().lower()
            if norm:
                known.add(norm)
    return known


def build_norm_example_index(rules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Índice de ejemplos normalizados solo por letras.

    normalized_letters -> lista de {rule_id, replacement, example}
    """
    index: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        rule_id = rule.get("id", "")
        replacement = rule.get("replacement", "")
        for ex in rule.get("examples", []):
            norm = normalize_letters(ex)
            if not norm:
                continue
            index.setdefault(norm, []).append(
                {
                    "rule_id": rule_id,
                    "replacement": replacement,
                    "example": ex,
                }
            )
    return index


def detect_candidates_with_suggestions(text: str, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detecta formas sospechosas (abreviaturas/siglas/símbolos).

    Solo se reportan aquellas que:
      - no están como ejemplo exacto en el diccionario; y
      - pueden parecerse a ejemplos conocidos al normalizarlas a letras.

    Devuelve lista de candidatos:
      {
        "form": str,
        "count": int,
        "normalized": str,
        "suggestions": [ {rule_id, replacement, example}, ... ]
      }
    """
    tokens = tokenize_for_candidates(text)
    known_exact = build_known_exact_set(rules)
    norm_index = build_norm_example_index(rules)

    candidates_by_form: Dict[str, Dict[str, Any]] = {}

    for tok in tokens:
        raw = tok["text"]
        t = raw.strip()
        if not t:
            continue

        raw_lower = t.lower()

        # ¿Nos interesa como sospechoso?
        if not (
            is_potential_abbreviation(t)
            or is_potential_sigla(t)
            or is_potential_symbolic(t)
        ):
            continue

        # Si ya es exactamente uno de los ejemplos, no lo tratamos como "desconocido".
        if raw_lower in known_exact:
            continue

        norm_letters = normalize_letters(t)
        suggestions = norm_index.get(norm_letters, [])

        key = t  # conservar forma tal como aparece
        entry = candidates_by_form.get(key)
        if not entry:
            entry = {
                "form": key,
                "count": 0,
                "normalized": norm_letters,
                "suggestions": suggestions,
            }
            candidates_by_form[key] = entry
        entry["count"] += 1

    return list(candidates_by_form.values())


# =============== UI: POPUP DE CANDIDATOS ===============

class UnknownCandidatesDialog(tk.Toplevel):
    """Popup que muestra las formas sospechosas y permite exportarlas."""

    def __init__(self, master, candidates, export_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.title("Candidatos a normalización sin regla completa")
        self.geometry("700x480")
        self.export_callback = export_callback

        label = tk.Label(
            self,
            text=(
                "Se detectaron posibles abreviaturas/siglas/símbolos que podrían "
                "requerir normalización.\n"
                "Todos se exportan POR DEFECTO.\n"
                "Seleccioná en la lista los que NO quieras exportar."
            ),
            justify="left",
        )
        label.pack(padx=10, pady=10)

        self.listbox = tk.Listbox(self, selectmode=tk.EXTENDED)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.candidate_items: List[Dict[str, Any]] = []

        for cand in sorted(candidates, key=lambda c: (-c.get("count", 0), c.get("form", ""))):
            form = cand.get("form", "")
            count = cand.get("count", 0)
            sugg = cand.get("suggestions", [])
            if sugg:
                sug_descs = [
                    f"{s.get('replacement', '')} (regla: {s.get('rule_id', '')})"
                    for s in sugg
                ]
                sug_str = "; ".join(sug_descs)
                display = f"{form:<10} (x{count})  → sugerencias: {sug_str}"
            else:
                display = f"{form:<10} (x{count})  → sin sugerencia en diccionario"

            self.listbox.insert(tk.END, display)
            self.candidate_items.append(cand)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        btn_export = tk.Button(btn_frame, text="Exportar candidatos", command=self.on_export)
        btn_export.pack(side=tk.LEFT, padx=5)

        btn_cancel = tk.Button(btn_frame, text="Cerrar sin exportar", command=self.destroy)
        btn_cancel.pack(side=tk.LEFT, padx=5)

    def on_export(self) -> None:
        """Exporta todos los candidatos excepto los seleccionados en la lista."""
        selected_indices = set(self.listbox.curselection())
        result: List[Dict[str, Any]] = []
        for idx, cand in enumerate(self.candidate_items):
            if idx not in selected_indices:
                result.append(cand)

        self.export_callback(result)
        self.destroy()


# =============== APP PRINCIPAL (Tkinter) ===============

class NormalizerApp:
    """Aplicación Tkinter para normalizar guiones con un diccionario JSON."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Normalizador de texto (MVP)")
        master.geometry("480x260")

        self.rules: List[Dict[str, Any]] = []
        self._load_core_rules()

        label = tk.Label(
            master,
            text=(
                "Normalizador de texto para audiolibros\n"
                "(diccionario expandible en JSON externo)"
            ),
        )
        label.pack(pady=10)

        self.btn_load_dict = tk.Button(
            master,
            text="Cargar diccionario extra (.json)",
            command=self.load_extra_dict,
        )
        self.btn_load_dict.pack(pady=5)

        self.btn_select = tk.Button(
            master,
            text="Seleccionar archivo de guion para normalizar",
            command=self.select_file,
            state=tk.NORMAL if self.rules else tk.DISABLED,
        )
        self.btn_select.pack(pady=5)

        self.btn_quit = tk.Button(master, text="Salir", command=master.quit)
        self.btn_quit.pack(pady=10)

    # --- carga de reglas ----------------------------------------------------
    def _load_core_rules(self) -> None:
        try:
            if not os.path.exists(CORE_DICT_PATH):
                raise FileNotFoundError(
                    f"No se encontró {CORE_DICT_PATH}. Crealo a partir del JSON base."
                )
            core_rules = load_rules_from_json(CORE_DICT_PATH)
            self.rules.extend(core_rules)
        except Exception as exc:
            messagebox.showerror(
                "Error al cargar diccionario base",
                f"No se pudo cargar {CORE_DICT_PATH}:\n{exc}\n\n"
                "Sin diccionario no se puede normalizar el texto.",
            )

    def load_extra_dict(self) -> None:
        path = filedialog.askopenfilename(
            title="Elegir diccionario extra (.json)",
            filetypes=[("JSON files", "*.json"), ("Todos los archivos", "*.*")],
        )
        if not path:
            return

        try:
            extra_rules = load_rules_from_json(path)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo cargar el diccionario extra:\n{exc}")
            return

        before = len(self.rules)
        self.rules.extend(extra_rules)
        after = len(self.rules)

        added = after - before
        messagebox.showinfo(
            "Diccionario cargado",
            f"Se agregaron {added} reglas desde:\n{path}",
        )

        if self.rules and self.btn_select["state"] == tk.DISABLED:
            self.btn_select["state"] = tk.NORMAL

    # --- flujo principal ----------------------------------------------------
    def select_file(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Elegir archivo de guion (TXT/PDF)",
            filetypes=[
                ("Text/PDF files", "*.txt;*.pdf"),
                ("Text files", "*.txt"),
                ("PDF files", "*.pdf"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if not filepath:
            return

        if not self.rules:
            messagebox.showerror(
                "Sin diccionario",
                "No hay reglas cargadas. Cargá al menos un diccionario JSON.",
            )
            return

        try:
            self.process_file(filepath)
        except Exception as exc:
            messagebox.showerror("Error", f"Ocurrió un error al procesar el archivo:\n{exc}")

    def process_file(self, filepath: str) -> None:
        """Normaliza el archivo de guion y guarda los resultados."""
        try:
            text = read_script(filepath)
        except Exception as exc:
            messagebox.showerror("Error al leer guion", f"No se pudo leer el archivo:\n{exc}")
            return

        # 1) aplicar normalización
        matches, normalized_text = find_replacements(text, self.rules)

        base, _ext = os.path.splitext(filepath)
        normalized_path = base + "_normalized.txt"
        map_path = base + "_normalization_map.json"
        unknown_path = base + "_unknown_candidates.json"

        # 2) guardar texto normalizado
        try:
            with open(normalized_path, "w", encoding="utf-8") as f:
                f.write(normalized_text)
        except Exception as exc:
            messagebox.showerror(
                "Error",
                f"No se pudo guardar el archivo normalizado:\n{exc}",
            )
            return

        # 3) guardar mapa de reemplazos
        mapping_data = {
            "source_file": os.path.basename(filepath),
            "normalized_file": os.path.basename(normalized_path),
            "replacements": matches,
        }
        try:
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            messagebox.showerror(
                "Error",
                f"No se pudo guardar el mapa de reemplazos:\n{exc}",
            )
            return

        # 4) detectar candidatos con sugerencias
        candidates = detect_candidates_with_suggestions(text, self.rules)

        info_msg = (
            "Normalización completada.\n\n"
            f"Archivo normalizado:\n  {normalized_path}\n"
            f"Mapa de reemplazos:\n  {map_path}\n"
        )

        if not candidates:
            info_msg += "\nNo se detectaron candidatos sospechosos."
            messagebox.showinfo("Listo", info_msg)
            return

        info_msg += (
            f"\nSe detectaron {len(candidates)} formas sospechosas.\n"
            "Se abrirá una ventana para revisarlas y, si querés, exportarlas."
        )
        messagebox.showinfo("Listo", info_msg)

        def export_callback(selected_candidates: List[Dict[str, Any]]) -> None:
            if not selected_candidates:
                messagebox.showinfo(
                    "Sin exportar",
                    "No se exportó ningún candidato (todos desmarcados).",
                )
                return
            data = {
                "source_file": os.path.basename(filepath),
                "candidates": selected_candidates,
            }
            with open(unknown_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            messagebox.showinfo(
                "Candidatos exportados",
                f"Se exportaron {len(selected_candidates)} candidatos a:\n  {unknown_path}",
            )

        UnknownCandidatesDialog(self.master, candidates, export_callback)


def main() -> None:
    root = tk.Tk()
    app = NormalizerApp(root)

    # Pequeña ayuda para ver rápido resultados en consola si se desea.
    help_text = (
        "Normalizador de texto (MVP)\n"
        "- Carga diccionario base desde normalization_core.json\n"
        "- Permite cargar diccionarios extra\n"
        "- Genera *_normalized.txt, *_normalization_map.json y *_unknown_candidates.json\n"
    )
    print(help_text)

    root.mainloop()


if __name__ == "__main__":
    main()


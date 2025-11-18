"""Standalone GUI to exercise the punctuation normalizer without the main app.

Provides a simple interface to load text, tweak settings, run normalization,
and inspect both the normalized output and the list of atomic changes.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

try:
    from txt2md_mvp.punctuation_normalizer import (
        normalize_punctuation,
        NormalizerSettings,
        NormalizationResult,
        LANGUAGE_ES,
        LANGUAGE_EN_US,
        LANGUAGE_EN_UK,
        GENRE_NARRATIVE,
        GENRE_ESSAY,
        GENRE_TECH,
        NBSP,
    )
except ImportError:  # pragma: no cover
    # Allow running as a standalone script by appending project root.
    import sys
    proj_root = Path(__file__).resolve().parents[2]
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    from txt2md_mvp.punctuation_normalizer import (
        normalize_punctuation,
        NormalizerSettings,
        NormalizationResult,
        LANGUAGE_ES,
        LANGUAGE_EN_US,
        LANGUAGE_EN_UK,
        GENRE_NARRATIVE,
        GENRE_ESSAY,
        GENRE_TECH,
        NBSP,
    )


PREFS_PATH = Path.home() / ".txt2md_punctuation_tester.json"


# Opciones de idioma y género (UI ↔ código)
LANGUAGE_OPTIONS_UI: Tuple[Tuple[str, str], ...] = (
    ("Español (ES)", LANGUAGE_ES),
    ("Inglés — EE.UU. (EN_US)", LANGUAGE_EN_US),
    ("Inglés — Reino Unido (EN_UK)", LANGUAGE_EN_UK),
)

GENRE_OPTIONS_UI: Tuple[Tuple[str, str], ...] = (
    ("Narrativa", GENRE_NARRATIVE),
    ("Ensayo", GENRE_ESSAY),
    ("Técnico / Académico", GENRE_TECH),
)


# Políticas con etiquetas humanas (UI ↔ código interno)
POLICY_OPTIONS_UI: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "quote_preference": (
        ("Automático (según idioma/género)", ""),
        ("Angulares (« »)", "angular"),
        ("Dobles (“ ”)", "dobles"),
        ("Simples (‘ ’)", "simples"),
        ("Dobles primarias (EN)", "double_primary"),
        ("Simples primarias (EN)", "single_primary"),
    ),
    "dialogue_policy": (
        ("Automático (según idioma)", ""),
        ("Raya para diálogos (—)", "raya_dialogo"),
        ("Comillas para diálogos", "quotes_dialogue"),
    ),
    "dash_policy": (
        ("Automático (según idioma)", ""),
        ("Raya parentética con espacios (— inciso —)", "raya_parentetica_espaciada"),
        ("Em dash sin espacios (—inciso—)", "em_dash_unspaced"),
        ("En dash con espacios permitido (– inciso –)", "en_dash_spaced_for_parenthesis_allowed"),
    ),
    "range_dash_policy": (
        ("Automático", ""),
        ("En dash para rangos (1999–2005)", "en_dash_for_ranges"),
    ),
    "ellipsis_policy": (
        ("Automático", ""),
        ("Elipsis tipográfica (…) ", "unicode_ellipsis"),
    ),
    "nbspace_policy": (
        ("Automático", ""),
        ("NBSP tras raya de diálogo", "nbspace_dialogue"),
        ("Espaciado estándar", "standard_spacing"),
    ),
    "decimal_grouping_policy": (
        ("Automático", ""),
        ("Mantener agrupación del origen", "keep_source"),
    ),
}


class PunctuationTesterGUI(tk.Tk):
    """Small Tkinter application to interact with the normalizer."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Punctuation Normalizer Tester")
        self.geometry("1200x780")

        # Preferencias persistentes (por idioma)
        self._prefs: Dict[str, Dict[str, str]] = {}
        self._load_all_prefs()

        self._build_controls()
        self.current_path: Optional[Path] = None
        self.last_result: Optional[NormalizationResult] = None

        # Seleccionar último idioma usado, si existe
        last_lang = self._prefs.get("__last_language__")
        if last_lang:
            label = self._lang_label_from_code(last_lang)
            if label:
                self.language_var.set(label)
        # Cargar preferencias del idioma seleccionado
        self._on_language_changed()

    # ------------------------------------------------------------------ #
    # UI construction                                                    #
    # ------------------------------------------------------------------ #
    def _build_controls(self) -> None:
        control_frame = ttk.Frame(self, padding=10)
        control_frame.pack(fill="x")

        ttk.Label(control_frame, text="Idioma:").grid(row=0, column=0, sticky="w", padx=5)
        self.language_var = tk.StringVar(value=self._lang_label_from_code(LANGUAGE_ES) or "Español (ES)")
        lang_menu = ttk.Combobox(
            control_frame,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_OPTIONS_UI],
            width=12,
            state="readonly",
        )
        lang_menu.grid(row=0, column=1, sticky="w", padx=5)
        lang_menu.bind("<<ComboboxSelected>>", lambda _e=None: self._on_language_changed())

        ttk.Label(control_frame, text="Género:").grid(row=0, column=2, sticky="w", padx=5)
        self.genre_var = tk.StringVar(value=self._genre_label_from_code(GENRE_NARRATIVE) or "Narrativa")
        genre_menu = ttk.Combobox(
            control_frame,
            textvariable=self.genre_var,
            values=[label for label, _ in GENRE_OPTIONS_UI],
            width=18,
            state="readonly",
        )
        genre_menu.grid(row=0, column=3, sticky="w", padx=5)

        self.policy_vars: Dict[str, tk.StringVar] = {}
        policy_row = 1
        policy_col = 0
        for policy_key, options in POLICY_OPTIONS_UI.items():
            ttk.Label(control_frame, text=self._policy_label(policy_key)).grid(
                row=policy_row, column=policy_col, sticky="w", padx=5, pady=2
            )
            labels = [label for label, _ in options]
            var = tk.StringVar(value=labels[0])
            menu = ttk.Combobox(
                control_frame,
                textvariable=var,
                values=labels,
                width=25,
                state="readonly",
            )
            menu.grid(row=policy_row, column=policy_col + 1, sticky="w", padx=5, pady=2)
            self.policy_vars[policy_key] = var
            policy_col += 2
            if policy_col >= 6:
                policy_col = 0
                policy_row += 1

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=policy_row + 1, column=0, columnspan=6, pady=(10, 0), sticky="w")

        ttk.Button(button_frame, text="Abrir archivo…", command=self._load_file).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Guardar salida…", command=self._save_output).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Normalizar", command=self._run_normalization).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Guardar preferencias", command=self._save_current_language_prefs).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Limpiar", command=self._clear_texts).pack(side="left", padx=5)

        splitter = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        splitter.pack(fill="both", expand=True, padx=10, pady=10)

        input_frame = ttk.Labelframe(splitter, text="Texto de entrada")
        self.input_text = tk.Text(input_frame, wrap="word", font=("Consolas", 11))
        self.input_text.pack(fill="both", expand=True)
        splitter.add(input_frame, weight=1)

        output_frame = ttk.Labelframe(splitter, text="Texto normalizado")
        self.output_text = tk.Text(output_frame, wrap="word", font=("Consolas", 11), state="disabled")
        self.output_text.pack(fill="both", expand=True)
        splitter.add(output_frame, weight=1)

        info_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        info_frame.pack(fill="both", expand=True)

        stats_frame = ttk.Labelframe(info_frame, text="Estadísticas")
        stats_frame.pack(side="left", fill="both", expand=True)
        self.stats_text = tk.Text(stats_frame, height=6, width=40, state="disabled", font=("Consolas", 10))
        self.stats_text.pack(fill="both", expand=True)

        changes_frame = ttk.Labelframe(info_frame, text="Cambios atómicos")
        changes_frame.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self.changes_tree = ttk.Treeview(
            changes_frame,
            columns=("rule", "description", "before", "after"),
            show="headings",
            height=8,
        )
        for col, heading, width in (
            ("rule", "Regla", 120),
            ("description", "Descripción", 200),
            ("before", "Antes", 180),
            ("after", "Después", 180),
        ):
            self.changes_tree.heading(col, text=heading)
            self.changes_tree.column(col, width=width, stretch=True)
        self.changes_tree.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(changes_frame, orient="vertical", command=self.changes_tree.yview)
        self.changes_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    # ------------------------------------------------------------------ #
    # Actions                                                            #
    # ------------------------------------------------------------------ #
    def _load_file(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Selecciona un archivo de texto",
            filetypes=[("Text files", "*.txt *.md *.markdown"), ("All files", "*.*")],
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")
        self.current_path = path
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, text)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")
        self._set_stats_text("Sin ejecutar")
        self._populate_changes([])

    def _save_output(self) -> None:
        if self.last_result is None:
            messagebox.showinfo("Sin resultados", "Ejecuta la normalización antes de guardar.")
            return
        path_str = filedialog.asksaveasfilename(
            title="Guardar texto normalizado",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not path_str:
            return
        path = Path(path_str)
        path.write_text(self.last_result.normalized_text, encoding="utf-8")
        messagebox.showinfo("Guardado", f"Archivo guardado en {path}")

    def _clear_texts(self) -> None:
        self.input_text.delete("1.0", tk.END)
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state="disabled")
        self._set_stats_text("Sin datos")
        self._populate_changes([])
        self.last_result = None
        self.current_path = None

    def _run_normalization(self) -> None:
        raw_text = self.input_text.get("1.0", tk.END).rstrip("\n")
        if not raw_text.strip():
            messagebox.showwarning("Texto vacío", "Introduce o abre un texto antes de normalizar.")
            return
        settings = self._collect_settings()
        try:
            result = normalize_punctuation(raw_text, settings)
        except Exception as exc:
            messagebox.showerror("Error en normalización", str(exc))
            return
        self.last_result = result
        self._display_output(result)

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #
    def _collect_settings(self) -> NormalizerSettings:
        language_code = self._lang_code_from_label(self.language_var.get()) or LANGUAGE_ES
        genre_code = self._genre_code_from_label(self.genre_var.get()) or GENRE_NARRATIVE
        settings: NormalizerSettings = {
            "language": language_code,
            "genre": genre_code,
        }
        for key, var in self.policy_vars.items():
            ui_label = var.get()
            code = self._policy_code_from_label(key, ui_label)
            if code is not None:
                settings[key] = code
        return settings

    def _display_output(self, result: NormalizationResult) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        display_text = result.normalized_text.replace(NBSP, " ")
        self.output_text.insert(tk.END, display_text)
        self.output_text.configure(state="disabled")
        self._set_stats_text(self._format_stats(result))
        self._populate_changes(result.changes)

    def _set_stats_text(self, text: str) -> None:
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.configure(state="disabled")

    def _format_stats(self, result: NormalizationResult) -> str:
        lines = [
            f"Cambios totales: {len(result.changes)}",
        ]
        for key, value in sorted(result.stats.items()):
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _populate_changes(self, changes) -> None:
        self.changes_tree.delete(*self.changes_tree.get_children())
        for change in changes:
            self.changes_tree.insert(
                "",
                "end",
                values=(
                    change.rule_id,
                    change.description,
                    change.before.replace("\n", "\\n"),
                    change.after.replace("\n", "\\n"),
                ),
            )

    # ------------------------------------------------------------------ #
    # Preferencias (persistencia)                                        #
    # ------------------------------------------------------------------ #
    def _prefs_path(self) -> Path:
        return PREFS_PATH

    def _load_all_prefs(self) -> None:
        try:
            if self._prefs_path().exists():
                self._prefs = json.loads(self._prefs_path().read_text(encoding="utf-8"))
            else:
                self._prefs = {}
        except Exception:
            self._prefs = {}

    def _save_all_prefs(self) -> None:
        try:
            self._prefs_path().write_text(json.dumps(self._prefs, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            messagebox.showwarning("Advertencia", "No se pudieron guardar las preferencias en disco.")

    def _save_current_language_prefs(self) -> None:
        lang = self._lang_code_from_label(self.language_var.get()) or LANGUAGE_ES
        settings = self._collect_settings()
        # Guardar sólo campos relevantes
        entry = {
            "genre": settings.get("genre", ""),
        }
        for key in POLICY_OPTIONS_UI.keys():
            val = settings.get(key)
            if val is None:
                continue
            entry[key] = val
        self._prefs[lang] = entry
        self._prefs["__last_language__"] = lang
        self._save_all_prefs()
        messagebox.showinfo("Preferencias", "Preferencias guardadas para el idioma seleccionado.")

    def _on_language_changed(self) -> None:
        lang = self._lang_code_from_label(self.language_var.get()) or LANGUAGE_ES
        self._prefs["__last_language__"] = lang
        self._save_all_prefs()
        self._apply_prefs_for_language(lang)

    def _apply_prefs_for_language(self, lang: str) -> None:
        data = self._prefs.get(lang)
        if not data:
            return
        # Género
        genre_code = data.get("genre")
        if genre_code:
            label = self._genre_label_from_code(genre_code)
            if label:
                self.genre_var.set(label)
        # Políticas
        for key in POLICY_OPTIONS_UI.keys():
            code = data.get(key)
            if code is None:
                continue
            label = self._policy_label_from_code(key, code)
            if label and key in self.policy_vars:
                self.policy_vars[key].set(label)

    # ------------------------------------------------------------------ #
    # Mapeos UI ↔ códigos                                                #
    # ------------------------------------------------------------------ #
    def _lang_label_from_code(self, code: str) -> Optional[str]:
        for label, c in LANGUAGE_OPTIONS_UI:
            if c == code:
                return label
        return None

    def _lang_code_from_label(self, label: str) -> Optional[str]:
        for lbl, code in LANGUAGE_OPTIONS_UI:
            if lbl == label:
                return code
        return None

    def _genre_label_from_code(self, code: str) -> Optional[str]:
        for label, c in GENRE_OPTIONS_UI:
            if c == code:
                return label
        return None

    def _genre_code_from_label(self, label: str) -> Optional[str]:
        for lbl, code in GENRE_OPTIONS_UI:
            if lbl == label:
                return code
        return None

    def _policy_label(self, key: str) -> str:
        mapping = {
            "quote_preference": "Sistema de comillas",
            "dialogue_policy": "Sistema para diálogos",
            "dash_policy": "Rayas / incisos",
            "range_dash_policy": "Rangos (fechas/números)",
            "ellipsis_policy": "Elipsis",
            "nbspace_policy": "Espaciado especial (NBSP)",
            "decimal_grouping_policy": "Agrupación de decimales",
        }
        return mapping.get(key, key)

    def _policy_code_from_label(self, key: str, label: str) -> Optional[str]:
        for lbl, code in POLICY_OPTIONS_UI.get(key, ()):  # type: ignore[arg-type]
            if lbl == label:
                return code
        return None

    def _policy_label_from_code(self, key: str, code: str) -> Optional[str]:
        for lbl, cd in POLICY_OPTIONS_UI.get(key, ()):  # type: ignore[arg-type]
            if cd == code:
                return lbl
        return None


def main() -> None:
    app = PunctuationTesterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

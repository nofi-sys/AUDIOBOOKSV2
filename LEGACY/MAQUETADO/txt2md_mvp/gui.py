from __future__ import annotations

import datetime
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path
import copy
import json
import re
import sys
from typing import Optional, List, Dict, Any, Callable, Tuple, Set

if __package__ is None or __package__ == "":
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from txt2md_mvp import api_logger, context_fetcher
from txt2md_mvp.inputs import gather_inputs
from txt2md_mvp.pipeline import process_file, process_text
from txt2md_mvp.md2docx import (
    StylesDialog,
    parse_markdown_blocks,
    block_to_dict,
    convert_markdown_to_docx,
    DEFAULT_STYLESET,
    STYLE_KEYS_ORDER,
)
from txt2md_mvp.render import render
from txt2md_mvp.translation_engine import (
    TranslationEngine,
    STYLE_PROFILES,
    TRANSLATABLE_BLOCK_TYPES,
    register_api_trace_listener,
    unregister_api_trace_listener,
)
from txt2md_mvp.translation_review import TranslationReviewDialog
from txt2md_mvp.pattern_learner import PatternLearner
from txt2md_mvp.costs import calculate_cost, PRICING_DATA
from txt2md_mvp.punctuation_normalizer import normalize_punctuation, NormalizerSettings
from txt2md_mvp.normalization_gui import PunctuationNormalizerDialog
from txt2md_mvp.structure_checker import StructureCheckerDialog

DEFAULT_WIKIPEDIA_LANGS = tuple(context_fetcher.DEFAULT_LANGUAGES)


class SupervisionDialog(tk.Toplevel):
    """Diálogo modal para que el usuario supervise una decisión del pipeline."""
    def __init__(self, parent, block_text: str, suggestion: str, confidence: float, prev_context: Optional[str], next_context: Optional[str]):
        super().__init__(parent)
        self.transient(parent)
        self.title("Supervisión Requerida")
        self.geometry("600x650") # Aumentar tamaño para el contexto
        self.result: Optional[str] = None
        self.suggestion = suggestion

        # Opciones de clasificación
        self.CLASSIFICATION_OPTIONS = [
            ("Encabezado Nivel 1", "h1"), ("Encabezado Nivel 2", "h2"), ("Encabezado Nivel 3", "h3"),
            ("Párrafo Normal", "p"), ("Cita en Bloque", "blockquote"), ("Separador de Escena", "hr"),
        ]

        # --- Layout ---
        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Contexto Anterior
        self._create_context_view(main_frame, "Contexto Anterior", prev_context, height=4)

        # Bloque en Cuestión (con más destaque)
        lbl_title = tk.Label(main_frame, text="¿Qué es este bloque de texto?", font=("Helvetica", 12, "bold"))
        lbl_title.pack(pady=(10, 5))
        block_view = ScrolledText(main_frame, height=6, wrap=tk.WORD, relief=tk.SOLID, borderwidth=1)
        block_view.insert(tk.END, block_text)
        block_view.configure(state="disabled", font=("Helvetica", 10, "bold"))
        block_view.pack(fill=tk.X, expand=False, pady=(0, 10))

        # Contexto Posterior
        self._create_context_view(main_frame, "Contexto Posterior", next_context, height=4)

        # Sugerencia del sistema
        suggestion_text = f"Sugerencia: {self._get_suggestion_display_name()} (Confianza: {confidence:.0%})"
        lbl_suggestion = tk.Label(main_frame, text=suggestion_text, fg="blue", font=("Helvetica", 10, "italic"))
        lbl_suggestion.pack(pady=(10, 10))

        # Opciones para el usuario
        options_frame = tk.Frame(main_frame)
        options_frame.pack(fill=tk.X)
        self.selected_option = tk.StringVar(value=suggestion)
        for text, value in self.CLASSIFICATION_OPTIONS:
            rb = tk.Radiobutton(options_frame, text=text, variable=self.selected_option, value=value)
            rb.pack(anchor=tk.W)

        # Botones de acción
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        self.btn_confirm = tk.Button(btn_frame, text="Confirmar Selección", command=self._on_confirm, default=tk.ACTIVE)
        self.btn_confirm.pack(side=tk.RIGHT, padx=(10, 0))
        btn_ignore = tk.Button(btn_frame, text="Tratar como Párrafo", command=self._on_ignore)
        btn_ignore.pack(side=tk.RIGHT)

        # Atajo de teclado
        self.bind("<Return>", lambda event: self.btn_confirm.invoke())

    def _create_context_view(self, parent, title: str, content: Optional[str], height: int):
        if content:
            lbl = tk.Label(parent, text=title, font=("Helvetica", 10, "italic"))
            lbl.pack(pady=(10, 2), anchor=tk.W)
            context_view = ScrolledText(parent, height=height, wrap=tk.WORD, relief=tk.GROOVE, borderwidth=1, fg="gray40")
            context_view.insert(tk.END, content)
            context_view.configure(state="disabled")
            context_view.pack(fill=tk.X, expand=False)

    def _get_suggestion_display_name(self) -> str:
        """Obtiene el nombre legible de la sugerencia."""
        for name, value in self.CLASSIFICATION_OPTIONS:
            if value == self.suggestion:
                return name
        return self.suggestion

    def _on_confirm(self) -> None:
        self.result = self.selected_option.get()
        self.destroy()

    def _on_ignore(self) -> None:
        self.result = "p"  # Tratar como párrafo
        self.destroy()

    def wait_for_decision(self) -> Optional[str]:
        """Pausa y espera a que el usuario cierre el diálogo."""
        self.wait_window(self)
        return self.result


class ContextInfoDialog(tk.Toplevel):
    """Pop-up para gestionar datos editoriales y acciones opcionales."""

    def __init__(
        self,
        parent: tk.Misc,
        meta: Optional[Dict[str, Any]] = None,
        saved_profiles: Optional[List[str]] = None,
        profile_loader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        profile_saver: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        wikipedia_sources: Optional[Dict[str, Any]] = None,
        ai_generate_callback: Optional[Callable[[str, str, str, Optional[Dict[str, Any]]], str]] = None,
        available_models: Optional[List[str]] = None,
    ) -> None:
        super().__init__(parent)
        self.transient(parent)
        self.title("Contexto editorial")
        self.geometry("720x620")
        self.result: Optional[Dict[str, Any]] = None

        meta = copy.deepcopy(meta) if isinstance(meta, dict) else {}
        self._profiles = sorted(saved_profiles or [], key=str.lower)
        self._profile_loader = profile_loader
        self._profile_saver = profile_saver
        self._ai_generate_callback = ai_generate_callback
        self.available_models = available_models or ["gpt-5-mini", "gpt-5"]
        default_model = self.available_models[0] if self.available_models else "gpt-5-mini"
        self.model_var = tk.StringVar(value=default_model)
        self._wikipedia_sources: Dict[str, Dict[str, Any]] = copy.deepcopy(wikipedia_sources or {})
        self._last_save_success = False

        self.title_var = tk.StringVar(value=str(meta.get("title", "") or "").strip())
        self.subtitle_var = tk.StringVar(value=str(meta.get("subtitle", "") or "").strip())
        self.author_var = tk.StringVar(value=str(meta.get("author", "") or "").strip())
        self.contributors_var = tk.StringVar(
            value=", ".join(meta.get("contributors", [])) if isinstance(meta.get("contributors"), list) else ""
        )
        self.append_bio_var = tk.BooleanVar(value=False)
        self.save_back_cover_var = tk.BooleanVar(value=False)
        self.profile_var = tk.StringVar(value="")
        self.save_name_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")

        container = tk.Frame(self, padx=14, pady=14)
        container.pack(fill=tk.BOTH, expand=True)

        meta_frame = tk.LabelFrame(container, text="Metadatos básicos")
        meta_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(meta_frame, text="Título:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(meta_frame, textvariable=self.title_var, width=48).grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        tk.Label(meta_frame, text="Subtítulo:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(meta_frame, textvariable=self.subtitle_var, width=48).grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        tk.Label(meta_frame, text="Autor/a:").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(meta_frame, textvariable=self.author_var, width=48).grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        tk.Label(meta_frame, text="Colaboradores:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        tk.Entry(meta_frame, textvariable=self.contributors_var, width=48).grid(row=3, column=1, sticky="ew", padx=4, pady=4)

        meta_frame.grid_columnconfigure(1, weight=1)

        profile_frame = tk.LabelFrame(container, text="Perfiles guardados")
        profile_frame.pack(fill=tk.X, pady=(0, 12))

        tk.Label(profile_frame, text="Seleccionar perfil existente:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.profile_combo = ttk.Combobox(
            profile_frame, textvariable=self.profile_var, values=self._profiles, state="readonly"
        )
        self.profile_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(profile_frame, text="Cargar", command=self._on_load_profile).grid(row=0, column=2, padx=4, pady=4)

        tk.Label(profile_frame, text="Guardar como:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(profile_frame, textvariable=self.save_name_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(profile_frame, text="Guardar perfil", command=self._on_save_profile).grid(row=1, column=2, padx=4, pady=4)

        profile_frame.grid_columnconfigure(1, weight=1)

        actions_frame = tk.LabelFrame(container, text="Acciones opcionales")
        actions_frame.pack(fill=tk.BOTH, expand=True)

        buttons_frame = tk.Frame(actions_frame)
        buttons_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(buttons_frame, text="Buscar en Wikipedia", command=self._on_fetch_wikipedia).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons_frame, text="Generar mini bio", command=self._on_generate_author_bio).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons_frame, text="Generar resumen", command=self._on_generate_summary).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons_frame, text="Generar contraportada", command=self._on_generate_back_cover).pack(side=tk.LEFT, padx=4)

        model_frame = tk.Frame(actions_frame)
        model_frame.pack(fill=tk.X, padx=4, pady=(0, 6))
        ttk.Label(model_frame, text="Modelo IA:").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(
            model_frame,
            textvariable=self.model_var,
            values=self.available_models,
            state="readonly",
            width=18,
        )
        self.model_combo.pack(side=tk.LEFT, padx=(6, 0))

        self.status_label = tk.Label(actions_frame, textvariable=self.status_var, fg="gray40", anchor="w")
        self.status_label.pack(fill=tk.X, padx=4, pady=(0, 6))

        notebook = ttk.Notebook(actions_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        bio_frame = tk.Frame(notebook, padx=6, pady=6)
        notebook.add(bio_frame, text="Mini bio")
        self.bio_text = ScrolledText(bio_frame, height=8, wrap=tk.WORD)
        self.bio_text.pack(fill=tk.BOTH, expand=True)
        tk.Checkbutton(bio_frame, text="Anexar bio al final del libro", variable=self.append_bio_var).pack(anchor="w", pady=(6, 0))

        summary_frame = tk.Frame(notebook, padx=6, pady=6)
        notebook.add(summary_frame, text="Resumen / contexto")
        self.summary_text = ScrolledText(summary_frame, height=8, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        back_cover_frame = tk.Frame(notebook, padx=6, pady=6)
        notebook.add(back_cover_frame, text="Contraportada")
        self.back_cover_text = ScrolledText(back_cover_frame, height=8, wrap=tk.WORD)
        self.back_cover_text.pack(fill=tk.BOTH, expand=True)
        tk.Checkbutton(back_cover_frame, text="Exportar contraportada como archivo aparte", variable=self.save_back_cover_var).pack(anchor="w", pady=(6, 0))

        buttons_footer = tk.Frame(container)
        buttons_footer.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons_footer, text="Cancelar", command=self._on_cancel).pack(side=tk.RIGHT, padx=4)
        ttk.Button(buttons_footer, text="Aceptar", command=self._on_accept).pack(side=tk.RIGHT, padx=4)

        self._wikipedia_sources: Dict[str, Dict[str, Any]] = copy.deepcopy(wikipedia_sources or {})
        self.grab_set()
        self.bind("<Escape>", lambda *_: self._on_cancel())

    def _on_load_profile(self) -> None:
        selected = self.profile_var.get().strip()
        if not selected:
            messagebox.showinfo("Perfiles", "No hay perfiles seleccionados.")
            return
        if not self._profile_loader:
            self.status_var.set("No hay cargador de perfiles configurado.")
            return
        data = self._profile_loader(selected)
        if not data:
            self.status_var.set(f"No se pudo cargar el perfil '{selected}'.")
            return
        self._apply_profile_data(data)
        self.save_name_var.set(selected)
        self.status_var.set(f"Perfil '{selected}' cargado correctamente.")

    def _on_fetch_wikipedia(self) -> None:
        term_book = self.title_var.get().strip()
        term_author = self.author_var.get().strip()
        if not term_book and not term_author:
            self.status_var.set("Completa al menos título o autor para buscar en Wikipedia.")
            return
        self.status_var.set("Buscando información en Wikipedia...")
        thread = threading.Thread(
            target=self._fetch_wikipedia_data,
            args=(term_book, term_author),
            daemon=True,
        )
        thread.start()

    def _fetch_wikipedia_data(self, book_term: str, author_term: str) -> None:
        languages = DEFAULT_WIKIPEDIA_LANGS
        try:
            book_data = (
                context_fetcher.fetch_wikipedia_summary(book_term, languages=languages)
                if book_term
                else None
            )
            author_data = (
                context_fetcher.fetch_wikipedia_summary(author_term, languages=languages)
                if author_term
                else None
            )
        except Exception as exc:
            self.after(0, lambda e=exc: self.status_var.set(f"Error consultando Wikipedia: {e}"))
            return
        self.after(0, lambda bd=book_data, ad=author_data: self._apply_wikipedia_data(bd, ad))

    def _apply_wikipedia_data(
        self,
        book_data: Optional[Dict[str, Any]],
        author_data: Optional[Dict[str, Any]],
    ) -> None:
        updates: List[str] = []
        if author_data and author_data.get("extract"):
            self._merge_text(self.bio_text, author_data["extract"])
            self.append_bio_var.set(True)
            self._wikipedia_sources["author"] = author_data
            lang = author_data.get("lang", "")
            updates.append(f"Autor ({lang or '??'})")
        if book_data and book_data.get("extract"):
            self._merge_text(self.summary_text, book_data["extract"])
            self._wikipedia_sources["book"] = book_data
            lang = book_data.get("lang", "")
            updates.append(f"Libro ({lang or '??'})")

        if updates:
            self.status_var.set("Wikipedia actualizado: " + ", ".join(updates))
        else:
            self.status_var.set("No se encontraron resúmenes en Wikipedia.")

    def _merge_text(self, widget: ScrolledText, content: str) -> None:
        text = (content or "").strip()
        if not text:
            return
        current = self._get_text(widget)
        if current.strip():
            if not current.endswith("\n"):
                widget.insert(tk.END, "\n")
            widget.insert(tk.END, "\n" + text)
        else:
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, text)

    def _on_generate_author_bio(self) -> None:
        source_text, metadata = self._compose_source_for_kind("author_bio")
        if not source_text.strip():
            self.status_var.set("No hay datos suficientes para generar la mini bio.")
            return
        self._launch_ai_generation("author_bio", source_text, metadata, self.bio_text)

    def _on_generate_summary(self) -> None:
        source_text, metadata = self._compose_source_for_kind("book_summary")
        if not source_text.strip():
            self.status_var.set("No hay datos suficientes para generar el resumen.")
            return
        self._launch_ai_generation("book_summary", source_text, metadata, self.summary_text)

    def _on_generate_back_cover(self) -> None:
        source_text, metadata = self._compose_source_for_kind("back_cover")
        if not source_text.strip():
            self.status_var.set("No hay datos suficientes para generar la contraportada.")
            return
        self._launch_ai_generation("back_cover", source_text, metadata, self.back_cover_text)

    def _compose_source_for_kind(self, kind: str) -> Tuple[str, Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        segments: List[str] = []
        title = self.title_var.get().strip()
        author = self.author_var.get().strip()
        if kind in {"book_summary", "back_cover"} and title:
            metadata["title"] = title
            segments.append(f"Título de la obra: {title}")
        if author:
            metadata["author"] = author
            if kind == "author_bio":
                segments.append(f"Autor/a: {author}")
        contributors = self._clean_split_list(self.contributors_var.get())
        if contributors:
            metadata["contributors"] = contributors

        if kind == "author_bio":
            wiki_author = self._wikipedia_sources.get("author")
            if wiki_author and wiki_author.get("extract"):
                segments.append(
                    f"Resumen Wikipedia ({wiki_author.get('lang', '??')}):\n{wiki_author.get('extract', '').strip()}"
                )
            current_bio = self._get_text(self.bio_text)
            if current_bio:
                segments.append(f"Bio actual disponible:\n{current_bio.strip()}")
        elif kind == "book_summary":
            wiki_book = self._wikipedia_sources.get("book")
            if wiki_book and wiki_book.get("extract"):
                segments.append(
                    f"Resumen Wikipedia ({wiki_book.get('lang', '??')}):\n{wiki_book.get('extract', '').strip()}"
                )
            existing_summary = self._get_text(self.summary_text)
            if existing_summary:
                segments.append(f"Resumen actual:\n{existing_summary.strip()}")
        elif kind == "back_cover":
            context_summary = self._get_text(self.summary_text)
            if context_summary:
                segments.append(f"Resumen de la obra:\n{context_summary.strip()}")
            wiki_book = self._wikipedia_sources.get("book")
            if wiki_book and wiki_book.get("extract"):
                segments.append(
                    f"Resumen Wikipedia ({wiki_book.get('lang', '??')}):\n{wiki_book.get('extract', '').strip()}"
                )
            notes = self._wikipedia_sources.get("author")
            if notes and notes.get("extract"):
                metadata["author_context"] = notes.get("extract", "")
        segments_text = "\n\n".join(filter(None, segments))
        return segments_text, metadata

    def _launch_ai_generation(
        self,
        kind: str,
        source_text: str,
        metadata: Dict[str, Any],
        target_widget: ScrolledText,
    ) -> None:
        if not self._ai_generate_callback:
            self.status_var.set("No hay generador IA configurado.")
            return
        model = self.model_var.get().strip() or "gpt-5-mini"
        label_map = {
            "author_bio": "mini bio",
            "book_summary": "resumen",
            "back_cover": "contraportada",
        }
        label = label_map.get(kind, kind)
        self.status_var.set(f"Generando {label} con {model}...")

        def worker() -> None:
            try:
                result = self._ai_generate_callback(kind, model, source_text, metadata)
            except Exception as exc:
                self.after(0, lambda e=exc: self.status_var.set(f"Error al generar {label}: {e}"))
                return
            if not result:
                self.after(0, lambda: self.status_var.set(f"No se obtuvo texto para la {label}."))
                return
            self.after(0, lambda txt=result: self._apply_generated_text(target_widget, txt, kind))
            self.after(0, lambda: self.status_var.set(f"{label.capitalize()} generada con {model}."))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_generated_text(self, widget: ScrolledText, text: str, kind: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, cleaned + "\n")
        if kind == "author_bio":
            self.append_bio_var.set(True)
        if kind == "back_cover":
            self.save_back_cover_var.set(True)

    @staticmethod
    def _clean_split_list(raw_value: str) -> List[str]:
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @staticmethod
    def _get_text(widget: ScrolledText) -> str:
        return widget.get("1.0", tk.END).strip()

    def _gather_payload(self) -> Dict[str, Any]:
        payload = {
            "meta": {
                key: value
                for key, value in {
                    "title": self.title_var.get().strip(),
                    "subtitle": self.subtitle_var.get().strip(),
                    "author": self.author_var.get().strip(),
                    "contributors": self._clean_split_list(self.contributors_var.get()),
                }.items()
                if value
            },
            "author_bio": self._get_text(self.bio_text),
            "append_author_bio": bool(self.append_bio_var.get()),
            "summary": self._get_text(self.summary_text),
            "back_cover": self._get_text(self.back_cover_text),
            "export_back_cover": bool(self.save_back_cover_var.get()),
        }
        if self._wikipedia_sources:
            payload["wikipedia_sources"] = copy.deepcopy(self._wikipedia_sources)
        return payload

    def _apply_profile_data(self, data: Dict[str, Any]) -> None:
        meta = data.get("meta") or {}
        self.title_var.set(str(meta.get("title", "")).strip())
        self.subtitle_var.set(str(meta.get("subtitle", "")).strip())
        self.author_var.set(str(meta.get("author", "")).strip())
        contributors = meta.get("contributors") or []
        if isinstance(contributors, list):
            self.contributors_var.set(", ".join(contributors))
        else:
            self.contributors_var.set(str(contributors))

        self.bio_text.delete("1.0", tk.END)
        if data.get("author_bio"):
            self.bio_text.insert(tk.END, data["author_bio"])
        self.append_bio_var.set(bool(data.get("append_author_bio")))

        self.summary_text.delete("1.0", tk.END)
        if data.get("summary"):
            self.summary_text.insert(tk.END, data["summary"])

        self.back_cover_text.delete("1.0", tk.END)
        if data.get("back_cover"):
            self.back_cover_text.insert(tk.END, data["back_cover"])
        self.save_back_cover_var.set(bool(data.get("export_back_cover")))
        self._wikipedia_sources = copy.deepcopy(data.get("wikipedia_sources") or {})

    def _on_save_profile(self) -> None:
        name = self.save_name_var.get().strip()
        if not name:
            self.status_var.set("Ingresa un nombre para guardar el perfil.")
            return
        if not self._profile_saver:
            self.status_var.set("No hay mecanismo para guardar perfiles.")
            return
        payload = self._gather_payload()
        try:
            saved = self._profile_saver(name, payload)
        except Exception as exc:
            self.status_var.set(f"Error al guardar el perfil: {exc}")
            return
        if saved:
            if name not in self._profiles:
                self._profiles.append(name)
                self._profiles.sort(key=str.lower)
                self.profile_combo["values"] = self._profiles
            self.status_var.set(f"Perfil '{name}' guardado.")
            self._last_save_success = True
        else:
            self.status_var.set(f"No se pudo guardar el perfil '{name}'.")
            self._last_save_success = False
        if saved and self._wikipedia_sources:
            self.status_var.set(
                self.status_var.get() + " (incluye referencias de Wikipedia)."
            )

    def _on_accept(self) -> None:
        payload = self._gather_payload()
        self.result = {
            "meta": payload.get("meta", {}),
            "author_bio": payload.get("author_bio", ""),
            "append_author_bio": bool(payload.get("append_author_bio")),
            "summary": payload.get("summary", ""),
            "back_cover": payload.get("back_cover", ""),
            "export_back_cover": bool(payload.get("export_back_cover")),
            "profile_loaded": self.profile_var.get().strip() or None,
            "profile_saved_as": self.save_name_var.get().strip() or None,
            "profile_saved": self._last_save_success,
            "wikipedia_sources": payload.get("wikipedia_sources") or {},
        }
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()

    def show(self) -> Optional[Dict[str, Any]]:
        self.wait_window(self)
        return self.result


class Txt2MdApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("txt2md_mvp GUI")
        self.geometry("860x600")
        self.resizable(True, True)

        self.inputs: list[Path] = []
        self.last_md_path: Optional[str] = None
        self.source_md_path: Optional[str] = None
        self.translated_md_path: Optional[str] = None
        self.last_document: Optional[List[Dict[str, Any]]] = None
        self.translated_document: Optional[List[Dict[str, Any]]] = None
        self.original_translated_document: Optional[List[Dict[str, Any]]] = None
        self.source_meta: Optional[Dict[str, Any]] = None
        self.translated_meta: Optional[Dict[str, Any]] = None
        self.original_translated_meta: Optional[Dict[str, Any]] = None
        self.last_glossary: Dict[str, str] = {}
        self.last_style_profile_key: Optional[str] = None
        self.current_input_path: Optional[Path] = None
        self.current_output_dir: Optional[Path] = None
        self.current_logs_dir: Optional[Path] = None
        self.current_log_file: Optional[Path] = None
        self.current_session_path: Optional[Path] = None
        self.translation_review_results: Optional[Dict[str, Any]] = None
        self.heading_paradigms: Dict[str, str] = {}
        self.markdown_variants: Dict[str, Dict[str, str]] = {}
        self.current_review_options: Dict[str, bool] = {"include_observations": False, "judgement_only": True}
        self._docx_label_to_key: Dict[str, str] = {}
        self._export_controls_override: Optional[bool] = None
        self.work_title_var = tk.StringVar(value="(sin título)")
        self.work_author_var = tk.StringVar(value="(sin autor)")
        self.context_artifacts: Dict[str, Any] = {}
        self.style_cfgs = copy.deepcopy(DEFAULT_STYLESET)
        self._styles_path = Path(__file__).resolve().parent / "style_defaults.json"
        self._load_saved_styles()
        self.translation_engine = TranslationEngine(token_callback=self._update_token_count)
        self._api_trace_listener = None
        def _relay_api_trace(message: str) -> None:
            self.after(0, lambda msg=message: self._log(msg))
        register_api_trace_listener(_relay_api_trace)
        self._api_trace_listener = _relay_api_trace

        self.glob_var = tk.StringVar(value="*.txt")
        self.outdir_var = tk.StringVar(value="out")
        self.recursive_var = tk.BooleanVar(value=False)
        self.clean_gutenberg_var = tk.BooleanVar(value=True)
        self.use_ai_supervision_var = tk.BooleanVar(value=False)
        self.interactive_mode_var = tk.BooleanVar(value=False)
        self.detect_jerga_var = tk.BooleanVar(value=False)
        self.resume_translation_var = tk.BooleanVar(value=False)
        self.test_translation_var = tk.BooleanVar(value=False)
        self.aggregate_blocks_var = tk.BooleanVar(value=False)
        self.aggregate_word_target_var = tk.IntVar(value=2500)
        self.use_punctuation_module_var = tk.BooleanVar(value=False)
        self.ai_model_var = tk.StringVar(value="gpt-5-mini")
        self.style_profile_choices = {info["label"]: key for key, info in STYLE_PROFILES.items()}
        default_style_label = next(iter(self.style_profile_choices)) if self.style_profile_choices else "Literario"
        self.style_profile_var = tk.StringVar(value=default_style_label)
        self.glossary_curation_model_var = tk.StringVar(value="gpt-5-mini")
        self.glossary_translation_model_var = tk.StringVar(value="gpt-5-mini")
        self.qa_model_var = tk.StringVar(value="gpt-5-mini")
        self.skip_qa_var = tk.BooleanVar(value=False)
        self.style_notes_var = tk.StringVar(value="")
        self.total_tokens = 0
        self.token_usage_by_model: Dict[str, Dict[str, int]] = {}
        self.token_usage_by_stage: Dict[str, Dict[str, int]] = {}
        self._token_purposes_seen: Set[str] = set()
        self.token_count_var = tk.StringVar(value="Tokens usados: 0")
        self.token_breakdown_var = tk.StringVar(value="Resumen tokens: (sin datos)")
        self.docx_source_var = tk.StringVar(value="")
        self._context_profiles_dir = Path(__file__).resolve().parent.parent / "context_profiles"

        self._build_layout()
        self._bind_events()
        self._refresh_context_metadata()
        self._reset_token_metrics()
        self._ensure_context_profiles_dir()

    def _build_layout(self) -> None:
        frm_inputs = tk.LabelFrame(self, text="Entradas")
        frm_inputs.pack(fill=tk.BOTH, expand=False, padx=10, pady=10)

        btn_files = tk.Button(frm_inputs, text="Agregar archivos", command=self._add_files)
        btn_files.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        btn_folder = tk.Button(frm_inputs, text="Agregar carpeta", command=self._add_folder)
        btn_folder.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        btn_clear = tk.Button(frm_inputs, text="Limpiar", command=self._clear_inputs)
        btn_clear.grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        btn_load_session = tk.Button(frm_inputs, text="Cargar Sesión", command=self._load_session)
        btn_load_session.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        frm_inputs.grid_columnconfigure(0, weight=1)
        frm_inputs.grid_columnconfigure(1, weight=1)
        frm_inputs.grid_columnconfigure(2, weight=1)
        frm_inputs.grid_columnconfigure(3, weight=1)

        self.listbox = tk.Listbox(frm_inputs, height=6)
        self.listbox.grid(row=1, column=0, columnspan=3, padx=5, pady=(0, 5), sticky="nsew")
        frm_inputs.grid_rowconfigure(1, weight=1)

        frm_context = tk.LabelFrame(self, text="Contexto editorial")
        frm_context.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Label(frm_context, text="Título:").grid(row=0, column=0, padx=5, pady=4, sticky="w")
        tk.Label(frm_context, textvariable=self.work_title_var, anchor="w").grid(row=0, column=1, padx=5, pady=4, sticky="ew")
        tk.Label(frm_context, text="Autor/a:").grid(row=1, column=0, padx=5, pady=4, sticky="w")
        tk.Label(frm_context, textvariable=self.work_author_var, anchor="w").grid(row=1, column=1, padx=5, pady=4, sticky="ew")
        ttk.Button(frm_context, text="Editar contexto...", command=self._open_context_dialog).grid(row=0, column=2, rowspan=2, padx=5, pady=4, sticky="ns")
        frm_context.grid_columnconfigure(1, weight=1)

        frm_opts = tk.LabelFrame(self, text="Opciones")
        frm_opts.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(frm_opts, text="Patrón glob").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        entry_glob = tk.Entry(frm_opts, textvariable=self.glob_var, width=20)
        entry_glob.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        chk_recursive = tk.Checkbutton(frm_opts, text="Recursivo", variable=self.recursive_var)
        chk_recursive.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        chk_clean_gutenberg = tk.Checkbutton(frm_opts, text="Limpiar Gutenberg", variable=self.clean_gutenberg_var)
        chk_clean_gutenberg.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        chk_ai_supervision = tk.Checkbutton(frm_opts, text="Enable AI Supervision", variable=self.use_ai_supervision_var)
        chk_ai_supervision.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        chk_interactive_mode = tk.Checkbutton(frm_opts, text="Modo Interactivo", variable=self.interactive_mode_var)
        chk_interactive_mode.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        tk.Label(frm_opts, text="Modelo IA:").grid(row=0, column=6, padx=(10, 5), pady=5, sticky="w")
        model_options = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        model_menu = tk.OptionMenu(frm_opts, self.ai_model_var, *model_options)
        model_menu.grid(row=0, column=6, padx=5, pady=5, sticky="w")

        tk.Label(frm_opts, text="Guia de estilo:").grid(row=0, column=7, padx=(10, 5), pady=5, sticky="w")
        style_options = list(self.style_profile_choices.keys())
        style_menu = tk.OptionMenu(frm_opts, self.style_profile_var, *style_options)
        style_menu.grid(row=0, column=7, padx=5, pady=5, sticky="w")

        tk.Label(frm_opts, text="Notas de estilo:").grid(row=0, column=8, padx=(10, 5), pady=5, sticky="w")
        entry_style_notes = tk.Entry(frm_opts, textvariable=self.style_notes_var, width=25)
        entry_style_notes.grid(row=0, column=9, padx=5, pady=5, sticky="w")

        frm_ai_models = tk.LabelFrame(self, text="Modelos IA especializados")
        frm_ai_models.pack(fill=tk.X, padx=10, pady=(0, 5))

        tk.Label(frm_ai_models, text="Curacion de glosario:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        cur_model_menu = tk.OptionMenu(frm_ai_models, self.glossary_curation_model_var, *model_options)
        cur_model_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        tk.Label(frm_ai_models, text="Traduccion de glosario:").grid(row=0, column=2, padx=(20, 5), pady=5, sticky="w")
        glos_trans_menu = tk.OptionMenu(frm_ai_models, self.glossary_translation_model_var, *model_options)
        glos_trans_menu.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        tk.Label(frm_ai_models, text="Auditoria QA:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        qa_model_menu = tk.OptionMenu(frm_ai_models, self.qa_model_var, *model_options)
        qa_model_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        chk_skip_qa = tk.Checkbutton(
            frm_ai_models,
            text="Omitir QA automática",
            variable=self.skip_qa_var,
        )
        chk_skip_qa.grid(row=1, column=2, padx=(20, 5), pady=5, sticky="w")


        tk.Label(frm_opts, text="Salida").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        entry_out = tk.Entry(frm_opts, textvariable=self.outdir_var)
        entry_out.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        btn_out = tk.Button(frm_opts, text="Seleccionar", command=self._choose_output_dir)
        btn_out.grid(row=1, column=2, padx=5, pady=5, sticky="ew")
        chk_resume = tk.Checkbutton(frm_opts, text="Reanudar traducción previa", variable=self.resume_translation_var)
        chk_resume.grid(row=1, column=3, padx=5, pady=5, sticky="w")
        chk_test = tk.Checkbutton(frm_opts, text="Modo prueba (3 bloques)", variable=self.test_translation_var)
        chk_test.grid(row=1, column=4, padx=5, pady=5, sticky="w")
        chk_aggregate = tk.Checkbutton(
            frm_opts,
            text="Agrupar párrafos hasta",
            variable=self.aggregate_blocks_var,
        )
        chk_aggregate.grid(row=1, column=5, padx=5, pady=5, sticky="w")
        spin_aggregate = tk.Entry(frm_opts, textvariable=self.aggregate_word_target_var, width=6)
        spin_aggregate.grid(row=1, column=6, padx=(0, 5), pady=5, sticky="w")
        lbl_words = tk.Label(frm_opts, text="palabras")
        lbl_words.grid(row=1, column=7, padx=(0, 5), pady=5, sticky="w")
        chk_detect_jerga = tk.Checkbutton(
            frm_opts,
            text="Analizar jerga (experimental)",
            variable=self.detect_jerga_var,
        )
        chk_detect_jerga.grid(row=1, column=8, padx=5, pady=5, sticky="w")
        chk_punctuation = tk.Checkbutton(
            frm_opts,
            text="Normalizar signos (rayas/comillas)",
            variable=self.use_punctuation_module_var,
        )
        chk_punctuation.grid(row=1, column=9, padx=5, pady=5, sticky="w")
        frm_opts.grid_columnconfigure(1, weight=1)

        frm_actions = tk.Frame(self)
        frm_actions.pack(fill=tk.X, padx=10, pady=10)

        self.btn_process = tk.Button(frm_actions, text="Procesar a Markdown", command=self._start_processing)
        self.btn_process.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))

        self.btn_translate = tk.Button(frm_actions, text="Traducir...", command=self._start_translation, state="disabled")
        self.btn_translate.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5,5))

        self.btn_review = tk.Button(
            frm_actions,
            text="Revisar traduccion...",
            command=self._open_translation_review,
            state="disabled",
        )
        self.btn_review.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.btn_normalize = tk.Button(
            frm_actions,
            text="Normalizar Puntuación...",
            command=self._run_punctuation_normalization,
            state="disabled"
        )
        self.btn_normalize.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.btn_structure_checker = tk.Button(
            frm_actions,
            text="Check Structure...",
            command=self._open_structure_checker,
            state="disabled"
        )
        self.btn_structure_checker.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.btn_export_original_md = tk.Button(
            frm_actions,
            text="Exportar MD original",
            command=lambda: self._export_markdown_variant("translated_original"),
            state="disabled",
        )
        self.btn_export_original_md.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        self.btn_export_corrected_md = tk.Button(
            frm_actions,
            text="Exportar MD corregido",
            command=lambda: self._export_markdown_variant("translated_corrected"),
            state="disabled",
        )
        self.btn_export_corrected_md.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 5))

        docx_frame = tk.Frame(frm_actions)
        docx_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        ttk.Label(docx_frame, text="Fuente DOCX:").pack(side=tk.TOP, anchor="w")
        self.docx_source_combo = ttk.Combobox(docx_frame, textvariable=self.docx_source_var, state="disabled", width=28)
        self.docx_source_combo.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self.docx_source_combo.bind("<<ComboboxSelected>>", self._on_docx_source_changed)

        self.btn_export_docx = tk.Button(
            docx_frame, text="Exportar a DOCX...", command=self._show_docx_export_dialog, state="disabled"
        )
        self.btn_export_docx.pack(side=tk.TOP, fill=tk.X)

        self.frm_progress = tk.Frame(self)
        # No lo empaquetamos todavía, se hará dinámicamente

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.frm_progress, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, expand=True, padx=5, pady=(5, 2))

        self.eta_var = tk.StringVar(value="Calculando tiempo restante...")
        self.eta_label = tk.Label(self.frm_progress, textvariable=self.eta_var, anchor="e")
        self.eta_label.pack(fill=tk.X, expand=True, padx=5, pady=(0, 5))

        self.log = ScrolledText(self, height=10, state="disabled")
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 0))

        status_bar = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))
        self.token_label = tk.Label(status_bar, textvariable=self.token_count_var, anchor=tk.E)
        self.token_label.pack(fill=tk.X, padx=5)
        self.token_breakdown_label = tk.Label(
            status_bar,
            textvariable=self.token_breakdown_var,
            anchor=tk.E,
            justify=tk.RIGHT,
        )
        self.token_breakdown_label.pack(fill=tk.X, padx=5)

    def _refresh_context_metadata(self) -> None:
        meta_source = self.source_meta or {}
        title = str(meta_source.get("title") or meta_source.get("book_title") or "").strip()
        if not title and isinstance(meta_source.get("metadata"), dict):
            title = str(meta_source["metadata"].get("title", "")).strip()
        self.work_title_var.set(title or "(sin título)")

        author = str(meta_source.get("author") or "").strip()
        if not author:
            if isinstance(meta_source.get("authors"), list):
                author = ", ".join(item for item in meta_source["authors"] if isinstance(item, str) and item.strip())
            elif isinstance(meta_source.get("contributors"), list):
                author = ", ".join(item for item in meta_source["contributors"] if isinstance(item, str) and item.strip())
        self.work_author_var.set(author or "(sin autor)")

    def _ensure_context_profiles_dir(self) -> None:
        try:
            self._context_profiles_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._log(f"Advertencia: no se pudo preparar el directorio de perfiles ({exc}).")

    def _context_profile_slug(self, name: str) -> str:
        base = re.sub(r"[^0-9a-zA-Z]+", "-", name.strip().lower()).strip("-")
        return base or "perfil"

    def _load_saved_profiles_catalog(self) -> List[str]:
        profiles: List[str] = []
        if not self._context_profiles_dir.exists():
            return profiles
        for path in self._context_profiles_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = str(data.get("name") or path.stem)
            if name not in profiles:
                profiles.append(name)
        profiles.sort(key=str.lower)
        return profiles

    def _load_context_profile(self, name: str) -> Optional[Dict[str, Any]]:
        slug = self._context_profile_slug(name)
        path = self._context_profiles_dir / f"{slug}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return {
            "meta": data.get("meta") or {},
            "author_bio": data.get("author_bio", ""),
            "append_author_bio": data.get("append_author_bio", False),
            "summary": data.get("summary", ""),
            "back_cover": data.get("back_cover", ""),
            "export_back_cover": data.get("export_back_cover", False),
        }

    def _save_context_profile(self, name: str, payload: Dict[str, Any]) -> bool:
        slug = self._context_profile_slug(name)
        if not slug:
            return False
        self._ensure_context_profiles_dir()
        path = self._context_profiles_dir / f"{slug}.json"
        record = {
            "name": name,
            "meta": payload.get("meta") or {},
            "author_bio": payload.get("author_bio", ""),
            "append_author_bio": bool(payload.get("append_author_bio")),
            "summary": payload.get("summary", ""),
            "back_cover": payload.get("back_cover", ""),
            "export_back_cover": bool(payload.get("export_back_cover")),
            "wikipedia_sources": payload.get("wikipedia_sources") or {},
            "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        try:
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log(f"Perfil de contexto '{name}' guardado en {path}.")
            return True
        except OSError as exc:
            self._log(f"No se pudo guardar el perfil '{name}': {exc}")
            return False

    def _open_context_dialog(self) -> None:
        dialog = ContextInfoDialog(
            parent=self,
            meta=copy.deepcopy(self.source_meta) if isinstance(self.source_meta, dict) else {},
            saved_profiles=self._load_saved_profiles_catalog(),
            profile_loader=self._load_context_profile,
            profile_saver=self._save_context_profile,
            wikipedia_sources=copy.deepcopy(
                self.context_artifacts.get("wikipedia_sources")
                if isinstance(self.context_artifacts.get("wikipedia_sources"), dict)
                else {}
            ),
            ai_generate_callback=self._generate_context_with_ai,
            available_models=["gpt-5", "gpt-5-mini"],
        )
        result = dialog.show()
        if not result:
            return

        merged_meta = copy.deepcopy(self.source_meta) if isinstance(self.source_meta, dict) else {}
        merged_meta.update(result.get("meta") or {})
        self.source_meta = merged_meta
        author_bio = (result.get("author_bio") or "").strip()
        summary = (result.get("summary") or "").strip()
        back_cover = (result.get("back_cover") or "").strip()
        context_update = {
            "author_bio": author_bio,
            "append_author_bio": bool(result.get("append_author_bio")),
            "summary": summary,
            "back_cover": back_cover,
            "export_back_cover": bool(result.get("export_back_cover")),
            "profile_loaded": result.get("profile_loaded"),
        }
        # Limpiar claves vacías para evitar ruido en la sesión
        if not author_bio:
            context_update.pop("author_bio")
        if not summary:
            context_update.pop("summary")
        if not back_cover:
            context_update.pop("back_cover")

        self.context_artifacts.update(context_update)
        if context_update.get("append_author_bio") and author_bio:
            # Marcar como pendiente de inserción en la siguiente traducción
            self.context_artifacts["bio_inserted"] = False
        elif "append_author_bio" in context_update and not context_update["append_author_bio"]:
            self.context_artifacts.pop("bio_inserted", None)
        wiki_sources = result.get("wikipedia_sources") or {}
        if wiki_sources:
            self.context_artifacts["wikipedia_sources"] = wiki_sources
        elif "wikipedia_sources" in self.context_artifacts:
            self.context_artifacts.pop("wikipedia_sources", None)
        self._refresh_context_metadata()
        self._persist_context_meta()
        self._log("Contexto editorial actualizado. Se registraron acciones pendientes para procesamiento posterior.")
        if result.get("profile_saved_as") and result.get("profile_saved"):
            self._log(f"Perfil '{result['profile_saved_as']}' guardado para reutilizacion futura.")

    def _persist_context_meta(self) -> None:
        if not self.current_session_path:
            return
        try:
            if self.current_session_path.exists():
                with open(self.current_session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            else:
                session_data = {}
        except (OSError, json.JSONDecodeError):
            return

        session_data.setdefault("meta", {})
        if isinstance(session_data["meta"], dict):
            session_data["meta"].update(self.source_meta or {})
        else:
            session_data["meta"] = copy.deepcopy(self.source_meta) if isinstance(self.source_meta, dict) else {}
        session_data["context_artifacts"] = copy.deepcopy(self.context_artifacts)

        try:
            with open(self.current_session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except OSError:
            self._log("No se pudo actualizar el archivo de sesión con los metadatos editoriales.")

    def _build_jerga_config(self) -> Dict[str, Any]:
        """Configura los parámetros para el pipeline de jerga."""
        return {
            "detector_model": "gpt-5-nano",
            "validator_model": self.glossary_translation_model_var.get() or "gpt-5-mini",
            "max_items": 128,
        }

    def _generate_context_with_ai(
        self,
        kind: str,
        model: str,
        source_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self.translation_engine:
            raise RuntimeError("Translation engine not available para la generación IA.")
        metadata = metadata or {}
        result = self.translation_engine.generate_context_snippet(
            kind=kind,
            base_text=source_text,
            model=model,
            metadata=metadata,
        )
        if result:
            self.after(
                0,
                lambda: self._log(
                    f"Contexto IA ({kind}) generado con {model}. Longitud: {len(result.split())} palabras."
                ),
            )
        return result

    def _update_token_count(
        self,
        count: int,
        model: str = "unknown",
        type: str = "output",
        purpose: str = "unspecified",
    ) -> None:
        self.total_tokens += count

        # Registrar uso por modelo
        model_usage = self.token_usage_by_model.setdefault(
            model, {"input": 0, "output": 0, "cached_input": 0}
        )
        model_usage[type] = model_usage.get(type, 0) + count

        if purpose and purpose not in {"unspecified", ""}:
            stage_usage = self.token_usage_by_stage.setdefault(
                purpose, {"input": 0, "output": 0, "cached_input": 0}
            )
            stage_usage[type] = stage_usage.get(type, 0) + count
            if purpose not in self._token_purposes_seen:
                self._token_purposes_seen.add(purpose)
                human_label = self._humanize_purpose(purpose)
                self.after(0, lambda msg=f"Monitoreando uso de tokens para {human_label}.": self._log(msg))

        formatted_tokens = f"{self.total_tokens:,}".replace(",", ".")
        self.token_count_var.set(f"Tokens usados: {formatted_tokens}")
        self._update_token_summary_display()

    def _reset_token_metrics(self) -> None:
        self.total_tokens = 0
        self.token_usage_by_model.clear()
        self.token_usage_by_stage.clear()
        self._token_purposes_seen.clear()
        self.token_count_var.set("Tokens usados: 0")
        self.token_breakdown_var.set("Resumen tokens: (sin datos)")

    def _update_token_summary_display(self) -> None:
        if not self.token_usage_by_stage:
            self.token_breakdown_var.set("Resumen tokens: (sin datos)")
            return

        parts: List[str] = []
        for purpose, totals in sorted(self.token_usage_by_stage.items()):
            total = sum(value for value in totals.values())
            if total <= 0:
                continue
            label = self._humanize_purpose(purpose)
            parts.append(f"{label}: {total}")

        if parts:
            self.token_breakdown_var.set("Resumen tokens: " + " · ".join(parts[:5]))
        else:
            self.token_breakdown_var.set("Resumen tokens: (sin datos)")

    @staticmethod
    def _humanize_purpose(purpose: str) -> str:
        lookup = {
            "glossary_curation": "Curación de glosario",
            "glossary_translation": "Traducción de glosario",
            "summary_generation": "Generación de resumen",
            "translation_chunk": "Traducción de bloques",
            "qa_prompt": "Control de calidad (muestreo)",
            "translation_review": "Revisión de traducción",
            "ai_supervision": "Supervisión AI",
        }
        key = (purpose or "").strip().lower()
        if not key:
            return "Sin clasificar"
        if key in lookup:
            return lookup[key]
        return key.replace("_", " ").capitalize()

    def _log_token_summary(self) -> None:
        if not self.token_usage_by_stage and not self.token_usage_by_model:
            self._log("Resumen de tokens: sin llamadas registradas.")
            return

        if self.token_usage_by_stage:
            self._log("Resumen de tokens por etapa:")
            for purpose, totals in sorted(self.token_usage_by_stage.items()):
                total = sum(value for value in totals.values())
                if total <= 0:
                    continue
                breakdown = ", ".join(f"{k}: {v}" for k, v in totals.items() if v)
                self._log(f"  - {self._humanize_purpose(purpose)} → {total} ({breakdown})")

        if self.token_usage_by_model:
            self._log("Resumen de tokens por modelo:")
            for model, totals in sorted(self.token_usage_by_model.items()):
                total = sum(value for value in totals.values())
                if total <= 0:
                    continue
                breakdown = ", ".join(f"{k}: {v}" for k, v in totals.items() if v)
                self._log(f"  - {model} → {total} ({breakdown})")

    def _register_markdown_variant(self, key: str, label: str, path: str) -> None:
        if not path:
            return
        self.markdown_variants[key] = {"label": label, "path": path}
        self._refresh_markdown_selectors()

    def _refresh_markdown_selectors(self) -> None:
        if not hasattr(self, "docx_source_combo"):
            return
        order = ["translated_corrected", "translated_original", "source"]
        keys: List[str] = [k for k in order if k in self.markdown_variants]
        for key in self.markdown_variants:
            if key not in keys:
                keys.append(key)

        labels: List[str] = []
        self._docx_label_to_key = {}
        for key in keys:
            label = self.markdown_variants[key]["label"]
            labels.append(label)
            self._docx_label_to_key[label] = key

        self.docx_source_combo["values"] = labels

        current_label = self.docx_source_var.get()
        if labels:
            if current_label not in labels:
                preferred = next((self.markdown_variants[k]["label"] for k in order if k in self.markdown_variants), labels[0])
                self.docx_source_var.set(preferred)
            self._on_docx_source_changed()
        else:
            self.docx_source_var.set("")
            self.last_md_path = None
        self._update_export_controls()

    def _get_markdown_variant_path(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        info = self.markdown_variants.get(key)
        if not info:
            return None
        return info.get("path")

    def _update_export_controls(self) -> None:
        disable_all = self._export_controls_override is False
        has_original = "translated_original" in self.markdown_variants
        has_corrected = "translated_corrected" in self.markdown_variants
        has_docx_options = bool(self._docx_label_to_key)

        docx_state = tk.DISABLED if disable_all or not has_docx_options else tk.NORMAL
        combo_state = "disabled" if disable_all or not has_docx_options else "readonly"
        original_state = tk.DISABLED if disable_all or not has_original else tk.NORMAL
        corrected_state = tk.DISABLED if disable_all or not has_corrected else tk.NORMAL

        if hasattr(self, "btn_export_docx"):
            self.btn_export_docx.configure(state=docx_state)
        if hasattr(self, "docx_source_combo"):
            self.docx_source_combo.configure(state=combo_state)
        if hasattr(self, "btn_export_original_md"):
            self.btn_export_original_md.configure(state=original_state)
        if hasattr(self, "btn_export_corrected_md"):
            self.btn_export_corrected_md.configure(state=corrected_state)

    def _on_docx_source_changed(self, _event: Optional[tk.Event] = None) -> None:
        label = self.docx_source_var.get()
        key = self._docx_label_to_key.get(label)
        path = self._get_markdown_variant_path(key)
        if path and Path(path).exists():
            self.last_md_path = path
        else:
            self.last_md_path = path

    def _get_document_meta_for_variant(
        self, key: str
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        if key == "source":
            return self.last_document, self.source_meta
        if key == "translated_original":
            meta = self.original_translated_meta or self.translated_meta
            return self.original_translated_document, meta
        if key in {"translated_corrected", "translated_current"}:
            return self.translated_document, self.translated_meta
        return None, None

    def _set_document_for_variant(self, key: str, document: List[Dict[str, Any]]) -> None:
        if key == "source":
            self.last_document = document
        elif key == "translated_original":
            self.original_translated_document = document
        elif key in {"translated_corrected", "translated_current"}:
            self.translated_document = document

    def _export_markdown_variant(self, variant_key: str) -> None:
        label = self.markdown_variants.get(variant_key, {}).get("label", variant_key)
        document, _meta = self._get_document_meta_for_variant(variant_key)
        if not document:
            messagebox.showinfo("Exportacion no disponible", f"No hay datos para {label.lower()}.")
            return
        if not self.current_output_dir or not self.current_input_path:
            messagebox.showinfo("Sin directorio", "Procesa un archivo primero para definir el directorio de salida.")
            return

        try:
            if self.current_session_path and self.current_session_path.exists():
                with open(self.current_session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            else:
                session_data = {}
        except Exception:
            session_data = {}

        style_profile = self.last_style_profile_key or next(iter(self.style_profile_choices.values()), "literario")
        make_current = variant_key != "translated_original"
        session_path_value = self.current_session_path or (self.current_output_dir / f"{self.current_input_path.stem}.session.json")
        try:
            path = self._persist_translated_outputs(
                session_data,
                session_path_value,
                self.current_output_dir,
                self.current_input_path,
                style_profile,
                document_override=copy.deepcopy(document),
                variant_key=variant_key,
                variant_label=label,
                make_current=make_current,
            )
        except Exception as exc:
            messagebox.showerror("Exportacion fallida", f"No se pudo exportar {label}: {exc}")
            return
        if self.current_session_path is None:
            self.current_session_path = session_path_value
        self._log(f"Markdown {label} guardado en: {path}")

    def _handle_review_progress(self, entries: List[Dict[str, Any]], options: Dict[str, bool]) -> None:
        self.current_review_options = copy.deepcopy(options)
        self.translation_review_results = {
            "entries": copy.deepcopy(entries),
            "options": copy.deepcopy(options),
        }
        if not self.current_session_path:
            return
        try:
            if self.current_session_path.exists():
                with open(self.current_session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            else:
                session_data = {}
        except Exception:
            session_data = {}
        session_data["translation_review"] = entries
        session_data["translation_review_options"] = options
        session_data["heading_paradigms"] = self.heading_paradigms
        try:
            with open(self.current_session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"No se pudo guardar el progreso de revision: {exc}")

    def _handle_heading_paradigm_change(self, paradigms: Dict[str, str]) -> None:
        self.heading_paradigms = copy.deepcopy(paradigms)
        if not self.current_session_path:
            return
        try:
            if self.current_session_path.exists():
                with open(self.current_session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            else:
                session_data = {}
        except Exception:
            session_data = {}
        session_data["heading_paradigms"] = self.heading_paradigms
        try:
            with open(self.current_session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self._log(f"No se pudo guardar el paradigma de encabezados: {exc}")

    def _bind_events(self) -> None:
        self.bind("<Delete>", lambda _: self._remove_selection())

    def _load_saved_styles(self) -> None:
        if self._styles_path.exists():
            try:
                data = json.loads(self._styles_path.read_text(encoding="utf-8"))
                merged = copy.deepcopy(DEFAULT_STYLESET)
                for key, value in data.items():
                    if isinstance(value, dict):
                        merged.setdefault(key, {}).update(value)
                self.style_cfgs = merged
            except Exception:
                pass

    def _save_style_cfgs(self) -> None:
        try:
            self._styles_path.write_text(json.dumps(self.style_cfgs, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert(tk.END, message + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _sanitize_for_path(self, value: Optional[str]) -> str:
        if not value:
            return "general"
        cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return cleaned or "general"

    def _configure_logging_for_project(self, *, project_name: Optional[str], category: str) -> Tuple[Path, Optional[Path]]:
        safe_project = self._sanitize_for_path(project_name)
        safe_category = self._sanitize_for_path(category or "general")
        base_logs_dir = Path("logs") / safe_project / safe_category
        session_label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        api_logger.set_log_directory(base_logs_dir, session_name=f"{session_label}_{safe_category}")
        return base_logs_dir, api_logger.get_session_log_file()

    def _add_files(self) -> None:
        chosen = filedialog.askopenfilenames(
            title="Seleccionar archivos .txt o .md",
            filetypes=[("Archivos de texto", "*.txt *.md"), ("Todos", "*.*")]
        )
        for file_path in chosen:
            path = Path(file_path)
            if path.exists():
                self._append_input(path)

    def _add_folder(self) -> None:
        directory = filedialog.askdirectory(title="Seleccionar carpeta")
        if not directory:
            return
        paths = list(gather_inputs([directory], self.glob_var.get() or None, self.recursive_var.get()))
        count = 0
        for path in paths:
            if path.exists():
                self._append_input(path)
                count += 1
        self._log(f"Se agregaron {count} archivos desde {directory}.")

    def _clear_inputs(self) -> None:
        self.inputs.clear()
        self.listbox.delete(0, tk.END)
        self.btn_translate.configure(state="disabled")
        self.btn_normalize.configure(state="disabled")
        self.btn_structure_checker.configure(state="disabled")
        self.last_md_path = None
        self.source_md_path = None
        self.translated_md_path = None
        self.last_document = None
        self.translated_document = None
        self.original_translated_document = None
        self.source_meta = None
        self.translated_meta = None
        self.original_translated_meta = None
        self.last_glossary = {}
        self.last_style_profile_key = None
        self.current_input_path = None
        self.current_output_dir = None
        self.current_session_path = None
        self.translation_review_results = None
        self.heading_paradigms = {}
        self.markdown_variants.clear()
        self.current_review_options = {"include_observations": False, "judgement_only": True}
        self._docx_label_to_key = {}
        self.docx_source_var.set("")
        self._export_controls_override = False
        self.context_artifacts.clear()
        self._refresh_context_metadata()
        self._update_export_controls()
        self._log("Lista de entradas vacía.")

    def _remove_selection(self) -> None:
        selection = list(self.listbox.curselection())
        if not selection:
            return
        for index in reversed(selection):
            removed = self.inputs.pop(index)
            self.listbox.delete(index)
            self._log(f"Se quitó: {removed}")

    def _append_input(self, path: Path) -> None:
        if path in self.inputs:
            return
        if path.suffix.lower() not in [".txt", ".md"]:
            self._log(f"Ignorado (no .txt o .md): {path}")
            return
        self.inputs.append(path)
        self.listbox.insert(tk.END, str(path))

        # Preparar directorio de salida
        if self._get_specific_output_dir(path):
            self._log(f"Directorio de salida preparado para: {path.name}")

    def _choose_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Seleccionar directorio de salida")
        if directory:
            self.outdir_var.set(directory)

    def _start_processing(self) -> None:
        if not self.inputs:
            messagebox.showwarning("Sin entradas", "Agrega al menos un archivo .txt para procesar.")
            return

        # El directorio base. El procesamiento individual usará subdirectorios.
        base_outdir = self.outdir_var.get().strip() or "out"
        clean_gutenberg = self.clean_gutenberg_var.get()
        use_ai_supervision = self.use_ai_supervision_var.get()
        interactive_mode = self.interactive_mode_var.get()
        ai_model = self.ai_model_var.get()
        self.btn_process.configure(state="disabled")
        self.btn_translate.configure(state="disabled")
        self.btn_review.configure(state="disabled")
        self.btn_normalize.configure(state="disabled")
        self.last_md_path = None
        self.source_md_path = None
        self.translated_md_path = None
        self.last_document = None
        self.translated_document = None
        self.original_translated_document = None
        self.source_meta = None
        self.translated_meta = None
        self.original_translated_meta = None
        self.last_glossary = {}
        self.last_style_profile_key = None
        self.current_input_path = None
        self.current_output_dir = None
        self.current_session_path = None
        self.translation_review_results = None
        self.heading_paradigms = {}
        self.markdown_variants.clear()
        self.current_review_options = {"include_observations": False, "judgement_only": True}
        self._docx_label_to_key = {}
        self.docx_source_var.set("")
        self._export_controls_override = None
        self.context_artifacts.clear()
        self._refresh_context_metadata()
        self._update_export_controls()
        use_punctuation_module = bool(self.use_punctuation_module_var.get())
        thread = threading.Thread(
            target=self._process_files,
            args=(
                self.inputs.copy(),
                base_outdir,
                clean_gutenberg,
                use_ai_supervision,
                interactive_mode,
                ai_model,
                use_punctuation_module,
            ),
            daemon=True
        )
        thread.start()

    def _process_files(self, files: list[Path], base_outdir: str, clean_gutenberg: bool, use_ai_supervision: bool, interactive_mode: bool, ai_model: str, use_punctuation_module: bool) -> None:
        for path in files:
            try:
                file_specific_outdir = self._get_specific_output_dir(path)
                if not file_specific_outdir:
                    continue

                if path.suffix.lower() == ".md":
                    # Flujo para archivos Markdown
                    self._log(f"Procesando archivo Markdown: {path.name}")
                    with open(path, "r", encoding="utf-8") as f:
                        md_text = f.read()

                    blocks, _ = parse_markdown_blocks(md_text)
                    meta = self._extract_meta_from_markdown(md_text)

                    # Convertir `Block` objects a la estructura de diccionario esperada
                    document = [block_to_dict(block) for block in blocks]

                    # Crear un archivo de sesión también para los .md
                    session_path = file_specific_outdir / f"{path.stem}.session.json"
                    session_data = {
                        'text': md_text,
                        'in_path': str(path.resolve()),
                        'document': document,
                        'meta': meta,
                        'md_path': str(path.resolve()),
                    }
                    with open(session_path, 'w', encoding='utf-8') as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)

                    self.after(0, lambda p=path, md=str(path), doc=document, meta=meta: self._log_success(p, md, doc, meta))

                elif path.suffix.lower() == ".txt":
                    # Flujo existente para archivos de texto
                    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                        text = f.read()

                    result = process_text(
                        text,
                        clean_gutenberg=clean_gutenberg,
                        use_ai_supervision=use_ai_supervision,
                        interactive_mode=interactive_mode,
                        use_punctuation_module=use_punctuation_module,
                        ai_model=ai_model,
                        token_callback=self._update_token_count,
                        supervision_callback=self._request_supervision if interactive_mode else None
                    )

                    md_path = file_specific_outdir / f"{path.stem}.md"
                    session_path = file_specific_outdir / f"{path.stem}.session.json"
                    meta = result.get("meta", {}) if isinstance(result, dict) else {}

                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(result["md"])

                    session_data = {
                        'text': text,
                        'in_path': str(path.resolve()),
                        'document': result['document'],
                        'meta': meta,
                        'md_path': str(md_path.resolve()),
                    }
                    with open(session_path, 'w', encoding='utf-8') as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)

                    self.after(0, lambda p=path, md=str(md_path), doc=result["document"], meta=meta: self._log_success(p, md, doc, meta))

            except Exception as exc:
                self.after(0, lambda p=path, e=exc: self._log_error(p, e))

        self.after(0, self._processing_done)

    def _log_success(self, src: Path, md: str, document: List[Dict[str, Any]], meta: Optional[Dict[str, Any]]) -> None:
        self.last_md_path = md
        self.source_md_path = md
        self.translated_md_path = None
        self.last_document = document
        self.translated_document = None
        self.original_translated_document = None
        self.source_meta = copy.deepcopy(meta) if meta else {}
        self.translated_meta = None
        self.original_translated_meta = None
        self.context_artifacts.clear()
        self._refresh_context_metadata()
        self._register_markdown_variant("source", "Maquetado original", md)
        self._log(f"Listo ({src}):\n  MD      : {md}")

    @staticmethod
    def _extract_meta_from_markdown(md_text: str) -> Dict[str, Any]:
        lines = md_text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}
        meta: Dict[str, Any] = {}
        for raw_line in lines[1:]:
            stripped = raw_line.strip()
            if stripped == "---":
                break
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value_str = value.strip()
            if not key:
                continue
            try:
                meta[key] = json.loads(value_str)
            except json.JSONDecodeError:
                meta[key] = value_str
        return meta

    @staticmethod
    def _has_table_of_contents(md_text: str) -> bool:
        for line in md_text.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("## contenidos") or stripped.startswith("## contents"):
                return True
        return False

    def _build_translated_meta(
        self,
        base_meta: Optional[Dict[str, Any]],
        *,
        source_path: Optional[str],
        style_profile: str,
    ) -> Dict[str, Any]:
        meta = copy.deepcopy(base_meta) if base_meta else {}
        timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        meta["generated_at"] = timestamp
        meta["engine"] = "txt2md_mvp.translation"
        meta["language"] = "es"
        translation_info = meta.get("translation", {}) if isinstance(meta.get("translation"), dict) else {}
        translation_info.update(
            {
                "source_path": source_path,
                "source_language": translation_info.get("source_language") or (base_meta or {}).get("language", "en"),
                "target_language": "es",
                "style_profile": style_profile,
                "status": "draft",
                "generated_at": timestamp,
            }
        )
        meta["translation"] = translation_info
        return meta

    def _prepare_translated_markdown(
        self,
        translated_document: List[Dict[str, Any]],
        base_meta: Optional[Dict[str, Any]],
        *,
        source_path: Optional[str],
        style_profile: str,
        add_toc: bool,
    ) -> Tuple[Dict[str, Any], str]:
        translated_meta = self._build_translated_meta(base_meta, source_path=source_path, style_profile=style_profile)
        md_text = render(translated_document, translated_meta, add_toc=add_toc)
        return translated_meta, md_text

    def _persist_translated_outputs(
        self,
        session_data: Dict[str, Any],
        session_path: Path,
        output_dir: Path,
        input_path: Path,
        style_profile: str,
        *,
        document_override: Optional[List[Dict[str, Any]]] = None,
        variant_key: str = "translated_current",
        variant_label: Optional[str] = None,
        make_current: bool = True,
    ) -> Path:
        original_md_path = session_data.get("md_path") or self.source_md_path
        original_md_text = ""
        add_toc = False
        if original_md_path:
            try:
                original_md_text = Path(original_md_path).read_text(encoding="utf-8")
                add_toc = self._has_table_of_contents(original_md_text)
            except Exception as exc:
                self.after(0, lambda e=exc: self._log(f"No se pudo leer el Markdown original para detectar TOC: {e}"))

        document = document_override if document_override is not None else (self.translated_document or [])
        if not document:
            raise ValueError("No hay documento traducido para persistir.")

        base_meta_for_translation = (
            copy.deepcopy(session_data.get("meta"))
            if isinstance(session_data.get("meta"), dict)
            else (copy.deepcopy(self.source_meta) if self.source_meta else {})
        )
        suffix_map = {
            "translated_current": "_translated.md",
            "translated_original": "_translated_original.md",
            "translated_corrected": "_translated_corrected.md",
        }
        suffix = suffix_map.get(variant_key, f"_{variant_key}.md")
        translated_md_path = output_dir / f"{input_path.stem}{suffix}"
        try:
            translated_meta, translated_md_text = self._prepare_translated_markdown(
                document,
                base_meta_for_translation,
                source_path=session_data.get("in_path"),
                style_profile=style_profile,
                add_toc=add_toc,
            )
            translated_md_path.write_text(translated_md_text, encoding="utf-8")
            document_clone = copy.deepcopy(document)
            if make_current:
                self.translated_meta = translated_meta
                self.translated_md_path = str(translated_md_path)
                self.last_md_path = str(translated_md_path)
                session_data["translated_document"] = document_clone
                session_data["translated_meta"] = translated_meta
                session_data["translated_md_path"] = str(translated_md_path)
            if variant_key == "translated_original":
                self.original_translated_document = document_clone
                self.original_translated_meta = copy.deepcopy(translated_meta)
                session_data["translated_document_original"] = document_clone
                session_data["translated_meta_original"] = translated_meta
                session_data["translated_md_path_original"] = str(translated_md_path)
            elif variant_key == "translated_corrected":
                session_data["translated_document_corrected"] = document_clone
                session_data["translated_meta_corrected"] = translated_meta
                session_data["translated_md_path_corrected"] = str(translated_md_path)
            if variant_label:
                self._register_markdown_variant(variant_key, variant_label, str(translated_md_path))
                self.after(
                    0,
                    lambda path=translated_md_path, lbl=variant_label: self._log(f"{lbl} guardado en: {path}"),
                )
            else:
                self.after(0, lambda path=translated_md_path: self._log(f"Markdown traducido guardado en: {path}"))
        except Exception as exc:
            self.after(0, lambda e=exc: self._log(f"No se pudo generar el Markdown traducido: {e}"))
            raise

        session_data["heading_paradigms"] = self.heading_paradigms
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.after(0, lambda e=exc: self._log(f"No se pudo actualizar la sesion despues de la traduccion: {e}"))
        return translated_md_path

    def _build_review_entries(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        if not self.last_document or not self.translated_document:
            return entries

        saved_by_index: Dict[int, Dict[str, Any]] = {}
        if self.translation_review_results:
            for saved_entry in self.translation_review_results.get("entries", []):
                idx = saved_entry.get("block_index")
                if isinstance(idx, int):
                    saved_by_index[idx] = copy.deepcopy(saved_entry)

        total_blocks = min(len(self.last_document), len(self.translated_document))
        for idx in range(total_blocks):
            source_block = self.last_document[idx]
            translated_block = self.translated_document[idx]
            block_type = translated_block.get("type", "")
            if block_type.startswith("h") or block_type in TRANSLATABLE_BLOCK_TYPES:
                entry = {
                    "block_index": idx,
                    "type": block_type,
                    "original_text": source_block.get("text", ""),
                    "translated_text": translated_block.get("text", ""),
                    "status": "pendiente",
                    "observaciones": "",
                    "revised_text": "",
                }
                saved = saved_by_index.get(idx)
                if saved:
                    entry["status"] = saved.get("status", entry["status"]) or "pendiente"
                    entry["observaciones"] = saved.get("observaciones", entry["observaciones"])
                    entry["revised_text"] = saved.get("revised_text", entry["revised_text"])
                    saved_translated = saved.get("translated_text")
                    if isinstance(saved_translated, str) and saved_translated:
                        entry["translated_text"] = saved_translated
                entries.append(entry)
        return entries

    def _log_error(self, src: Path, error: Exception) -> None:
        self._log(f"Error procesando {src}: {error}")

    def _processing_done(self) -> None:
        self.btn_process.configure(state="normal")
        if self.last_md_path:
            self.btn_translate.configure(state="normal")
            self.btn_review.configure(state="disabled")
            self.btn_normalize.configure(state="disabled")
            self.btn_structure_checker.configure(state="normal")
            self._log("\nProcesamiento de Markdown completado. Ahora puedes traducir o exportar a DOCX.")
        self._update_export_controls()
        messagebox.showinfo("Procesamiento completado", "Se procesaron todos los archivos seleccionados.")

    def _run_punctuation_normalization(self) -> None:
        if not self.translated_document:
            messagebox.showinfo("Sin Traducción", "Primero genera una traducción para poder normalizarla.")
            return

        original_blocks = [
            block.get("text", "")
            for block in self.translated_document
            if block.get("type") in TRANSLATABLE_BLOCK_TYPES
        ]

        if not any(block.strip() for block in original_blocks):
            messagebox.showinfo("Sin texto", "El documento traducido no contiene texto normalizable.")
            return

        initial_text = "\n\n".join(original_blocks)
        settings: NormalizerSettings = {"language": "ES", "genre": "narrativa"}
        dialog = PunctuationNormalizerDialog(self, initial_text, settings)
        normalized_text = dialog.show()

        if normalized_text is None:
            self._log("Normalización cancelada por el usuario.")
            return

        if normalized_text == initial_text:
            self._log("La normalización no produjo cambios.")
            return

        normalized_blocks = normalized_text.split("\n\n")
        if len(original_blocks) != len(normalized_blocks):
            messagebox.showerror(
                "Error de Normalización",
                "El número de bloques de texto cambió durante la normalización, "
                "lo cual podría corromper el documento. No se aplicarán los cambios."
                f"\n\nBloques originales: {len(original_blocks)}\nBloques normalizados: {len(normalized_blocks)}"
            )
            self._log("Error crítico: La normalización alteró la estructura de bloques del documento.")
            return

        changes_applied_count = 0
        normalized_block_idx = 0
        for doc_block in self.translated_document:
            if doc_block.get("type") in TRANSLATABLE_BLOCK_TYPES:
                if doc_block.get("text") != normalized_blocks[normalized_block_idx]:
                    doc_block["text"] = normalized_blocks[normalized_block_idx]
                    changes_applied_count += 1
                normalized_block_idx += 1

        self._log(f"Normalización completada. Se aplicaron {changes_applied_count} cambios.")
        self._log("Puedes exportar el MD corregido para guardar los cambios.")

        if (
            self.current_session_path
            and self.current_output_dir
            and self.current_input_path
            and (self.last_style_profile_key or self.style_profile_choices)
        ):
            session_data: Dict[str, Any] = {}
            if self.current_session_path.exists():
                try:
                    with open(self.current_session_path, "r", encoding="utf-8") as f:
                        session_data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    session_data = {}

            style_profile = self.last_style_profile_key or next(iter(self.style_profile_choices.values()))
            self._persist_translated_outputs(
                session_data,
                self.current_session_path,
                self.current_output_dir,
                self.current_input_path,
                style_profile,
                variant_key="translated_corrected",
                variant_label="Traduccion corregida",
                make_current=True,
            )

    def _request_supervision(self, block_text: str, suggestion: str, confidence: float, prev_context: Optional[str], next_context: Optional[str]) -> Optional[str]:
        """Solicita la supervisión del usuario desde un hilo secundario de forma segura."""
        # Evento para sincronizar el hilo de trabajo con el hilo de la GUI.
        decision_made = threading.Event()
        user_choice = None

        def _create_dialog():
            nonlocal user_choice
            dlg = SupervisionDialog(self, block_text, suggestion, confidence, prev_context, next_context)
            user_choice = dlg.wait_for_decision()
            decision_made.set() # Avisa al hilo de trabajo que la decisión ha sido tomada.

        # Programar la creación del diálogo en el hilo principal de la GUI.
        self.after(0, _create_dialog)

        # El hilo de trabajo espera aquí hasta que `decision_made.set()` es llamado.
        decision_made.wait()

        return user_choice

    def _request_glossary_reuse(self, glossary_path: Path) -> bool:
        """Solicita al usuario la reutilización del glosario desde un hilo secundario."""
        decision_made = threading.Event()
        user_choice = None

        def _create_dialog():
            nonlocal user_choice
            try:
                user_choice = messagebox.askyesno(
                    "Reutilizar Glosario",
                    f"Se encontró un glosario existente:\n\n{glossary_path}\n\n"
                    "¿Deseas reutilizarlo y omitir la costosa fase de curación?",
                    icon=messagebox.QUESTION
                )
            finally:
                decision_made.set()

        self.after(0, _create_dialog)
        decision_made.wait()
        return bool(user_choice)

    def _request_glossary_reuse(self, glossary_path: Path) -> bool:
        """Solicita al usuario la reutilización del glosario desde un hilo secundario."""
        decision_made = threading.Event()
        user_choice = None

        def _create_dialog():
            nonlocal user_choice
            try:
                user_choice = messagebox.askyesno(
                    "Reutilizar Glosario",
                    f"Se encontró un glosario existente:\n\n{glossary_path}\n\n"
                    "¿Deseas reutilizarlo y omitir la costosa fase de curación?",
                    icon=messagebox.QUESTION
                )
            finally:
                decision_made.set()

        self.after(0, _create_dialog)
        decision_made.wait()
        return bool(user_choice)

    def _handle_processing_error(self, message: str) -> None:
        self.btn_process.configure(state="normal")
        messagebox.showerror("Error", message)
        self._log(message)

    def _load_session(self) -> None:
        session_path_str = filedialog.askopenfilename(
            title="Seleccionar archivo de sesión",
            filetypes=[("Session files", "*.session.json"), ("All files", "*.*")]
        )
        if not session_path_str:
            return

        try:
            session_path = Path(session_path_str)
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            if "document" not in session_data or "in_path" not in session_data:
                messagebox.showerror("Error de Sesión", "El archivo de sesión es inválido o está incompleto.")
                return

            in_path = Path(session_data["in_path"])
            if not in_path.exists():
                messagebox.showwarning(
                    "Archivo no encontrado",
                    "El archivo original indicado en la sesión no existe en esta máquina."
                    "\n\nSelecciona la nueva ubicación para continuar.",
                )
                new_path = filedialog.askopenfilename(
                    title="Relocalizar archivo original",
                    initialdir=session_path.parent,
                    filetypes=[("Archivos de texto", "*.txt"), ("Markdown", "*.md"), ("Todos", "*.*")],
                )
                if not new_path:
                    messagebox.showerror("Sesión incompleta", "No se seleccionó un archivo para la sesión.")
                    return
                in_path = Path(new_path)
                session_data["in_path"] = str(in_path.resolve())
                try:
                    with open(session_path, "w", encoding="utf-8") as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)
                    self._log("Sesión actualizada con la nueva ubicación del archivo de origen.")
                except Exception as exc:
                    self._log(f"No se pudo actualizar la sesión con la nueva ruta: {exc}")

            self._clear_inputs()
            self._append_input(in_path)

            self.current_session_path = session_path
            self.current_input_path = in_path
            self.current_output_dir = session_path.parent
            logs_dir, log_file = self._configure_logging_for_project(project_name=in_path.stem, category="translation")
            self.current_logs_dir = logs_dir
            self.current_log_file = log_file
            target = log_file or logs_dir
            self._log(f"Interacciones de IA se registrarán en: {target}")

            self.last_document = copy.deepcopy(session_data["document"])  # type: ignore[arg-type]
            self.source_meta = copy.deepcopy(session_data.get("meta", {})) if isinstance(session_data.get("meta"), dict) else {}
            self.last_glossary = session_data.get("glossary", {}) if isinstance(session_data.get("glossary"), dict) else {}
            self.heading_paradigms = (
                copy.deepcopy(session_data.get("heading_paradigms"))
                if isinstance(session_data.get("heading_paradigms"), dict)
                else {}
            )
            use_punctuation_module = bool(session_data.get("use_punctuation_module", False))
            self.use_punctuation_module_var.set(use_punctuation_module)
            self._log(
                f"Normalizador de signos: {'activado' if use_punctuation_module else 'desactivado'} (desde la sesión)."
            )
            detect_jerga_enabled = bool(session_data.get("detect_jerga", False))
            self.detect_jerga_var.set(detect_jerga_enabled)
            if detect_jerga_enabled:
                self._log("Análisis de jerga activado desde la sesión previa.")
            saved_review_entries = session_data.get("translation_review")
            saved_review_options = session_data.get("translation_review_options")
            if isinstance(saved_review_entries, list):
                options = {"include_observations": False, "judgement_only": True}
                if isinstance(saved_review_options, dict):
                    options.update({"include_observations": bool(saved_review_options.get("include_observations", False)), "judgement_only": bool(saved_review_options.get("judgement_only", True))})
                self.current_review_options = options.copy()
                self.translation_review_results = {"entries": copy.deepcopy(saved_review_entries), "options": copy.deepcopy(options)}
            else:
                self.translation_review_results = None
                self.current_review_options = {"include_observations": False, "judgement_only": True}
            stored_context = session_data.get("context_artifacts")
            self.context_artifacts = copy.deepcopy(stored_context) if isinstance(stored_context, dict) else {}
            self._refresh_context_metadata()
            stored_jerga = session_data.get("jerga_report")
            if isinstance(stored_jerga, dict):
                self.translation_engine.last_jerga_report = copy.deepcopy(stored_jerga)
            else:
                self.translation_engine.last_jerga_report = None

            source_md_path = session_data.get("md_path")
            if source_md_path and Path(source_md_path).exists():
                self.source_md_path = source_md_path
            else:
                self.source_md_path = str(session_path.with_suffix(".md"))
            if self.source_md_path:
                self._register_markdown_variant("source", "Maquetado original", self.source_md_path)

            translated_doc = session_data.get("translated_document")
            translated_md_path = session_data.get("translated_md_path")
            if translated_doc and translated_md_path:
                self.translated_document = copy.deepcopy(translated_doc)  # type: ignore[arg-type]
                self.translated_meta = (
                    copy.deepcopy(session_data.get("translated_meta", {}))
                    if isinstance(session_data.get("translated_meta"), dict)
                    else {}
                )
                self.translated_md_path = translated_md_path
                self.last_md_path = translated_md_path
                if not Path(translated_md_path).exists():
                    self._log(
                        "Advertencia: el Markdown traducido referenciado en la sesión no se encontró en disco. Se regenerará tras guardar cambios."
                    )
            else:
                self.translated_document = None
                self.translated_meta = None
                self.translated_md_path = None
                self.last_md_path = self.source_md_path

            original_doc = session_data.get("translated_document_original")
            original_md_path = session_data.get("translated_md_path_original")
            original_meta = session_data.get("translated_meta_original")
            if original_doc and original_md_path:
                self.original_translated_document = copy.deepcopy(original_doc)  # type: ignore[arg-type]
                self.original_translated_meta = (
                    copy.deepcopy(original_meta) if isinstance(original_meta, dict) else None
                )
                self._register_markdown_variant("translated_original", "Traduccion original", original_md_path)
            elif translated_doc and translated_md_path:
                self.original_translated_document = copy.deepcopy(translated_doc)  # type: ignore[arg-type]
                self.original_translated_meta = copy.deepcopy(self.translated_meta) if self.translated_meta else None
                self._register_markdown_variant("translated_original", "Traduccion original", translated_md_path)
            else:
                self.original_translated_document = None
                self.original_translated_meta = None

            corrected_doc = session_data.get("translated_document_corrected")
            corrected_md_path = session_data.get("translated_md_path_corrected")
            corrected_meta = session_data.get("translated_meta_corrected")
            if corrected_doc and corrected_md_path:
                self.translated_document = copy.deepcopy(corrected_doc)  # type: ignore[arg-type]
                if isinstance(corrected_meta, dict):
                    self.translated_meta = copy.deepcopy(corrected_meta)
                self.translated_md_path = corrected_md_path
                self.last_md_path = corrected_md_path
                self._register_markdown_variant("translated_corrected", "Traduccion corregida", corrected_md_path)
            elif corrected_md_path:
                self._register_markdown_variant("translated_corrected", "Traduccion corregida", corrected_md_path)

            translation_info = None
            if isinstance(self.translated_meta, dict):
                translation_info = self.translated_meta.get("translation")
            if isinstance(translation_info, dict):
                self.last_style_profile_key = translation_info.get("style_profile")
            else:
                self.last_style_profile_key = None

            # Cargar configuraciones de estilo y modelos de IA si existen
            if "style_cfgs" in session_data:
                self.style_cfgs = session_data["style_cfgs"]
                self._log("Configuración de estilos cargada desde la sesión.")

            if "ai_models" in session_data:
                models = session_data["ai_models"]
                self.ai_model_var.set(models.get("general", "gpt-5-mini"))
                self.glossary_curation_model_var.set(models.get("glossary_curation", "gpt-5-mini"))
                self.glossary_translation_model_var.set(models.get("glossary_translation", "gpt-5-mini"))
                self.qa_model_var.set(models.get("qa", "gpt-5-mini"))
                self._log("Configuración de modelos de IA cargada desde la sesión.")

            self.btn_translate.config(state="normal")
            self.btn_review.config(state="normal" if self.translated_document else "disabled")
            self.btn_normalize.config(state="normal" if self.translated_document else "disabled")
            self.btn_structure_checker.config(state="normal")

            if self.translated_document and self.translated_md_path:
                self._log(f"Sesión cargada desde {session_path}. Traducción previa disponible en {self.translated_md_path}.")
            else:
                self._log(f"Sesión cargada desde {session_path}. Puede traducir o exportar a DOCX.")

            # Habilitar exportación inmediata con la variante preferida.
            self._export_controls_override = None
            preferred_key: Optional[str] = None
            for candidate in ["translated_corrected", "translated_original", "source"]:
                if candidate in self.markdown_variants:
                    preferred_key = candidate
                    break
            if preferred_key:
                self.docx_source_var.set(self.markdown_variants[preferred_key]["label"])
            self._refresh_markdown_selectors()
            self._update_export_controls()

        except Exception as e:
            messagebox.showerror("Error al cargar sesión", f"No se pudo cargar el archivo de sesión: {e}")
            self._log(f"Error al cargar: {e}")

    def _start_translation(self) -> None:
        if not self.last_document:
            messagebox.showwarning("Sin Documento", "Primero procesa un archivo para generar la estructura del documento.")
            return

        self.btn_translate.config(state="disabled")
        self.btn_process.config(state="disabled")
        self.btn_review.config(state="disabled")
        self._log("Iniciando preparación para la traducción...")
        self.translated_document = None
        self.translated_meta = None
        self.translated_md_path = None
        self.last_glossary = {}
        self.translation_review_results = None
        self._reset_token_metrics()

        # Mostrar la barra de progreso
        self.frm_progress.pack(fill=tk.X, padx=10, pady=5, before=self.log)
        self.progress_var.set(0)
        self.eta_var.set("Iniciando...")
        if hasattr(self, "_translation_start_time"):
            del self._translation_start_time

        thread = threading.Thread(target=self._translation_flow, daemon=True)
        thread.start()

    def _translation_flow(self) -> None:
        # 1. Preparation
        if not self.inputs:
            self.after(0, lambda: messagebox.showerror("Error", "No hay archivo de entrada seleccionado en la lista."))
            self.after(0, self._hide_progress_and_reenable_buttons)
            return

        input_path = self.inputs[0]
        output_dir = self._get_specific_output_dir(input_path)
        if not output_dir:
            self.after(0, self._hide_progress_and_reenable_buttons)
            return

        self.current_input_path = input_path
        self.current_output_dir = output_dir
        project_name = input_path.stem
        logs_dir, log_file = self._configure_logging_for_project(project_name=project_name, category="translation")
        self.current_logs_dir = logs_dir
        self.current_log_file = log_file
        self.after(
            0,
            lambda target=(log_file or logs_dir): self._log(f"Interacciones de IA se registrarán en: {target}"),
        )

        full_text = "\n\n".join([block.get("text", "") for block in self.last_document])
        ai_model = self.ai_model_var.get()

        input_filename = str(input_path)

        session_path = output_dir / f"{input_path.stem}.session.json"
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            session_data = {}
        self.current_session_path = session_path
        session_data.setdefault("in_path", str(input_path.resolve()))
        if "text" not in session_data:
            session_data["text"] = full_text
        session_data["document"] = self.last_document
        if "meta" not in session_data or not isinstance(session_data.get("meta"), dict):
            session_data["meta"] = copy.deepcopy(self.source_meta) if self.source_meta else {}
        if "md_path" not in session_data and self.source_md_path:
            session_data["md_path"] = self.source_md_path
        if "bio_inserted" in self.context_artifacts:
            self.context_artifacts["bio_inserted"] = False

        glossary_curation_model = self.glossary_curation_model_var.get()
        glossary_translation_model = self.glossary_translation_model_var.get()
        self.after(0, lambda: self._log(f"Modelo curacion glosario: {glossary_curation_model}"))
        self.after(0, lambda: self._log(f"Modelo traduccion glosario: {glossary_translation_model}"))

        # --- Preparation Phase ---
        detect_jerga_flag = self.detect_jerga_var.get()
        jerga_config = self._build_jerga_config() if detect_jerga_flag else None
        self.after(0, lambda: self._log("Iniciando fase de preparación (glosario, resumen, jerga)..."))

        # --- Glossary Reuse Logic ---
        base_filename = input_path.stem
        potential_glossary_path = output_dir / f"{base_filename}_glossary.json"
        reuse_glossary = False
        if self.resume_translation_var.get() and potential_glossary_path.exists():
            should_reuse = self._request_glossary_reuse(potential_glossary_path)
            if should_reuse:
                reuse_glossary = True
                self.after(0, lambda: self._log(f"El usuario ha decidido reutilizar el glosario: {potential_glossary_path}"))
            else:
                self.after(0, lambda: self._log("El usuario ha decidido generar un nuevo glosario."))

        artifact_paths = self.translation_engine.run_preparation(
            self.last_document,
            output_dir,
            input_filename=input_filename,
            model=ai_model,
            glossary_curation_model=self.glossary_curation_model_var.get(),
            jerga_config=jerga_config,
            reuse_glossary_path=potential_glossary_path if reuse_glossary else None,
        )

        glossary_path = artifact_paths.get("glossary")
        if not glossary_path:
            self.after(0, lambda: messagebox.showerror("Error de Preparación", "No se pudo generar el archivo de glosario."))
            self.after(0, self._hide_progress_and_reenable_buttons)
            return

        summary_path = artifact_paths.get("summary")
        if not summary_path:
            self.after(0, lambda: self._log("Advertencia: no se pudo generar el resumen."))

        # 2. User Interaction for Glossary
        self.after(0, lambda: self._log(f"Sugerencias de glosario guardadas en: {glossary_path}"))
        if self.interactive_mode_var.get():
            if not messagebox.askokcancel(
                "Revisar Glosario",
                (
                    f"Por favor, revisa y edita el archivo de glosario en:\n\n{glossary_path}\n\n"
                    "Completa las traducciones para los términos clave.\n"
                    "Haz clic en 'Aceptar' cuando estés listo para continuar."
                )
            ):
                self.after(0, lambda: self._log("Traducción cancelada por el usuario."))
                self.after(0, self._hide_progress_and_reenable_buttons)
                return

        try:
            with open(glossary_path, "r", encoding="utf-8") as f:
                glossary_data = json.load(f)
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error de Artefactos", f"No se pudieron leer los archivos de preparación: {e}"))
            self.after(0, self._hide_progress_and_reenable_buttons)
            return

        glossary: Dict[str, str] = {}
        if isinstance(glossary_data, dict) and "entries" in glossary_data:
            glossary_entries = glossary_data.get("entries", [])
            if isinstance(glossary_entries, list):
                for entry in glossary_entries:
                    if not isinstance(entry, dict):
                        continue
                    term = entry.get("lemma")
                    translation = entry.get("translation")
                    if isinstance(term, str) and isinstance(translation, str) and term and translation:
                        glossary[term] = translation

        if not glossary:
            self.after(0, lambda: self._log("El glosario está vacío. Continuando sin términos."))

        self.last_glossary = glossary
        session_data["glossary"] = glossary_data
        session_data["summary"] = summary
        use_punctuation_module = bool(self.use_punctuation_module_var.get())
        session_data["use_punctuation_module"] = use_punctuation_module

        # 3. Translation Phase
        style_label = self.style_profile_var.get()
        default_style_key = next(iter(STYLE_PROFILES)) if STYLE_PROFILES else ""
        style_profile = self.style_profile_choices.get(style_label, default_style_key)
        self.last_style_profile_key = style_profile
        style_notes = self.style_notes_var.get().strip()
        try:
            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            self.after(0, lambda e=exc: self._log(f"No se pudo actualizar la sesión con el resumen: {e}"))

        self.after(0, lambda: self._log(f"Guia de estilo aplicada: {style_label}"))
        if style_notes:
            self.after(0, lambda notes=style_notes: self._log(f"Notas de estilo: {notes}"))
        self.after(0, lambda: self._log("Iniciando bucle de traduccion... (esto puede tardar)"))
        simplified_log_path = output_dir / f"{input_path.stem}_translation_log.txt"
        resume_translation = self.resume_translation_var.get()
        if resume_translation:
            if simplified_log_path.exists():
                self.after(0, lambda: self._log("Reanudando traduccion desde el log existente."))
            else:
                self.after(0, lambda: self._log("No se encontro log previo; la traduccion empezara desde cero."))
                resume_translation = False

        test_mode = self.test_translation_var.get()
        if test_mode:
            self.after(0, lambda: self._log("Modo prueba activado: solo se traducirán los primeros 3 bloques."))

        aggregate_mode = self.aggregate_blocks_var.get()
        try:
            aggregate_target = int(self.aggregate_word_target_var.get())
        except (TypeError, ValueError, tk.TclError):
            aggregate_target = 2500
        if aggregate_target <= 0:
            aggregate_target = 2500
        if aggregate_mode:
            self.after(0, lambda target=aggregate_target: self._log(f"Agrupando párrafos hasta {target} palabras por bloque."))

        prev_translation_usage = dict(self.token_usage_by_stage.get("translation_chunk", {}))

        self.translated_document = self.translation_engine.run_translation(
            self.last_document,
            glossary,
            summary,
            style_profile=style_profile,
            style_notes=style_notes,
            model=ai_model,
            simplified_log_path=simplified_log_path,
            progress_callback=self._update_translation_progress,
            resume=resume_translation,
            max_translated_blocks=3 if test_mode else None,
            aggregate_mode=aggregate_mode,
            aggregate_target_words=aggregate_target,
            use_punctuation_module=use_punctuation_module,
            detect_jerga=self.detect_jerga_var.get(),
            jerga_config=self._build_jerga_config() if self.detect_jerga_var.get() else None,
        )
        if self.translation_engine.was_last_translation_cancelled():
            failure_reason = self.translation_engine.last_translation_failure_reason()
            if failure_reason:
                self.after(0, lambda msg=failure_reason: self._log(msg))
            else:
                self.after(0, lambda: self._log("La traducción se detuvo antes de completarse."))
            try:
                self._persist_translated_outputs(
                    session_data,
                    session_path,
                    output_dir,
                    input_path,
                    style_profile,
                    document_override=self.translated_document,
                    variant_key="translated_partial",
                    variant_label="Traducción parcial (interrumpida)",
                    make_current=False,
                )
            except Exception as exc:
                self.after(0, lambda e=exc: self._log(f"No se pudo guardar el Markdown parcial: {e}"))
            message = failure_reason or "La traducción se interrumpió antes de completarse. Reintenta más tarde."
            self.after(
                0,
                lambda msg=message: messagebox.showerror("Traducción interrumpida", msg),
            )
            self.after(0, self._hide_progress_and_reenable_buttons)
            return

        post_translation_usage = self.token_usage_by_stage.get("translation_chunk", {})
        delta_input = post_translation_usage.get("input", 0) - prev_translation_usage.get("input", 0)
        delta_output = post_translation_usage.get("output", 0) - prev_translation_usage.get("output", 0)
        delta_cached = post_translation_usage.get("cached_input", 0) - prev_translation_usage.get("cached_input", 0)
        delta_total = delta_input + delta_output + delta_cached
        if delta_total:
            formatted_input = f"{max(delta_input, 0):,}".replace(",", ".")
            formatted_output = f"{max(delta_output, 0):,}".replace(",", ".")
            formatted_cached = f"{max(delta_cached, 0):,}".replace(",", ".")
            formatted_total = f"{max(delta_total, 0):,}".replace(",", ".")
            log_message = (
                "Traducción: tokens consumidos en esta corrida -> "
                f"entrada {formatted_input}, salida {formatted_output}, cache {formatted_cached}, total {formatted_total}."
            )
            self.after(0, lambda msg=log_message: self._log(msg))

        if self.detect_jerga_var.get() and self.translation_engine.last_jerga_report:
            jerga_report = copy.deepcopy(self.translation_engine.last_jerga_report)
            session_data["jerga_report"] = jerga_report
            detected_count = len(jerga_report.get("detected", []))
            validated_count = len(jerga_report.get("validated", []))
            self.after(
                0,
                lambda dc=detected_count, vc=validated_count: self._log(
                    f"Jerga: analisis completado (candidatos={dc}, validados={vc})."
                ),
            )

        author_bio_text = (self.context_artifacts.get("author_bio") or "").strip()
        if author_bio_text and self.context_artifacts.get("append_author_bio"):
            already_present = any(
                block.get("_context_tag") == "author_bio" or block.get("text", "").strip() == author_bio_text
                for block in self.translated_document
            )
            if not already_present:
                if self.translated_document and self.translated_document[-1].get("type") != "hr":
                    self.translated_document.append(
                        {"type": "hr", "text": "---", "_context_tag": "author_bio_separator"}
                    )
                self.translated_document.append(
                    {"type": "h2", "text": "Sobre el autor", "_context_tag": "author_bio_heading"}
                )
                self.translated_document.append(
                    {"type": "p", "text": author_bio_text, "_context_tag": "author_bio"}
                )
                self.context_artifacts["bio_inserted"] = True
                self.after(0, lambda: self._log("Se anexó la mini bio del autor al final del libro."))
                wiki_sources = self.context_artifacts.get("wikipedia_sources", {})
                wiki_author = wiki_sources.get("author") or {}
                if wiki_author and (wiki_author.get("url") or wiki_author.get("source_url")):
                    self.after(
                        0,
                        lambda src=(wiki_author.get("url") or wiki_author.get("source_url")): self._log(
                            f"Bio basada en Wikipedia: {src}"
                        ),
                    )

        back_cover_text = (self.context_artifacts.get("back_cover") or "").strip()
        if back_cover_text and self.context_artifacts.get("export_back_cover"):
            back_cover_path = output_dir / f"{input_path.stem}_contraportada.md"
            try:
                back_cover_path.write_text(back_cover_text.strip() + "\n", encoding="utf-8")
                self.context_artifacts["back_cover_path"] = str(back_cover_path)
                self.after(0, lambda path=back_cover_path: self._log(f"Contraportada exportada en: {path}"))
            except Exception as exc:
                self.after(0, lambda e=exc: self._log(f"No se pudo exportar la contraportada: {e}"))
        elif "back_cover_path" in self.context_artifacts and not self.context_artifacts.get("export_back_cover"):
            # Si el usuario deshabilitó la exportación, limpiar referencia previa.
            self.context_artifacts.pop("back_cover_path", None)

        session_data["context_artifacts"] = copy.deepcopy(self.context_artifacts)

        # 4. Final Verification
        final_translated_text = "\n\n".join([block.get("text", "") for block in self.translated_document])
        alerts = self.translation_engine.consistency_checker.check(final_translated_text, glossary)

        if alerts:
            alert_msg = "Se encontraron posibles inconsistencias en la traduccion:\n\n" + "\n".join(alerts)
            self.after(0, lambda msg=alert_msg: self._log(msg))
            self.after(0, lambda msg=alert_msg: messagebox.showwarning("Alerta de Consistencia", msg))

        if self.skip_qa_var.get():
            self.after(0, lambda: self._log("QA automática omitida a petición del usuario."))
        else:
            qa_model = self.qa_model_var.get()
            source_blocks = [
                block.get("text", "")
                for block in self.last_document
                if block.get("type", "").startswith("h") or block.get("type") == "p"
            ]
            translated_blocks = [
                block.get("text", "")
                for block in self.translated_document
                if block.get("type", "").startswith("h") or block.get("type") == "p"
            ]
            qa_results = self.translation_engine.translation_qa.evaluate(
                source_blocks,
                translated_blocks,
                qa_model=qa_model,
            )

            trap_hits = qa_results.get("trap_hits", [])
            if trap_hits:
                for hit in trap_hits:
                    term = hit.get("term")
                    count = hit.get("count")
                    self.after(0, lambda t=term, c=count: self._log(f"QA alerta: {t} aparece {c} vez/veces en la traduccion."))

            qa_reviews = qa_results.get("qa_reviews", [])
            if qa_reviews:
                for review in qa_reviews:
                    idx = review.get("index")
                    tono = review.get("fidelidad_tono")
                    calcos = review.get("calcos_detectados") or []
                    self.after(
                        0,
                        lambda i=idx, t=tono, c=len(calcos): self._log(
                            f"QA muestreo bloque {i}: fidelidad={t}, calcos detectados={c}"
                        ),
                    )

            qa_report_path = output_dir / f"{Path(input_filename).stem}_qa_report.json"
            try:
                qa_report_path.write_text(json.dumps(qa_results, indent=2, ensure_ascii=False), encoding="utf-8")
                self.after(0, lambda path=qa_report_path: self._log(f"Informe QA guardado en: {path}"))
            except Exception as exc:
                self.after(0, lambda e=exc: self._log(f"No se pudo guardar el informe QA: {e}"))

        self._persist_translated_outputs(
            session_data,
            session_path,
            output_dir,
            input_path,
            style_profile,
            variant_key="translated_current",
            variant_label="Traduccion actual",
            make_current=True,
        )
        self._persist_translated_outputs(
            session_data,
            session_path,
            output_dir,
            input_path,
            style_profile,
            variant_key="translated_original",
            variant_label="Traduccion original",
            make_current=False,
        )

        self.after(0, self._log_token_summary)
        self.after(0, lambda: self._log("Traduccion completada. Ahora puedes exportar el documento traducido a DOCX."))
        self.after(0, self._hide_progress_and_reenable_buttons)

    def _get_specific_output_dir(self, input_path: Path) -> Optional[Path]:
        base_outdir = Path(self.outdir_var.get().strip() or "out")
        file_specific_outdir = base_outdir / input_path.stem
        try:
            file_specific_outdir.mkdir(parents=True, exist_ok=True)
            return file_specific_outdir
        except Exception as e:
            self._log(f"Error al crear directorio de salida para {input_path.stem}: {e}")
            return None

    def _hide_progress_and_reenable_buttons(self):
        self.frm_progress.pack_forget()
        self.btn_translate.config(state="normal")
        self.btn_process.config(state="normal")
        self._export_controls_override = None
        self._update_export_controls()
        self.btn_review.config(state="normal" if self.translated_document else "disabled")
        self.btn_normalize.config(state="normal" if self.translated_document else "disabled")
        self.btn_structure_checker.config(state="normal" if self.last_document else "disabled")

    def _open_structure_checker(self) -> None:
        selected_label = self.docx_source_var.get().strip()
        variant_key = self._docx_label_to_key.get(selected_label)

        if not variant_key:
            messagebox.showinfo("Sin Fuente", "Selecciona un documento para revisar desde la lista 'Fuente DOCX'.")
            return

        document_to_edit, _ = self._get_document_meta_for_variant(variant_key)

        if not document_to_edit:
            messagebox.showinfo("Sin Documento", f"No hay documento '{selected_label}' para revisar.")
            return

        dialog = StructureCheckerDialog(self, document_to_edit)
        result = dialog.show()

        if result is not None:
            self._set_document_for_variant(variant_key, result)
            self._log(f"Estructura del documento '{selected_label}' actualizada.")
            self._export_markdown_variant(variant_key)

    def _open_translation_review(self) -> None:
        if not self.translated_document:
            messagebox.showinfo("Sin traduccion", "Primero genera una traduccion para poder revisarla.")
            return

        entries = self._build_review_entries()
        initial_options = {"include_observations": False, "judgement_only": True}
        initial_options.update({"include_observations": bool(self.current_review_options.get("include_observations", False)), "judgement_only": bool(self.current_review_options.get("judgement_only", True))})
        if self.translation_review_results and isinstance(self.translation_review_results, dict):
            saved_opts = self.translation_review_results.get("options")
            if isinstance(saved_opts, dict):
                initial_options.update({"include_observations": bool(saved_opts.get("include_observations", False)), "judgement_only": bool(saved_opts.get("judgement_only", True))})
        self.current_review_options = initial_options.copy()

        previous_log_dir = api_logger.get_log_directory()
        project_name = self.current_input_path.stem if self.current_input_path else "general"
        review_logs_dir, review_log_file = self._configure_logging_for_project(
            project_name=project_name,
            category="review",
        )
        review_target = review_log_file or review_logs_dir
        self._log(f"Los logs de revisión se guardarán en: {review_target}")
        try:
            dialog = TranslationReviewDialog(
                parent=self,
                entries=entries,
                translation_engine=self.translation_engine,
                pricing_data=PRICING_DATA,
                glossary=self.last_glossary,
                qa_model=self.qa_model_var.get(),
                consistency_checker=self.translation_engine.consistency_checker,
                token_usage_by_model=self.token_usage_by_model,
                heading_paradigms=self.heading_paradigms,
                paradigm_callback=self._handle_heading_paradigm_change,
                progress_callback=self._handle_review_progress,
                include_observations=initial_options["include_observations"],
                judgement_only=initial_options["judgement_only"],
            )
            result = dialog.show()
        finally:
            api_logger.set_log_directory(previous_log_dir)
        self.current_review_options = dialog._current_mode_options()
        self.qa_model_var.set(dialog.review_model_var.get())
        self.heading_paradigms = copy.deepcopy(dialog.heading_paradigms)
        if not result:
            return

        entries_result = result.get("entries", [])
        options_result = result.get("options", self.current_review_options)
        if not isinstance(options_result, dict):
            options_result = {"include_observations": False, "judgement_only": True}
        self.current_review_options = options_result.copy()
        self.translation_review_results = {
            "entries": copy.deepcopy(entries_result),
            "options": copy.deepcopy(self.current_review_options),
        }
        revisions_applied = 0

        for item in entries_result:
            idx = item.get("block_index")
            if idx is None or idx >= len(self.translated_document):
                continue
            revised_text = (item.get("revised_text") or "").strip()
            if revised_text:
                if revised_text != self.translated_document[idx].get("text", ""):
                    self.translated_document[idx]["text"] = revised_text
                    revisions_applied += 1
            else:
                updated_text = item.get("translated_text")
                if isinstance(updated_text, str) and updated_text != self.translated_document[idx].get("text", ""):
                    self.translated_document[idx]["text"] = updated_text
                    revisions_applied += 1

        session_data: Dict[str, Any] = {}
        if self.current_session_path and self.current_session_path.exists():
            try:
                with open(self.current_session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                session_data = {}

        if session_data is not None:
            session_data.setdefault("meta", self.source_meta or {})
            session_data.setdefault("document", self.last_document)
            session_data["translation_review"] = entries_result
            session_data["translation_review_options"] = self.current_review_options
            session_data["heading_paradigms"] = self.heading_paradigms

        if revisions_applied > 0:
            self._log(f"Revisión: se aplicaron {revisions_applied} modificaciones a la traduccion.")
            if (
                self.current_session_path
                and self.current_output_dir
                and self.current_input_path
                and (self.last_style_profile_key or self.style_profile_choices)
            ):
                style_profile = self.last_style_profile_key or next(iter(self.style_profile_choices.values()))
                self._persist_translated_outputs(
                    session_data,
                    self.current_session_path,
                    self.current_output_dir,
                    self.current_input_path,
                    style_profile,
                    variant_key="translated_current",
                    variant_label="Traduccion actual",
                    make_current=True,
                )
                self._persist_translated_outputs(
                    session_data,
                    self.current_session_path,
                    self.current_output_dir,
                    self.current_input_path,
                    style_profile,
                    variant_key="translated_corrected",
                    variant_label="Traduccion corregida",
                    make_current=True,
                )
            else:
                self._log("No se pudo regenerar el Markdown traducido: falta contexto de sesión o de estilo.")
                if self.current_session_path:
                    try:
                        with open(self.current_session_path, "w", encoding="utf-8") as f:
                            json.dump(session_data, f, ensure_ascii=False, indent=2)
                    except Exception as exc:
                        self._log(f"No se pudo guardar la sesión tras la revisión: {exc}")
        else:
            self._log("Revisión completada sin cambios en el texto.")
            if self.current_session_path and session_data is not None:
                try:
                    with open(self.current_session_path, "w", encoding="utf-8") as f:
                        json.dump(session_data, f, ensure_ascii=False, indent=2)
                except Exception as exc:
                    self._log(f"No se pudo guardar la sesión tras la revisión: {exc}")

        if "translated_corrected" in self.markdown_variants:
            corrected_label = self.markdown_variants["translated_corrected"]["label"]
            self.docx_source_var.set(corrected_label)
        self._update_export_controls()

        self.btn_review.config(state="normal")

    def _update_translation_progress(self, current_step: int, total_steps: int):
        if not hasattr(self, "_translation_start_time"):
            self._translation_start_time = time.time()

        progress_percentage = (current_step / total_steps) * 100
        self.progress_var.set(progress_percentage)

        elapsed_time = time.time() - self._translation_start_time
        if current_step > 0:
            avg_time_per_step = elapsed_time / current_step
            remaining_steps = total_steps - current_step
            eta_seconds = remaining_steps * avg_time_per_step

            # Formatear ETA a MM:SS
            minutes, seconds = divmod(int(eta_seconds), 60)
            eta_str = f"{minutes:02d}:{seconds:02d}"
            self.eta_var.set(f"Tiempo restante: {eta_str}")
        else:
            self.eta_var.set("Calculando...")

        # Forzar actualización de la GUI
        self.update_idletasks()


    def _show_docx_export_dialog(self):
        selected_label = self.docx_source_var.get().strip()
        variant_key = self._docx_label_to_key.get(selected_label)

        if not variant_key:
            messagebox.showwarning("Sin fuente", "Selecciona un Markdown a exportar desde la lista.")
            return

        document_to_export, meta_to_use = self._get_document_meta_for_variant(variant_key)
        if not document_to_export:
            messagebox.showwarning("Sin Documento", "No hay documento para exportar. Procesa un archivo primero.")
            return

        md_path = self._get_markdown_variant_path(variant_key)
        if md_path and not Path(md_path).exists():
            self._log(f"No se encontró {md_path}. Se regenerará antes de exportar a DOCX.")
            self._export_markdown_variant(variant_key)
            md_path = self._get_markdown_variant_path(variant_key)

        try:
            md_text = ""
            if md_path and Path(md_path).exists():
                md_text = Path(md_path).read_text(encoding="utf-8")
            else:
                md_text = render(document_to_export, meta_to_use or {}, add_toc=False)

            _, used_styles_set = parse_markdown_blocks(md_text)
            used_styles_ordered = [sk for sk in STYLE_KEYS_ORDER if sk in used_styles_set]
            if not used_styles_ordered:
                used_styles_ordered.append("paragraph")

            dlg = StylesDialog(self, used_styles_ordered, self.style_cfgs)
            self.wait_window(dlg)

            self._save_style_cfgs()

            self._log("Exportando a DOCX con estilos personalizados...")

            input_path = self.current_input_path or (self.inputs[0] if self.inputs else None)
            if not input_path:
                messagebox.showwarning("Sin entrada", "No se pudo determinar el archivo de salida.")
                return
            output_dir = self._get_specific_output_dir(input_path)
            if not output_dir:
                return

            if md_path:
                export_path_stem = Path(md_path).stem
            else:
                export_path_stem = f"{input_path.stem}_{variant_key}"
            output_path = output_dir / f"{export_path_stem}.docx"

            output_real_path = convert_markdown_to_docx(
                document=document_to_export,
                style_cfgs=self.style_cfgs,
                output_path=str(output_path),
            )

            session_path = output_dir / f"{input_path.stem}.session.json"
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                session_data = {}

            session_data["style_cfgs"] = self.style_cfgs
            session_data["ai_models"] = {
                "general": self.ai_model_var.get(),
                "glossary_curation": self.glossary_curation_model_var.get(),
                "glossary_translation": self.glossary_translation_model_var.get(),
                "qa": self.qa_model_var.get(),
            }
            session_data["use_punctuation_module"] = bool(self.use_punctuation_module_var.get())
            session_data["detect_jerga"] = bool(self.detect_jerga_var.get())

            with open(session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)

            self._log(f"Configuracion de estilos y modelos guardada en la sesion: {session_path}")

            self._log(f"Exportacion a DOCX completada: {output_real_path}")
            messagebox.showinfo("Exportacion completada", f"Archivo guardado en:\n{output_real_path}")

        except Exception as e:
            self._log(f"Error durante la exportacion a DOCX: {e}")
            messagebox.showerror("Error de exportacion", f"No se pudo exportar a DOCX: {e}")

    def destroy(self) -> None:
        if getattr(self, "_api_trace_listener", None):
            unregister_api_trace_listener(self._api_trace_listener)  # type: ignore[arg-type]
            self._api_trace_listener = None
        super().destroy()


def launch_app() -> None:
    app = Txt2MdApp()
    app.mainloop()


if __name__ == "__main__":
    launch_app()

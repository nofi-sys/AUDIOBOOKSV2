from __future__ import annotations

import copy
import threading
import textwrap
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText


STATUS_LABELS = {
    "pendiente": "[ ] Pendiente",
    "ok": "[OK] OK",
    "dudoso": "[?] Dudoso",
    "mal": "[X] Mal",
}

STATUS_OPTIONS = [
    ("ok", STATUS_LABELS["ok"]),
    ("dudoso", STATUS_LABELS["dudoso"]),
    ("mal", STATUS_LABELS["mal"]),
]


try:  # pragma: no cover - import fallback para ejecuciÃ³n directa
    from .costs import calculate_cost, PRICING_DATA
except ImportError:  # pragma: no cover - ejecuciÃ³n directa
    from costs import calculate_cost, PRICING_DATA  # type: ignore


if TYPE_CHECKING:  # pragma: no cover - solo para tips
    from .translation_engine import TranslationEngine, ConsistencyChecker


class DebugDialog(tk.Toplevel):
    """Un diÃ¡logo para mostrar informaciÃ³n de depuraciÃ³n detallada."""

    def __init__(self, parent: tk.Misc, title: str, prompt: str, response: str, error: str) -> None:
        super().__init__(parent)
        self.transient(parent)
        self.title(title)
        self.geometry("800x600")

        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Prompt
        prompt_frame = ttk.LabelFrame(main_frame, text="Prompt Enviado a la IA")
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        prompt_text = ScrolledText(prompt_frame, wrap=tk.WORD, height=10)
        prompt_text.insert(tk.END, prompt)
        prompt_text.configure(state="disabled")
        prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Response
        response_frame = ttk.LabelFrame(main_frame, text="Respuesta Recibida de la IA")
        response_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        response_text = ScrolledText(response_frame, wrap=tk.WORD, height=5)
        response_text.insert(tk.END, response or " (sin respuesta)")
        response_text.configure(state="disabled")
        response_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Error
        error_frame = ttk.LabelFrame(main_frame, text="Error de Procesamiento")
        error_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        error_text_widget = ScrolledText(error_frame, wrap=tk.WORD, height=5)
        error_text_widget.insert(tk.END, error)
        error_text_widget.configure(state="disabled")
        error_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Close button
        close_button = ttk.Button(main_frame, text="Cerrar", command=self.destroy)
        close_button.pack(pady=10)


class HeadingParadigmDialog(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Misc,
        heading_types: List[str],
        options: Dict[str, List[str]],
        current_values: Dict[str, str],
        title: str,
    ) -> None:
        super().__init__(parent)
        self.transient(parent)
        self.title(title)
        self.resizable(False, False)
        self.result: Dict[str, str] = {}
        self._fields: Dict[str, ttk.Combobox] = {}

        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            container,
            text="Selecciona o edita el formato correcto de cada encabezado. Este formato servirÃ¡ como referencia durante la revisiÃ³n.",
            wraplength=420,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 10))

        for heading_type in heading_types:
            frame = ttk.Frame(container)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=f"Encabezado {heading_type.upper()}:").pack(anchor="w")
            candidates = sorted({text.strip() for text in options.get(heading_type, []) if text.strip()})
            combo = ttk.Combobox(frame, values=candidates, width=58, state="normal")
            combo.pack(fill=tk.X, pady=2)
            if heading_type in current_values and current_values[heading_type].strip():
                combo.set(current_values[heading_type].strip())
            elif candidates:
                combo.set(candidates[0])
            self._fields[heading_type] = combo

        buttons = ttk.Frame(container)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Aceptar", command=self._on_accept).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Cancelar", command=self._on_cancel).pack(side=tk.RIGHT)

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)

    def _on_accept(self) -> None:
        result: Dict[str, str] = {}
        for heading_type, widget in self._fields.items():
            value = widget.get().strip()
            if not value:
                messagebox.showwarning(
                    "Formato requerido",
                    f"Ingrese un formato vÃ¡lido para {heading_type.upper()}.",
                    parent=self,
                )
                widget.focus_set()
                return
            result[heading_type] = value
        self.result = result
        self.destroy()

    def _on_cancel(self) -> None:
        self.result = {}
        self.destroy()


class TranslationReviewDialog(tk.Toplevel):
    """Dialogo para revisar y corregir bloques traducidos."""

    def __init__(
        self,
        parent: tk.Misc,
        entries: List[Dict[str, Any]],
        translation_engine: "TranslationEngine",
        pricing_data: Optional[Dict[str, Dict[str, float]]] = None,
        glossary: Optional[Dict[str, str]] = None,
        qa_model: Optional[str] = None,
        consistency_checker: Optional["ConsistencyChecker"] = None,
        token_usage_by_model: Optional[Dict[str, Dict[str, int]]] = None,
        heading_paradigms: Optional[Dict[str, str]] = None,
        paradigm_callback: Optional[Callable[[Dict[str, str]], None]] = None,
        progress_callback: Optional[Callable[[List[Dict[str, Any]], Dict[str, bool]], None]] = None,
        include_observations: bool = False,
        judgement_only: bool = True,
    ) -> None:
        super().__init__(parent)
        self.transient(parent)
        self.title("Revision de traduccion")
        self.geometry("1100x720")
        self.minsize(900, 600)

        self.engine = translation_engine
        self.pricing_data: Dict[str, Dict[str, float]] = pricing_data or PRICING_DATA.copy()
        self.glossary = glossary or {}
        self.qa_model = qa_model or "gpt-5-mini"
        self.consistency_checker = consistency_checker
        self.token_usage_by_model = token_usage_by_model or {}
        self.heading_paradigms: Dict[str, str] = copy.deepcopy(heading_paradigms) if heading_paradigms else {}

        self.review_model_var = tk.StringVar(value=qa_model or "gpt-5-mini")
        self.process_single_row_var = tk.BooleanVar(value=False)
        self.reprocess_completed_var = tk.BooleanVar(value=False)
        self.include_observations_var = tk.BooleanVar(value=include_observations and not judgement_only)
        self.judgement_only_var = tk.BooleanVar(value=judgement_only)
        self._updating_mode_flags = False
        if judgement_only:
            self.include_observations_var.set(False)

        self.entries: List[Dict[str, Any]] = copy.deepcopy(entries)
        self.tree_items: List[str] = []
        self.selected_index: Optional[int] = None
        self.result: Optional[Dict[str, Any]] = None
        self._ai_thread: Optional[threading.Thread] = None
        self._cancel_event = threading.Event()
        self._paradigm_callback = paradigm_callback
        self._progress_callback = progress_callback
        self.heading_notice_var = tk.StringVar(value="")

        self._build_ui()
        self._populate_tree()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grab_set()
        self._ensure_heading_paradigms()
        self.judgement_only_var.trace_add("write", lambda *_: self._on_mode_flags_change())
        self.include_observations_var.trace_add("write", lambda *_: self._emit_progress())
        self._on_mode_flags_change()

    def show(self) -> Optional[Dict[str, Any]]:
        self.wait_window(self)
        return self.result

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        columns = ("original", "translated", "status", "observaciones", "revision")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("original", text="Original")
        self.tree.heading("translated", text="Traduccion")
        self.tree.heading("status", text="Estado")
        self.tree.heading("observaciones", text="Observaciones")
        self.tree.heading("revision", text="Revision propuesta")

        self.tree.column("original", width=240, anchor=tk.W)
        self.tree.column("translated", width=240, anchor=tk.W)
        self.tree.column("status", width=120, anchor=tk.CENTER)
        self.tree.column("observaciones", width=240, anchor=tk.W)
        self.tree.column("revision", width=240, anchor=tk.W)

        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_select_entry)
        self.tree.bind("<Double-1>", self._on_double_click)

        detail_frame = ttk.Frame(main_frame)
        detail_frame.pack(fill=tk.BOTH, expand=False, pady=(12, 0))

        self.txt_original = self._create_text_widget(detail_frame, "Original", column=0, editable=False)
        self.txt_translation = self._create_text_widget(detail_frame, "Traduccion actual", column=1, editable=False)
        self.txt_revision = self._create_text_widget(detail_frame, "Revision propuesta", column=2, editable=True)

        detail_frame.columnconfigure(0, weight=1)
        detail_frame.columnconfigure(1, weight=1)
        detail_frame.columnconfigure(2, weight=1)

        lower_frame = ttk.Frame(main_frame)
        lower_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))

        ttk.Label(lower_frame, textvariable=self.heading_notice_var, foreground="darkorange").pack(
            fill=tk.X, expand=False, pady=(0, 4)
        )

        status_box = ttk.Frame(lower_frame)
        status_box.pack(fill=tk.X, expand=False)
        ttk.Label(status_box, text="Estado del bloque:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="pendiente")
        status_values = [STATUS_LABELS["pendiente"]] + [label for _code, label in STATUS_OPTIONS]
        self.status_combo = ttk.Combobox(status_box, state="readonly", values=status_values)
        self.status_combo.pack(side=tk.LEFT, padx=(6, 0))
        self.status_combo.current(0)
        self.status_combo.current(0)
        self.status_combo.bind("<<ComboboxSelected>>", self._on_status_combo_selected)

        bulk_values = status_values
        self.bulk_status_var = tk.StringVar(value=STATUS_LABELS["pendiente"])
        self.bulk_status_combo = ttk.Combobox(status_box, state="readonly", values=bulk_values, textvariable=self.bulk_status_var, width=18)
        self.bulk_status_combo.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(status_box, text="Aplicar a selecciÃ³n", command=self._apply_status_bulk).pack(side=tk.LEFT, padx=(5, 0))

        self.txt_observaciones = ScrolledText(lower_frame, height=4, wrap=tk.WORD)
        self.txt_observaciones.pack(fill=tk.BOTH, expand=False, pady=(6, 0))

        mode_frame = ttk.LabelFrame(main_frame, text="Modo de revisión")
        mode_frame.pack(fill=tk.X, expand=False, pady=(10, 0))

        self.chk_include_observaciones = ttk.Checkbutton(
            mode_frame,
            text="Solicitar observaciones",
            variable=self.include_observations_var,
            command=self._on_mode_flags_change,
        )
        self.chk_include_observaciones.grid(row=0, column=0, sticky="w", padx=8, pady=(4, 2))

        self.chk_judgement_only = ttk.Checkbutton(
            mode_frame,
            text="Solo estado (OK/Dudoso/Mal)",
            variable=self.judgement_only_var,
            command=self._on_mode_flags_change,
        )
        self.chk_judgement_only.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 6))

        mode_frame.columnconfigure(0, weight=1)

        actions = ttk.Frame(main_frame)
        actions.pack(fill=tk.X, expand=False, pady=(12, 0))

        self.btn_ai = ttk.Button(actions, text="Revision IA", command=self._run_ai_review)
        self.btn_ai.pack(side=tk.LEFT, padx=5)

        self.btn_stop_ai = ttk.Button(
            actions,
            text="Detener analisis",
            command=self._on_stop_ai,
            state=tk.DISABLED,
        )
        self.btn_stop_ai.pack(side=tk.LEFT, padx=5)

        self.btn_apply_revision = ttk.Button(actions, text="Aplicar revision seleccionada", command=self._apply_revision_to_translation)
        self.btn_apply_revision.pack(side=tk.LEFT, padx=5)

        self.btn_accept_all = ttk.Button(
            actions,
            text="Aceptar todas las revisiones",
            command=self._apply_all_revisions,
        )
        self.btn_accept_all.pack(side=tk.LEFT, padx=5)

        self.btn_manage_paradigms = ttk.Button(
            actions,
            text="Gestionar paradigmas",
            command=self._manage_heading_paradigms,
        )
        self.btn_manage_paradigms.pack(side=tk.LEFT, padx=5)

        self.btn_glossary = ttk.Button(actions, text="Verificar glosario", command=self._run_glossary_check)
        self.btn_glossary.pack(side=tk.LEFT, padx=5)

        self.btn_cost = ttk.Button(actions, text="Estimar Costo", command=self._show_cost_estimation)
        self.btn_cost.pack(side=tk.LEFT, padx=5)

        ttk.Label(actions, text="Modelo de RevisiÃ³n:").pack(side=tk.LEFT, padx=(10, 2))
        model_options = sorted(self.pricing_data.keys())
        current_model = self.review_model_var.get()
        if current_model and current_model not in model_options:
            model_options.append(current_model)
        self.review_model_combo = ttk.Combobox(actions, textvariable=self.review_model_var, values=model_options, state="readonly", width=15)
        self.review_model_combo.pack(side=tk.LEFT, padx=2)
        if current_model and current_model in model_options:
            self.review_model_combo.set(current_model)
        elif model_options:
            self.review_model_combo.current(0)

        self.chk_single_row = ttk.Checkbutton(actions, text="Procesar solo la fila seleccionada", variable=self.process_single_row_var)
        self.chk_single_row.pack(side=tk.LEFT, padx=(10, 0))

        self.chk_reprocess = ttk.Checkbutton(
            actions,
            text="Incluir filas resueltas",
            variable=self.reprocess_completed_var,
        )
        self.chk_reprocess.pack(side=tk.LEFT, padx=(10, 0))

        self.btn_close = ttk.Button(actions, text="Cerrar", command=self._on_close)
        self.btn_close.pack(side=tk.RIGHT, padx=5)

    def _create_text_widget(self, parent: ttk.Frame, label: str, column: int, editable: bool) -> ScrolledText:
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, sticky="nsew", padx=5)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        text_widget = ScrolledText(frame, height=8, wrap=tk.WORD)
        if not editable:
            text_widget.configure(state="disabled")
        text_widget.pack(fill=tk.BOTH, expand=True)
        return text_widget

    # Data binding -------------------------------------------------------
    def _populate_tree(self) -> None:
        # Limpiar cualquier entrada existente en el Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree_items.clear()
        if not self.entries:
            return

        for entry in self.entries:
            # Asegurarse de que cada entrada tenga un estado por defecto si no estÃ¡ presente
            if "status" not in entry:
                entry["status"] = "pendiente"

            item_id = self.tree.insert(
                "",
                tk.END,
                values=self._entry_values(entry),
            )
            self.tree_items.append(item_id)

        if self.tree_items:
            self.tree.selection_set(self.tree_items[0])
            self.tree.focus(self.tree_items[0])
            self._on_select_entry()
            self._update_heading_notice()
        else:
            self.heading_notice_var.set("")
        self._emit_progress()

    def _emit_progress(self) -> None:
        """Notifica el avance actual al callback externo, si existe."""
        if not self._progress_callback:
            return
        try:
            payload = copy.deepcopy(self.entries)
            options = self._current_mode_options()
            self._progress_callback(payload, options)
        except Exception as exc:
            print(f"[TranslationReviewDialog] No se pudo emitir progreso: {exc}")

    def _update_heading_notice(self) -> None:
        """Actualiza el mensaje de ayuda sobre encabezados según la fila activa."""
        if self.selected_index is None or self.selected_index >= len(self.entries):
            self.heading_notice_var.set("")
            return

        entry = self.entries[self.selected_index]
        block_type = str(entry.get("type", ""))
        if not block_type.startswith("h"):
            self.heading_notice_var.set("")
            return

        expected = self.heading_paradigms.get(block_type, "").strip()
        if expected:
            self.heading_notice_var.set(f"Encabezado {block_type.upper()} debe seguir el paradigma: {expected}")
        else:
            self.heading_notice_var.set(f"Encabezado {block_type.upper()} sin paradigma definido. Usa 'Gestionar paradigmas'.")

    @staticmethod
    def _has_observation_error(entry: Dict[str, Any]) -> bool:
        observations = str(entry.get("observaciones", ""))
        return "error" in observations.lower()

    def _entry_values(self, entry: Dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            self._shorten(entry.get("original_text", "")),
            self._shorten(entry.get("translated_text", "")),
            STATUS_LABELS.get(entry.get("status", "pendiente"), STATUS_LABELS["pendiente"]),
            self._shorten(entry.get("observaciones", "")),
            self._shorten(entry.get("revised_text", "")),
        )

    @staticmethod
    def _shorten(text: str, width: int = 80) -> str:
        return textwrap.shorten(text.replace("\n", " "), width=width, placeholder="...") if text else ""

    def _on_select_entry(self, _event: Optional[tk.Event] = None) -> None:
        self._persist_detail_changes()
        selection = self.tree.selection()
        if not selection:
            self.selected_index = None
            self._clear_detail_widgets()
            return
        item_id = selection[0]
        if item_id not in self.tree_items:
            return
        self.selected_index = self.tree_items.index(item_id)
        entry = self.entries[self.selected_index]
        self._load_entry_into_detail(entry)
        self._update_heading_notice()

    def _load_entry_into_detail(self, entry: Dict[str, Any]) -> None:
        self._set_text(self.txt_original, entry.get("original_text", ""), editable=False)
        self._set_text(self.txt_translation, entry.get("translated_text", ""), editable=False)
        self._set_text(self.txt_revision, entry.get("revised_text", ""), editable=True)
        self._set_text(self.txt_observaciones, entry.get("observaciones", ""), editable=True)
        status_code = entry.get("status", "pendiente")
        label = STATUS_LABELS.get(status_code, STATUS_LABELS["pendiente"])
        self.status_combo.set(label if label in self.status_combo.cget("values") else STATUS_LABELS["pendiente"])
        self._update_heading_notice()

    def _clear_detail_widgets(self) -> None:
        self._set_text(self.txt_original, "", editable=False)
        self._set_text(self.txt_translation, "", editable=False)
        self._set_text(self.txt_revision, "", editable=True)
        self._set_text(self.txt_observaciones, "", editable=True)
        self.status_combo.set(STATUS_LABELS["pendiente"])
        self.heading_notice_var.set("")

    @staticmethod
    def _set_text(widget: ScrolledText, text: str, editable: bool) -> None:
        state = widget.cget("state")
        if not editable:
            widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        if not editable:
            widget.configure(state="disabled")

    def _persist_detail_changes(self) -> None:
        if self.selected_index is None:
            return
        entry = self.entries[self.selected_index]
        entry["revised_text"] = self.txt_revision.get("1.0", tk.END).strip()
        entry["observaciones"] = self.txt_observaciones.get("1.0", tk.END).strip()
        label = self.status_combo.get()
        for code, option_label in STATUS_OPTIONS:
            if option_label == label:
                entry["status"] = code
                break
        else:
            entry["status"] = "pendiente"
        self._refresh_tree_row(self.selected_index)
        self._emit_progress()
        self._update_heading_notice()

    def _on_status_combo_selected(self, _event: Optional[tk.Event] = None) -> None:
        label = self.status_combo.get()
        label_to_code = {STATUS_LABELS["pendiente"]: "pendiente"}
        for code, option_label in STATUS_OPTIONS:
            label_to_code[option_label] = code
        target_code = label_to_code.get(label)
        if not target_code:
            return

        self._persist_detail_changes()

        selection = list(self.tree.selection())
        if len(selection) <= 1:
            return

        changed = False
        for item_id in selection:
            if item_id not in self.tree_items:
                continue
            idx = self.tree_items.index(item_id)
            if idx == self.selected_index or idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if entry.get("status") == target_code:
                continue
            entry["status"] = target_code
            self._refresh_tree_row(idx)
            changed = True

        if changed:
            self._emit_progress()
            if self.selected_index is not None:
                self._update_heading_notice()

    def _refresh_tree_row(self, index: int) -> None:
        if index < 0 or index >= len(self.tree_items):
            return
        item_id = self.tree_items[index]
        self.tree.item(item_id, values=self._entry_values(self.entries[index]))

    def _collect_heading_samples(self) -> Dict[str, List[str]]:
        samples: Dict[str, List[str]] = {}
        for entry in self.entries:
            block_type = str(entry.get("type", ""))
            if not block_type.startswith("h"):
                continue
            candidate = (
                entry.get("revised_text")
                or entry.get("translated_text")
                or entry.get("original_text")
                or ""
            ).strip()
            if not candidate:
                continue
            samples.setdefault(block_type, [])
            if candidate not in samples[block_type]:
                samples[block_type].append(candidate)
        return samples

    def _ensure_heading_paradigms(self) -> None:
        heading_types = {
            str(entry.get("type", ""))
            for entry in self.entries
            if str(entry.get("type", "")).startswith("h")
        }
        missing = [ht for ht in sorted(heading_types) if ht and ht not in self.heading_paradigms]
        if not missing:
            self._notify_paradigm_change()
            return
        self._open_paradigm_dialog(missing, "Seleccionar formato de encabezado")

    def _manage_heading_paradigms(self) -> None:
        heading_types = sorted(
            {
                str(entry.get("type", ""))
                for entry in self.entries
                if str(entry.get("type", "")).startswith("h")
            }
        )
        if not heading_types:
            messagebox.showinfo("Sin encabezados", "No hay encabezados disponibles en esta revisiÃ³n.")
            return
        self._open_paradigm_dialog(heading_types, "Gestionar formatos de encabezado")

    def _open_paradigm_dialog(self, heading_types: List[str], title: str) -> None:
        samples = self._collect_heading_samples()
        dialog = HeadingParadigmDialog(
            parent=self,
            heading_types=heading_types,
            options=samples,
            current_values=self.heading_paradigms,
            title=title,
        )
        self.wait_window(dialog)
        if dialog.result:
            self.heading_paradigms.update(dialog.result)
            self._notify_paradigm_change()
            self._update_heading_notice()

    def _notify_paradigm_change(self) -> None:
        if self._paradigm_callback:
            try:
                self._paradigm_callback(copy.deepcopy(self.heading_paradigms))
            except Exception as exc:
                print(f"[TranslationReviewDialog] No se pudo comunicar el cambio de paradigma: {exc}")
        self._emit_progress()

    def _current_mode_options(self) -> Dict[str, bool]:
        include = bool(self.include_observations_var.get())
        if self.judgement_only_var.get():
            include = False
        return {"include_observations": include, "judgement_only": bool(self.judgement_only_var.get())}

    def _on_mode_flags_change(self) -> None:
        if getattr(self, "_updating_mode_flags", False):
            return
        self._updating_mode_flags = True
        try:
            if self.judgement_only_var.get():
                if self.include_observations_var.get():
                    self.include_observations_var.set(False)
                if hasattr(self, "chk_include_observaciones"):
                    self.chk_include_observaciones.configure(state=tk.DISABLED)
            else:
                if hasattr(self, "chk_include_observaciones"):
                    self.chk_include_observaciones.configure(state=tk.NORMAL)
            self._emit_progress()
        finally:
            self._updating_mode_flags = False

    # Actions -------------------------------------------------------------
    def _run_ai_review(self) -> None:
        if self._ai_thread and self._ai_thread.is_alive():
            messagebox.showinfo("Revision en curso", "Ya hay una revision ejecutandose.")
            return

        self._persist_detail_changes()
        model = self.review_model_var.get()
        if not model:
            messagebox.showwarning("Sin modelo", "Por favor, selecciona un modelo de IA para la revision.")
            return

        include_completed = self.reprocess_completed_var.get()

        rows_to_process_indices: List[int] = []
        if self.process_single_row_var.get():
            if self.selected_index is not None:
                rows_to_process_indices.append(self.selected_index)
        else:
            rows_to_process_indices = [
                i
                for i, entry in enumerate(self.entries)
                if include_completed or entry.get("status") != "ok"
            ]

        if not rows_to_process_indices:
            messagebox.showinfo("Nada que revisar", "No hay filas pendientes para procesar.")
            return

        judgement_only = bool(self.judgement_only_var.get())
        include_observations = bool(self.include_observations_var.get()) and not judgement_only
        self._cancel_event.clear()

        def worker() -> None:
            for index in rows_to_process_indices:
                if self._cancel_event.is_set():
                    break

                entry = self.entries[index]
                original_text = entry.get("original_text", "")
                translated_text = entry.get("translated_text", "")
                block_type = str(entry.get("type", ""))
                extra_guidance = None
                expected_format = self.heading_paradigms.get(block_type) if block_type.startswith("h") else None
                if expected_format:
                    current_display = translated_text.strip() or "(sin texto)"
                    extra_guidance = (
                        f"Este bloque es un encabezado de tipo {block_type.upper()}. "
                        f"Paradigma aprobado: '{expected_format}'. "
                        "Entrega una versión final que replique exactamente capitalización, puntuación, espaciado y signos gráficos del paradigma, "
                        "pero sin alterar el contenido léxico ni la numeración del título. "
                        "Solo aplica ajustes tipográficos necesarios (tildes, traducción de la etiqueta genérica, etc.) y conserva la cifra o nombre propio tal como aparece en el texto base. "
                        f"Texto actual a revisar: '{current_display}'. "
                        "Si notas diferencias, corrígelas para que el resultado coincida con el paradigma en estilo, manteniendo el valor textual original. Esta instrucción tiene prioridad incluso cuando el modo dictamen rápido esté activo."
                    )

                if not translated_text.strip():
                    continue

                prompt = ""
                response_content = ""
                try:
                    prompt = self.engine.translation_qa.review_block_prompt_template(
                        original_text,
                        translated_text,
                        self.glossary,
                        include_observations=include_observations,
                        verdict_only=judgement_only,
                    )

                    if self._cancel_event.is_set():
                        break

                    result, response_content = self.engine.review_block(
                        original_text,
                        translated_text,
                        self.glossary,
                        model=model,
                        extra_guidance=extra_guidance,
                        include_observations=include_observations,
                        verdict_only=judgement_only,
                    )
                    if not result:
                        raise ValueError("La IA no devolvio un resultado estructurado.")

                    if self._cancel_event.is_set():
                        break

                    self.after(0, lambda i=index, r=result: self._apply_ai_result_to_row(i, r))

                except Exception as exc:
                    error_message = f"Ocurrio un error al revisar una fila: {exc}"
                    self.after(
                        0,
                        lambda p=prompt, r=response_content, e=error_message: DebugDialog(
                            self, "Error de Revision IA", p, r, e
                        ).grab_set(),
                    )
                    error_result = {
                        "status": "mal",
                        "observaciones": f"Error: {exc}",
                        "revision": translated_text,
                    }
                    self.after(0, lambda i=index, r=error_result: self._apply_ai_result_to_row(i, r))

            self.after(0, self._on_ai_worker_finished)

        self._toggle_ai_controls(running=True)
        self._ai_thread = threading.Thread(target=worker, daemon=True)
        self._ai_thread.start()

    def _toggle_ai_controls(self, *, running: bool) -> None:
        if running:
            self.btn_ai.configure(state=tk.DISABLED)
            self.btn_stop_ai.configure(state=tk.NORMAL)
            self.btn_apply_revision.configure(state=tk.DISABLED)
            self.btn_accept_all.configure(state=tk.DISABLED)
            self.btn_glossary.configure(state=tk.DISABLED)
            self.btn_cost.configure(state=tk.DISABLED)
            self.btn_close.configure(state=tk.DISABLED)
            self.chk_single_row.configure(state=tk.DISABLED)
            self.review_model_combo.configure(state="disabled")
        else:
            self.btn_ai.configure(state=tk.NORMAL)
            self.btn_stop_ai.configure(state=tk.DISABLED)
            self.btn_apply_revision.configure(state=tk.NORMAL)
            self.btn_accept_all.configure(state=tk.NORMAL)
            self.btn_glossary.configure(state=tk.NORMAL)
            self.btn_cost.configure(state=tk.NORMAL)
            self.btn_close.configure(state=tk.NORMAL)
            self.chk_single_row.configure(state=tk.NORMAL)
            self.review_model_combo.configure(state="readonly")

    def _on_stop_ai(self) -> None:
        if self._ai_thread and self._ai_thread.is_alive():
            self._cancel_event.set()
            self.btn_stop_ai.configure(state=tk.DISABLED)

    def _on_ai_worker_finished(self) -> None:
        was_cancelled = self._cancel_event.is_set()
        self._ai_thread = None
        self._toggle_ai_controls(running=False)
        if was_cancelled:
            self._cancel_event.clear()
            messagebox.showinfo("Revision detenida", "El analisis se detuvo antes de completar todas las filas.")

    def _apply_ai_result_to_row(self, index: int, result: Dict[str, Any]) -> None:
        if index < 0 or index >= len(self.entries):
            return

        entry = self.entries[index]
        block_type = str(entry.get("type", ""))
        status = str(result.get("status", entry.get("status", "pendiente"))).lower()
        if status not in STATUS_LABELS:
            status = "dudoso"
        entry["status"] = status
        raw_observations = result.get("observaciones")
        observations = "" if raw_observations is None else str(raw_observations).strip()
        entry["observaciones"] = observations
        revision = result.get("revision", "")
        if isinstance(revision, str):
            new_text = revision.strip()
            if not new_text:
                new_text = entry.get("translated_text", "")
            entry["revised_text"] = new_text

        if not self.include_observations_var.get():
            entry["observaciones"] = ""
        if self.judgement_only_var.get() and not block_type.startswith("h"):
            entry["revised_text"] = ""

        self._refresh_tree_row(index)
        # Si la fila procesada es la que estÃ¡ seleccionada, actualizamos los detalles
        if index == self.selected_index:
            self._load_entry_into_detail(entry)
        self._emit_progress()

    def _apply_revision_to_translation(self) -> None:
        if self.selected_index is None:
            messagebox.showinfo("Seleccion requerida", "Selecciona un bloque con revision para aplicarla.")
            return
        self._persist_detail_changes()
        entry = self.entries[self.selected_index]
        if self._has_observation_error(entry):
            messagebox.showwarning("Observaciones con error", "La IA devolviÃ³ un error. Revisa las observaciones antes de aplicar la revisiÃ³n.")
            return
        revision = entry.get("revised_text", "").strip()
        if not revision:
            messagebox.showinfo("Revision vacia", "No hay texto revisado que aplicar.")
            return
        entry["translated_text"] = revision
        entry["status"] = "ok"
        self._load_entry_into_detail(entry)
        self._refresh_tree_row(self.selected_index)
        self._emit_progress()

    def _apply_all_revisions(self) -> None:
        self._persist_detail_changes()
        applied = 0
        skipped = 0
        judgement_only = bool(self.judgement_only_var.get())
        for idx, entry in enumerate(self.entries):
            block_type = str(entry.get("type", ""))
            if self._has_observation_error(entry):
                skipped += 1
                continue
            if judgement_only and not block_type.startswith("h"):
                continue
            revised = (entry.get("revised_text") or "").strip()
            if not revised or revised == entry.get("translated_text", ""):
                continue
            entry["translated_text"] = revised
            entry["status"] = "ok"
            entry["revised_text"] = ""
            self._refresh_tree_row(idx)
            applied += 1
        if applied:
            self._emit_progress()
            if self.selected_index is not None and 0 <= self.selected_index < len(self.entries):
                self._load_entry_into_detail(self.entries[self.selected_index])
                self._update_heading_notice()
            messagebox.showinfo("Revisiones aplicadas", f"Se aplicaron {applied} revisiones.")
        elif not skipped:
            messagebox.showinfo("Sin revisiones", "No hay revisiones propuestas para aplicar.")
        if skipped:
            messagebox.showinfo("Observaciones con error", f"Se omitieron {skipped} filas con observaciones de error.")

    def _apply_status_bulk(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Sin selecciÃ³n", "Selecciona una o mÃ¡s filas para actualizar el estado.")
            return
        label = self.bulk_status_combo.get() if hasattr(self, 'bulk_status_combo') else ''
        label_to_code = {STATUS_LABELS['pendiente']: 'pendiente'}
        for code, option_label in STATUS_OPTIONS:
            label_to_code[option_label] = code
        target_code = label_to_code.get(label)
        if not target_code:
            return
        changed = False
        for item_id in selection:
            if item_id not in self.tree_items:
                continue
            idx = self.tree_items.index(item_id)
            if idx < 0 or idx >= len(self.entries):
                continue
            entry = self.entries[idx]
            if entry.get('status') == target_code:
                continue
            entry['status'] = target_code
            self._refresh_tree_row(idx)
            changed = True
        if changed:
            self._emit_progress()
            if self.selected_index is not None:
                self._update_heading_notice()
    def _run_glossary_check(self) -> None:
        if not self.consistency_checker:
            messagebox.showinfo("Glosario", "No hay verificador de consistencia disponible.")
            return
        issues_found = 0
        for idx, entry in enumerate(self.entries):
            translated_text = entry.get("translated_text", "")
            issues = self.consistency_checker.check_block(translated_text, self.glossary)
            if issues:
                entry.setdefault("observaciones", "")
                extra = " \n".join(issues)
                if entry["observaciones"]:
                    entry["observaciones"] += "\n" + extra
                else:
                    entry["observaciones"] = extra
                if entry.get("status") != "mal":
                    entry["status"] = "dudoso"
                self._refresh_tree_row(idx)
                issues_found += 1
        if issues_found:
            messagebox.showwarning("Glosario", f"Se detectaron inconsistencias en {issues_found} bloques.")
        else:
            messagebox.showinfo("Glosario", "No se detectaron conflictos con el glosario.")

    def _on_close(self) -> None:
        self._persist_detail_changes()
        if self._ai_thread and self._ai_thread.is_alive():
            self._cancel_event.set()
        self.result = {"entries": copy.deepcopy(self.entries), "options": self._current_mode_options()}
        self._emit_progress()
        self.destroy()

    def _on_double_click(self, event: tk.Event) -> None:
        item_id = self.tree.identify_row(event.y)
        if not item_id:
            return

        index = self.tree_items.index(item_id)
        if index >= 0:
            entry = self.entries[index]
            entry["status"] = "ok"
            self._refresh_tree_row(index)

    def _show_cost_estimation(self) -> None:
        model = self.review_model_var.get()
        if not model:
            messagebox.showwarning("Sin modelo", "Por favor, selecciona un modelo de IA para la revisiÃ³n.")
            return

        rows_to_process = []
        if self.process_single_row_var.get() and self.selected_index is not None:
            rows_to_process.append(self.entries[self.selected_index])
        else:
            rows_to_process = [entry for entry in self.entries if entry.get("status") != "ok"]

        if not rows_to_process:
            messagebox.showinfo("Nada que estimar", "No hay filas pendientes de revisiÃ³n para procesar.")
            return

        # EstimaciÃ³n de tokens (muy aproximada)
        base_prompt_text = self.engine.translation_qa.review_block_prompt_template(
            original_text="",
            translated_text="",
            glossary={},
            include_observations=bool(self.include_observations_var.get()) and not self.judgement_only_var.get(),
            verdict_only=bool(self.judgement_only_var.get()),
        )
        estimated_input_tokens = 0
        for entry in rows_to_process:
            original = entry.get("original_text", "")
            translated = entry.get("translated_text", "")
            # Se usa una aproximaciÃ³n de 4 caracteres por token
            estimated_input_tokens += (len(base_prompt_text) + len(original) + len(translated)) / 4

        estimated_cost = calculate_cost(
            model=model,
            pricing_data=self.pricing_data,
            input_tokens=round(estimated_input_tokens),
            output_tokens=0,  # No podemos estimar los tokens de salida a priori
        )

        summary_message = (
            f"EstimaciÃ³n de Costo para la RevisiÃ³n\n\n"
            f"Modelo seleccionado: {model}\n"
            f"Filas a procesar: {len(rows_to_process)}\n"
            f"Tokens de entrada estimados: {estimated_input_tokens:,.0f}\n\n"
            f"Costo de entrada estimado: ${estimated_cost:.4f}\n\n"
            "Nota: El costo real dependerÃ¡ de los tokens de salida generados por la IA, que no pueden predecirse."
        )
        messagebox.showinfo("EstimaciÃ³n de Costo (A Priori)", summary_message)

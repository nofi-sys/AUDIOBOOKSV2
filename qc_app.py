from __future__ import annotations
"""Tkinter GUI for the QC application (v5.3‑tc).

* Usa columna **tc** (time‑code) como instante de INICIO de la frase.
* Para reproducir un clip se toma el tc de la fila actual y el tc de la
  fila siguiente (o el final del archivo si no hay siguiente).
* Se mantiene **toda** la funcionalidad original: transcribe, AI‑review,
  edición de celdas, undo/redo, fusión, etc.
"""

import io
import time
import json
import os
import queue
import threading
import traceback
import shutil
import subprocess
import tempfile
from pathlib import Path

import pygame
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

try:
    import vlc  # opcional, para control de velocidad suave
except Exception:
    vlc = None

from utils.gui_errors import show_error

from alignment import build_rows, build_rows_from_words, WARN_WER
from text_utils import read_script, normalize
from rapidfuzz.distance import Levenshtein
from qc_utils import canonical_row, log_correction_metadata
from audacity_session import AudacityLabelSession

from audio_video_editor import build_intervals
# --------------------------------------------------------------------------------------
# utilidades de audio ------------------------------------------------------------------
# --------------------------------------------------------------------------------------

PLAYBACK_PAD = 1.0  # extra cushion to avoid premature cut-offs


def _format_tc(val: str | float) -> str:
    """Return ``val`` formatted as ``HH:MM:SS.d`` with one decimal."""
    try:
        t = float(val)
    except (TypeError, ValueError):
        return str(val)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


def _parse_tc(text: str) -> str:
    """Parse ``text`` formatted with ``_format_tc`` back to seconds string."""
    if ":" in text:
        try:
            h, m, s = text.split(":")
            total = int(h) * 3600 + int(m) * 60 + float(s)
            return str(round(total, 2))
        except Exception:
            pass
    return text


def play_interval(path: str, start: float, end: float | None) -> None:
    """Play ``path`` from ``start`` seconds until ``end`` using pygame."""

    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=start)
    if end is not None:
        dur = max(0.0, end - start)
        ms = int((dur + PLAYBACK_PAD) * 1000)
        tk._default_root.after(ms, pygame.mixer.music.stop)


# --------------------------------------------------------------------------------------
# GUI principal ------------------------------------------------------------------------
# --------------------------------------------------------------------------------------

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QC‑Audiolibro v5.3‑tc")
        self.geometry("1850x760")

        # ------------------------- variables de la interfaz --------------------------
        self.v_ref = tk.StringVar(self)
        self.v_asr = tk.StringVar(self)
        self.v_audio = tk.StringVar(self)
        self.v_json = tk.StringVar(self)
        self.ai_one = tk.BooleanVar(self, value=False)
        self.v_ai_model = tk.StringVar(self, value="gpt-5")

        # --- Estadísticas y Filtros ---
        self.v_stats_total = tk.StringVar(self, value="Total: 0")
        self.v_stats_mal = tk.StringVar(self, value="Filas 'mal': 0")
        self.v_stats_pct = tk.StringVar(self, value="(0.0%)")
        self.v_filter_text = tk.StringVar(self)
        self.v_filter_mal = tk.BooleanVar(self, value=False) # Keep for now, replace UI later
        self.filter_verdicts: dict[str, tk.BooleanVar] = {}
        self.all_rows: list[list] = []  # Almacén persistente de filas

        # Estados internos
        self.q: queue.Queue = queue.Queue()
        self.ok_rows: set[int] = set()
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.merged_rows: dict[str, list[list[str]]] = {}
        self.correction_stats: dict[str, int] = {}
        self.prev_asr: dict[str, str] = {}
        self.asr_confidence: dict[str, float] = {}
        self._stop_reprocess = False

        self.selected_cell: tuple[str, str] | None = None
        self.tree_tag = "sel_cell"
        self.merged_tag = "merged"

        # Repro
        self._clip_item: str | None = None
        self._clip_start = 0.0
        self._clip_end: float | None = None
        self._clip_offset = 0.0  # offset within the current clip

        # Velocidad de reproducción
        self._play_rate: float = 1.0
        self._rate_wall_start: float | None = None
        self._rate_pos_start: float | None = None
        self._rate_job: str | None = None

        # Motor de audio
        self._audio_engine: str = "pygame"
        self._vlc_instance = None
        self._vlc_player = None
        if vlc is not None:
            try:
                self._vlc_instance = vlc.Instance()
                self._audio_engine = "vlc"
            except Exception:
                self._vlc_instance = None
                self._audio_engine = "pygame"
        # Preferir ffplay si está disponible
        self._ffplay_path: str | None = shutil.which("ffplay")
        self._ffplay_proc = None
        if self._ffplay_path:
            self._audio_engine = "ffplay"
        # Soporte de velocidad acelerada solo si hay ffplay
        self._supports_fast: bool = bool(self._ffplay_path)
        self._warned_fast: bool = False

        self.marker_path: Path | None = None
        self.audio_session: AudacityLabelSession | None = None

        self.pos_scale: tk.Scale | None = None
        self.pos_label: ttk.Label | None = None

        self._build_ui()
        self.after(250, self._poll)

        self._prog_win: tk.Toplevel | None = None  # ventana de progreso
        self._prog_bar: ttk.Progressbar | None = None
        self._prog_label: ttk.Label | None = None

    # ---------------------------------------------------------------- build UI ------
    def _build_ui(self) -> None:
        # Main container for top controls
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=3, pady=2)

        # --- Frame for Archivos y Procesamiento ---
        files_frame = ttk.LabelFrame(controls_frame, text="Archivos y Procesamiento")
        files_frame.pack(side="left", fill="y", padx=5, pady=5, anchor="n")

        self._lbl_entry(files_frame, "Guion:", self.v_ref, 0, ("PDF/TXT", "*.pdf;*.txt"))
        self._lbl_entry(files_frame, "TXT ASR:", self.v_asr, 1, ("TXT/CSV", "*.txt;*.csv"))
        self._lbl_entry(files_frame, "Audio:", self.v_audio, 2, ("Media", "*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.mp4"))
        ttk.Label(files_frame, text="JSON:").grid(row=3, column=0, sticky="e", pady=(4,0))
        ttk.Entry(files_frame, textvariable=self.v_json, width=50).grid(row=3, column=1, pady=(4,0))
        ttk.Button(files_frame, text="Abrir JSON…", command=self.load_json).grid(row=3, column=2, pady=(4,0))
        ttk.Button(files_frame, text="Procesar", width=12, command=self.launch).grid(row=0, column=3, rowspan=2, padx=6, ipady=5)

        # --- Frame for AI tools ---
        ai_tools_frame = ttk.LabelFrame(controls_frame, text="Herramientas AI")
        ai_tools_frame.pack(side="left", fill="y", padx=5, pady=5, anchor="n")

        # Row 0
        btn_transcribe = ttk.Button(ai_tools_frame, text="Transcribir", command=self.transcribe)
        btn_transcribe.grid(row=0, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        btn_retranscribe = ttk.Button(ai_tools_frame, text="Re-transcribir mal", command=self.reprocess_bad)
        btn_retranscribe.grid(row=0, column=2, columnspan=2, sticky="ew", padx=2, pady=2)
        btn_pause_re = ttk.Button(ai_tools_frame, text="Pausar", command=self.pause_reprocess)
        btn_pause_re.grid(row=0, column=4, padx=2, pady=2)

        # Row 1
        ttk.Button(ai_tools_frame, text="AI Review", command=self.ai_review).grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ai_models = ["gpt-5", "gpt-5-mini", "gpt-5-nano"]
        self.ai_model_combo = ttk.Combobox(ai_tools_frame, textvariable=self.v_ai_model, values=ai_models, width=12)
        self.ai_model_combo.grid(row=1, column=2, columnspan=2, sticky="ew", padx=2, pady=2)
        self.ai_model_combo.set(ai_models[0])
        ttk.Checkbutton(ai_tools_frame, text="una fila", variable=self.ai_one).grid(row=1, column=4, padx=2, pady=2)

        # Row 2
        ttk.Button(ai_tools_frame, text="Corregir transcript con AI", command=self.ai_correct_row).grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=2)
        ttk.Button(ai_tools_frame, text="Detener análisis", command=self.stop_ai_review).grid(row=2, column=3, columnspan=2, sticky="ew", padx=2, pady=2)

        # Row 3
        ttk.Button(ai_tools_frame, text="Revisión AI Avanzada", command=self.advanced_ai_review).grid(row=3, column=0, columnspan=5, sticky="ew", padx=2, pady=2)

        # --- Frame for Other tools ---
        other_tools_frame = ttk.LabelFrame(controls_frame, text="Otras Herramientas y Reportes")
        other_tools_frame.pack(side="left", fill="y", padx=5, pady=5, anchor="n")

        ttk.Button(other_tools_frame, text="Corregir Desplazamientos", command=self.second_pass_sync).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(other_tools_frame, text="Crear EDL", command=self.create_edl).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(other_tools_frame, text="Informe Corrección", command=self.generate_correction_report).grid(row=2, column=0, sticky="ew", padx=2, pady=2)

        # Frame para estadísticas y filtros
        stats_filter_frame = ttk.LabelFrame(self, text="Estadísticas y Filtros", padding=5)
        stats_filter_frame.pack(fill="x", padx=3, pady=2)

        stats_labels_frame = ttk.Frame(stats_filter_frame)
        stats_labels_frame.pack(side="left", padx=5)

        ttk.Label(stats_labels_frame, textvariable=self.v_stats_total).pack(side="left", padx=4)
        ttk.Label(stats_labels_frame, textvariable=self.v_stats_mal).pack(side="left", padx=4)
        ttk.Label(stats_labels_frame, textvariable=self.v_stats_pct).pack(side="left", padx=4)

        # Controles de filtro
        filter_controls_frame = ttk.Frame(stats_filter_frame)
        filter_controls_frame.pack(side="right", padx=5)
        ttk.Checkbutton(
            filter_controls_frame,
            text="Mostrar solo filas 'mal'",
            variable=self.v_filter_mal,
            command=self._apply_filter,
        ).pack()

        # Tabla principal -----------------------------------------------------------
        self._build_table()
        # Zona inferior con barra y log
        self.bottom_frame = ttk.Frame(self)
        self.bottom_frame.pack(side="bottom", fill="x")
        self._build_player_bar(self.bottom_frame)

        # Menu contextual y atajos -------------------------------------------
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(
            label="Mover ↑", command=lambda: self._move_cell("up")
        )
        self.menu.add_command(
            label="Mover ↓", command=lambda: self._move_cell("down")
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="↑ última palabra", command=lambda: self._move_word("up", "last")
        )
        self.menu.add_command(
            label="↓ última palabra", command=lambda: self._move_word("down", "last")
        )
        self.menu.add_command(
            label="↑ primera palabra", command=lambda: self._move_word("up", "first")
        )
        self.menu.add_command(
            label="↓ primera palabra", command=lambda: self._move_word("down", "first")
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="Fusionar filas seleccionadas", command=self._merge_selected_rows
        )
        self.menu.add_command(label="Desagrupar fila", command=self._unmerge_row)
        self.menu.add_separator()
        self.menu.add_command(label="Alternar ASR", command=self._toggle_asr)

        self.bind_all("<Control-z>", self.undo)
        self.bind_all("<Control-Shift-Z>", self.redo)

        # --- Frame for Stats and Filters ---
        stats_filter_frame = ttk.LabelFrame(self, text="Estadísticas y Filtros", padding=5)
        stats_filter_frame.pack(fill="x", padx=3, pady=2)

        stats_labels_frame = ttk.Frame(stats_filter_frame)
        stats_labels_frame.pack(side="left", padx=5)

        ttk.Label(stats_labels_frame, textvariable=self.v_stats_total).pack(side="left", padx=4)
        ttk.Label(stats_labels_frame, textvariable=self.v_stats_mal).pack(side="left", padx=4)
        ttk.Label(stats_labels_frame, textvariable=self.v_stats_pct).pack(side="left", padx=4)

        # Controles de filtro
        filter_controls_frame = ttk.Frame(stats_filter_frame)
        filter_controls_frame.pack(side="right", padx=5)

        ttk.Label(filter_controls_frame, text="Buscar:").pack(side="left", padx=(10, 2))
        search_entry = ttk.Entry(filter_controls_frame, textvariable=self.v_filter_text, width=30)
        search_entry.pack(side="left", padx=(0, 5))
        search_entry.bind("<Return>", lambda event: self._apply_filter())

        ttk.Button(filter_controls_frame, text="Filtrar por Veredicto", command=self._open_filter_dialog).pack(side="left", padx=5)
        ttk.Button(filter_controls_frame, text="Limpiar Filtros", command=self._clear_filters).pack(side="left", padx=5)


        # Cuadro de log -------------------------------------------------------
        self.log_box = scrolledtext.ScrolledText(self.bottom_frame, height=5, state="disabled")
        self.log_box.pack(fill="x", padx=3, pady=2)

        # Eventos de tabla ----------------------------------------------------
        self.tree.bind("<Button-1>", self._cell_click)
        self.tree.bind("<Button-3>", self._popup_menu)
        self.tree.bind("<Double-1>", self._handle_double)
        self.tree.bind("<<TreeviewSelect>>", self._update_position)

        style = ttk.Style(self)
        style.configure("Treeview", rowheight=45)

        # instantánea inicial para undo --------------------------------------
        self._snapshot()

    def _lbl_entry(self, parent, text, var, row, ft):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="e", padx=(0,4))
        ttk.Entry(parent, textvariable=var, width=50).grid(row=row, column=1)
        ttk.Button(parent, text="…", width=3, command=lambda: self._browse(var, ft)).grid(
            row=row, column=2, padx=(4,0))

    # ---------------------------------------------------------------- table ----------
    def _build_table(self) -> None:
        cols = ("ID", "✓", "OK", "AI", "Score", "WER", "tc", "Original", "ASR")
        widths = (50, 30, 40, 40, 50, 60, 60, 750, 750)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=3, pady=2)

        self.tree = ttk.Treeview(
            table_frame, columns=cols, show="headings", height=27, selectmode="extended")
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            if c not in ("Original", "ASR"):
                self.tree.column(c, width=w, anchor="w", stretch=False)
            else:
                self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")

        sb_y = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        sb_y.pack(side="right", fill="y")
        sb_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        sb_x.pack(side="bottom", fill="x")
        self.tree.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)

        pos_frame = ttk.Frame(table_frame)
        pos_frame.pack(side="right", fill="y", padx=(4, 0))
        self.pos_label = ttk.Label(pos_frame, text="0/0")
        self.pos_label.pack()
        self.pos_scale = tk.Scale(
            pos_frame,
            from_=1,
            to=1,
            orient="vertical",
            showvalue=False,
            command=self._on_pos_change,
        )
        self.pos_scale.pack(fill="y")

        self.tree.tag_configure("sel_cell", background="#d0e0ff")
        self.tree.tag_configure("merged", background="#f5f5f5")
        self.tree.tag_configure("processing", background="#fff2ab")

        # bindings
        self.tree.bind("<Double-1>", self._handle_double)

    # ------------------------------------------------------------- player bar -------
    def _build_player_bar(self, parent: tk.Widget | None = None) -> None:
        if parent is None:
            parent = self
        bar = ttk.Frame(parent)
        bar.pack(side="top", anchor="ne", padx=4, pady=4)
        ttk.Button(bar, text="▶", command=self._play_current_clip).pack(side="left", padx=4)
        ttk.Button(bar, text="←", command=self._prev_bad_row).pack(side="left", padx=4)
        ttk.Button(bar, text="→", command=self._next_bad_row).pack(side="left", padx=4)
        ttk.Button(bar, text="OK", command=self._clip_ok).pack(side="left", padx=4)
        ttk.Button(bar, text="mal", command=self._clip_bad).pack(side="left", padx=4)
        ttk.Button(bar, text="Marcar", command=self.set_marker).pack(side="left", padx=4)
        ttk.Button(bar, text="Guardar punto", command=self.save_bookmark).pack(
            side="left", padx=4)
        ttk.Button(bar, text="Ir al punto", command=self.goto_bookmark).pack(
            side="left", padx=4)

    def _update_stats(self) -> None:
        """Calcula y muestra las estadísticas de filas 'mal' usando self.all_rows."""
        total_rows = len(self.all_rows)
        mal_rows = sum(1 for row in self.all_rows if row[3] == "mal")

        pct = (mal_rows / total_rows * 100) if total_rows > 0 else 0.0
        self.v_stats_total.set(f"Total: {total_rows}")
        self.v_stats_mal.set(f"Filas 'mal': {mal_rows}")
        self.v_stats_pct.set(f"({pct:.1f}%)")

    def _apply_filter(self) -> None:
        """Rellena la tabla aplicando los filtros de texto y veredicto."""
        self.tree.delete(*self.tree.get_children())

        rows_to_display = self.all_rows

        # 1. Filtrar por texto
        search_term = self.v_filter_text.get().lower().strip()
        if search_term:
            rows_to_display = [
                r for r in rows_to_display
                if len(r) >= 2 and (
                    search_term in str(r[-2]).lower() or
                    search_term in str(r[-1]).lower()
                )
            ]

        # 2. Filtrar por veredicto
        active_verdict_filters = {
            verdict for verdict, var in self.filter_verdicts.items() if var.get()
        }
        if active_verdict_filters:
            rows_to_display = [
                r for r in rows_to_display
                if len(r) > 3 and r[3] in active_verdict_filters
            ]

        # Rellenar la tabla con las filas filtradas
        for r in rows_to_display:
            vals = self._row_from_alignment(r)
            self.tree.insert("", tk.END, values=vals)

        self._update_scale_range()
        self._update_stats()

    def _clear_filters(self) -> None:
        """Resetea todos los filtros de texto y veredictos."""
        self.v_filter_text.set("")
        self.v_filter_mal.set(False)
        for verdict_var in self.filter_verdicts.values():
            verdict_var.set(False)
        self._apply_filter()

    def _open_filter_dialog(self) -> None:
        """Abre una ventana para seleccionar los veredictos AI a filtrar."""
        # 1. Obtener todos los veredictos únicos
        all_verdicts = sorted(list(set(
            r[3] for r in self.all_rows if r and len(r) > 3 and r[3]
        )))
        if not all_verdicts:
            messagebox.showinfo("Sin Veredictos", "No hay veredictos 'AI' para filtrar en los datos cargados.")
            return

        # 2. Crear la ventana Toplevel
        dialog = tk.Toplevel(self)
        dialog.title("Filtrar por Veredicto AI")
        dialog.transient(self)
        dialog.grab_set()

        # 3. Crear checkboxes para cada veredicto
        for verdict in all_verdicts:
            if verdict not in self.filter_verdicts:
                self.filter_verdicts[verdict] = tk.BooleanVar(self, value=False)
            var = self.filter_verdicts[verdict]
            cb = ttk.Checkbutton(dialog, text=verdict, variable=var)
            cb.pack(anchor="w", padx=10, pady=2)

        # 4. Botones de aplicar y cerrar
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Aplicar", command=lambda: [self._apply_filter(), dialog.destroy()]).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Cerrar", command=dialog.destroy).pack(side="left", padx=5)

    # ---------------------------------------------------------------------------------
    # navegación de archivos ----------------------------------------------------------
    def _browse(self, var: tk.StringVar, ft: tuple[str, str]):
        p = filedialog.askopenfilename(filetypes=[ft])
        if p:
            var.set(p)

    # ------------------------------------------------------------------ utils
    def clear_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.ok_rows.clear()
        self.all_rows.clear()
        self._update_scale_range()
        self._update_stats()

    def save_json(self) -> None:
        if not self.v_json.get():
            p = filedialog.asksaveasfilename(
                filetypes=[("QC JSON", "*.qc.json;*.json")],
                defaultextension=".json",
            )
            if not p:
                return
            self.v_json.set(p)
        try:
            rows = [list(self.tree.item(i)["values"]) for i in self.tree.get_children()]
            for r in rows:
                idx_tc = max(0, len(r) - 3)
                r[idx_tc] = _parse_tc(str(r[idx_tc]))
            Path(self.v_json.get()).write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8"
            )
            self._log(f"✔ Guardado {self.v_json.get()}")
        except Exception as e:
            show_error("Error", e)

    # ---------------------------------------------------------------------------------
    # mensajes log ---------------------------------------------------------------------
    def _log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    # ---------------------------------------------------------------------------------
    # Transcripción -------------------------------------------------------------------
    def transcribe(self):
        if not self.v_audio.get():
            messagebox.showwarning("Falta info", "Selecciona archivo de audio")
            return
        if not self.v_ref.get():
            messagebox.showwarning(
                "Falta info", "Selecciona guion para guiar la transcripción"
            )
            return
        self._log("⏳ Transcribiendo…")
        self._show_progress("Transcribiendo…", determinate=True)
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self) -> None:
        try:
            from transcriber import transcribe_word_csv

            out = transcribe_word_csv(
                self.v_audio.get(),
                script_path=self.v_ref.get(),
                progress_queue=self.q,
            )
            self.q.put(("SET_ASR", str(out)))
            self.q.put(f"✔ Transcripción guardada en {out}")
        except BaseException as exc:  # noqa: BLE001 - catch SystemExit too
            show_error("Error", exc)

    # ──────────────────────────────────────────────────────────────────────────────
    # Normaliza la lista que llega de build_rows a las 9 columnas de la GUI
    # Orden final: [ID, ✓, OK, AI, Score, WER, tc, Original, ASR]
    # ──────────────────────────────────────────────────────────────────────────────
    def _row_from_alignment(self, r: list) -> list:
        # Devuelve SIEMPRE 9 columnas, rellenando las que falten.
        # Formatea tc como HH:MM:SS.d
        try:
            vals = list(r)

            # Caso base: Fila de 6 columnas del alineador
            # [ID, flag, WER, tc, Original, ASR]
            if len(vals) == 6:
                # -> [ID, ✓, OK, AI, Score, WER, tc, Original, ASR]
                vals.insert(2, "")  # OK
                vals.insert(3, "")  # AI
                vals.insert(4, "")  # Score
                return vals

            # Para filas de JSON, que pueden tener 7, 8 o 9 columnas,
            # las normalizamos a 9 insertando Score si falta.
            if len(vals) in (7, 8):
                vals.insert(4, "") # Score

            # Rellena por si acaso y trunca a 9
            while len(vals) < 9:
                vals.append("")
            vals = vals[:9]

            # Formatea tc (índice 6) y texto (índices 7, 8)
            vals[6] = _format_tc(vals[6])
            vals[7] = str(vals[7])
            vals[8] = str(vals[8])

            return vals
        except Exception:
            # Fallback de emergencia
            padded_r = list(r)
            while len(padded_r) < 9:
                padded_r.append("")
            return padded_r[:9]

    # ───────────────────────────────── ventana de progreso ─────────────────────────
    def _show_progress(self, text: str = "Procesando…", *, determinate: bool = False) -> None:
        """Create modal window with progress bar."""
        if self._prog_win:
            return
        win = tk.Toplevel(self)
        win.title(text)
        win.resizable(False, False)
        win.transient(self)
        win.grab_set()
        ttk.Label(win, text=text, padding=12).pack()
        mode = "determinate" if determinate else "indeterminate"
        pb = ttk.Progressbar(win, mode=mode, length=220, maximum=100)
        pb.pack(padx=12, pady=(0, 6))
        if determinate:
            pb["value"] = 0
        else:
            pb.start(10)
        lbl = ttk.Label(win, text="0%" if determinate else "")
        lbl.pack(pady=(0, 12))
        self._prog_win = win
        self._prog_bar = pb
        self._prog_label = lbl

    def _close_progress(self) -> None:
        """Cierra la ventana de progreso, si existe."""
        if self._prog_win:
            self._prog_win.destroy()
            self._prog_win = None
            self._prog_bar = None
            self._prog_label = None

    def _update_progress(self, pct: int) -> None:
        if self._prog_bar:
            self._prog_bar["value"] = pct
        if self._prog_label:
            self._prog_label["text"] = f"{pct}%"

    # ───────────────────────────────────────────────────────────────────────────────

    # ---------------------------------------------------------------------------------
    # Procesar align ------------------------------------------------------------------
    def launch(self) -> None:
        """Arranca el alineado REF-ASR en un hilo y muestra feedback inmediato."""
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info", "Selecciona guion y TXT ASR.")
            return

        self._log("⏳ Procesando…")
        self._show_progress("Procesando…")

        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------------------------
    # AI review -----------------------------------------------------------------------
    def ai_review(self):
        if not self.v_json.get():
            messagebox.showwarning("Falta info", "Cargar JSON primero")
            return
        if not os.getenv("OPENAI_API_KEY"):
            messagebox.showwarning(
                "Falta OPENAI_API_KEY",
                "Configura la variable OPENAI_API_KEY antes de continuar",
            )
            return

        model = self.v_ai_model.get()
        if self.ai_one.get():
            sel = self.tree.selection()
            if not sel:
                messagebox.showwarning("Falta info", "Selecciona una fila")
                return
            iid = sel[0]
            row_id = self.tree.set(iid, "ID")
            self._log(f"⏳ Revisión AI fila {row_id}…")
            self.q.put(("AI_START_ID", row_id))

            original = self.tree.set(iid, "Original")
            asr = self.tree.set(iid, "ASR")
            if not original.strip() or not asr.strip():
                messagebox.showwarning(
                    "Falta texto",
                    "La fila seleccionada no tiene texto en 'Original' o 'ASR'.")
                return
            self._log("⏳ Revisión AI fila…")

            threading.Thread(
                target=self._ai_review_one_worker,
                args=(row_id, model),
                daemon=True,
            ).start()
        else:
            self._log(
                "⏳ Solicitando revisión AI (esto puede tardar unos segundos)…"
            )
            threading.Thread(
                target=self._ai_review_worker,
                args=(self.all_rows, model),
                daemon=True,
            ).start()

    def ai_correct_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning("Falta info", "Selecciona una fila para corregir")
            return
        iid = sel[0]
        row_id = self.tree.set(iid, "ID")

        row_data = next((r for r in self.all_rows if str(r[0]) == row_id), None)
        if not row_data:
            return
        original, asr = row_data[6], row_data[7]

        if not original.strip() or not asr.strip():
            messagebox.showwarning(
                "Falta texto",
                "La fila seleccionada no tiene texto en 'Original' o 'ASR' para corregir.")
            return

        self._snapshot()
        self._log(f"⏳ Corrección y supervisión AI para fila {row_id}…")
        self.q.put(("AI_START_ID", row_id))

        tags = list(self.tree.item(iid, "tags"))
        if "processing" not in tags:
            tags.append("processing")
            self.tree.item(iid, tags=tuple(tags))


        model = self.v_ai_model.get()
        threading.Thread(
            target=self._ai_correct_worker,
            args=(row_id, original, asr, model),
            daemon=True,
        ).start()

    def _ai_correct_worker(self, row_id: str, original: str, asr: str, model: str) -> None:
        try:
            from ai_review import correct_and_supervise_text
            final_asr, verdict, proposed = correct_and_supervise_text(
                original, asr, model=model
            )
            self.q.put(
                (
                    "AI_CORRECTION_SUPERVISED_ID",
                    (row_id, original, asr, final_asr, verdict, proposed),
                )
            )
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())
        finally:
            self.q.put(("AI_DONE_ID", row_id))

    def stop_ai_review(self):
        try:
            from ai_review import stop_review

            stop_review()
            self._log("⏹ Deteniendo análisis AI…")
        except Exception as exc:
            self._log(str(exc))

    def _ai_review_worker(self, rows_to_review: list[list] | None = None, model: str | None = None) -> None:
        """Run batch AI review updating the GUI incrementally."""
        try:
            import ai_review
            model = model or self.v_ai_model.get()

            if not rows_to_review:
                approved, remaining = ai_review.review_file(self.v_json.get(), model=model)
                self.q.put(("RELOAD", None))
                if ai_review._stop_review:
                    self.q.put("⚠ Revisión detenida")
                else:
                    self.q.put(f"✔ Auto-aprobadas {approved} / Restantes {remaining}")
                return

            def progress(stage: str, idx: int, row: list) -> None:
                row_id = str(row[0])
                if stage == "start":
                    self.q.put(("AI_START_ID", row_id))
                else:
                    self.q.put(("AI_ROW_ID", (row_id, row[3], row[2])))

            try:
                approved, remaining = ai_review.review_file(
                    self.v_json.get(), progress_callback=progress, model=model
                )
            except TypeError:
                # Fallback for older ai_review versions without progress_callback
                approved, remaining = ai_review.review_file(self.v_json.get(), model=model)
                self.q.put(("RELOAD", None))
                if ai_review._stop_review:
                    self.q.put("⚠ Revisión detenida")
                else:
                    self.q.put(
                        f"✔ Auto-aprobadas {approved} / Restantes {remaining}"
                    )
                return

            if ai_review._stop_review:
                self.q.put("⚠ Revisión detenida")
            else:
                self.q.put(
                    f"✔ Auto-aprobadas {approved} / Restantes {remaining}"
                )
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            err = buf.getvalue()
            print(err)
            self.q.put(err)

    def _ai_review_one_worker(self, row_id: str, model: str) -> None:
        try:
            from ai_review import review_row
            row_data = next((r for r in self.all_rows if str(r[0]) == row_id), None)
            if not row_data:
                self.q.put(f"Error: No se encontró la fila con ID {row_id}")
                return

            review_row(row_data, model=model)
            verdict = row_data[3]
            ok = row_data[2]
            self.q.put(("AI_ROW_ID", (row_id, verdict, ok)))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())

    # ------------------------------------------------------------------ reprocess
    def reprocess_bad(self) -> None:
        """Retranscribe rows marked as 'mal' without OK."""
        if not self.v_audio.get():
            messagebox.showwarning("Falta audio", "Selecciona archivo de audio")
            return
        self._stop_reprocess = False
        for iid in self.tree.get_children():
            ai = self.tree.set(iid, "AI").lower()
            ok = self.tree.set(iid, "OK").lower()
            if ai == "mal" and ok != "ok" and iid not in self.prev_asr:
                tags = list(self.tree.item(iid, "tags"))
                if "processing" not in tags:
                    tags.append("processing")
                    self.tree.item(iid, tags=tuple(tags))
                self.update_idletasks()
                self._retranscribe_row(iid)
                tags = list(self.tree.item(iid, "tags"))
                if "processing" in tags:
                    tags.remove("processing")
                    self.tree.item(iid, tags=tuple(tags))
                if self._stop_reprocess:
                    break

    def pause_reprocess(self) -> None:
        """Signal :meth:`reprocess_bad` loop to stop."""
        self._stop_reprocess = True

    def _toggle_asr(self) -> None:
        """Swap between original and re‑transcribed ASR for selected row."""
        sel = self.tree.selection()
        if not sel:
            return
        iid = sel[0]
        current = self.tree.set(iid, "ASR")
        if iid in self.prev_asr:
            self.tree.set(iid, "ASR", self.prev_asr[iid])
            self.prev_asr[iid] = current

    def _retranscribe_row(self, iid: str) -> None:
        """Transcribe a single row with Whisper large model."""
        try:
            from transcriber import transcribe_file
        except Exception as exc:  # pragma: no cover - dependency missing
            self._log(str(exc))
            return

        children = list(self.tree.get_children())
        idx = children.index(iid)

        try:
            row_start = float(_parse_tc(self.tree.set(iid, "tc")))
        except Exception:
            row_start = 0.0

        start = row_start

        # Search next valid tc strictly greater than current row
        end = None
        for j in range(idx + 1, len(children)):
            try:
                next_tc = float(_parse_tc(self.tree.set(children[j], "tc")))
            except Exception:
                continue
            if next_tc > row_start:
                end = next_tc
                break

        clip_path = self._extract_clip(self.v_audio.get(), start, end)
        if not clip_path:
            self._log(f"Error: No se pudo extraer el clip para la fila {self.tree.set(iid, 'ID')}")
            return

        words = self.tree.set(iid, "Original")
        prompt = " ".join(sorted(set(words.split())))
        tmp_prompt = Path(clip_path).with_suffix(".prompt.txt")
        tmp_prompt.write_text(prompt, encoding="utf8")

        out = transcribe_file(
            clip_path,
            model_size="large-v3",
            script_path=str(tmp_prompt),
            show_messagebox=False,
        )
        new_text = Path(out).read_text(encoding="utf8").strip()
        self.prev_asr[iid] = self.tree.set(iid, "ASR")
        self.tree.set(iid, "ASR", new_text)

        prev_asr = (
            self.tree.set(children[idx - 1], "ASR") if idx - 1 >= 0 else ""
        )
        next_asr = (
            self.tree.set(children[idx + 1], "ASR")
            if idx + 1 < len(children)
            else ""
        )
        enumerated = []
        if prev_asr:
            enumerated.append(f"-1: {prev_asr}")
        enumerated.append(f"0: {new_text}")
        if next_asr:
            enumerated.append(f"1: {next_asr}")
        ai_text = "\n".join(enumerated)

        # AI re-review using new transcription
        try:
            from ai_review import review_row, score_row, RETRANS_PROMPT

            model = self.v_ai_model.get()
            row = [0, "", "", 0.0, 0.0, words, ai_text]
            review_row(row, base_prompt=RETRANS_PROMPT, model=model)
            rating = score_row(row, model=model)

            self.tree.set(iid, "Score", rating)
            if float(rating) >= 4:
                self.tree.set(iid, "AI", "ok")
            else:
                self.tree.set(iid, "AI", row[3])
            if row[2]:
                self.tree.set(iid, "OK", row[2])
            self.asr_confidence[iid] = float(rating)
        except Exception as e:
            self._log(f"Error en re-evaluación AI: {e}")
            self.asr_confidence[iid] = 0.0

        try:
            os.remove(clip_path)
            os.remove(tmp_prompt)
        except OSError:
            pass

        self.save_json()

    def _extract_clip(self, path: str, start: float, end: float | None) -> str | None:
        """Return temporary audio clip from start to end using ffmpeg."""
        ffmpeg_path = "ffmpeg"
        if self._ffplay_path:
            ffmpeg_path = str(Path(self._ffplay_path).parent / "ffmpeg")

        tmp = tempfile.NamedTemporaryFile(suffix=Path(path).suffix, delete=False)
        tmp.close()
        cmd = [ffmpeg_path, "-y", "-i", path, "-ss", str(start)]
        if end is not None:
            cmd += ["-to", str(end)]
        cmd += ["-c", "copy", tmp.name]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return tmp.name
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


    # ---------------------------------------------------------------------------------
    # JSON ---------------------------------------------------------------------------
    def load_json(self):
        if not self.v_json.get():
            p = filedialog.askopenfilename(filetypes=[("QC JSON", "*.qc.json;*.json")])
            if not p:
                return
            self.v_json.set(p)

        try:
            self.all_rows = json.loads(Path(self.v_json.get()).read_text(encoding="utf8"))
            self.correction_stats.clear()
            self._apply_filter()  # Rellena la tabla y actualiza stats
            self._snapshot()
            self._load_marker()
            self._load_bookmark_selection()
            self._log(f"✔ Cargado {self.v_json.get()}")
        except Exception as e:
            show_error("Error", e)
    # Reproducción -------------------------------------------------------------------

    def _handle_double(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not item:
            return
        if col == "#3":
            self._toggle_ok(item)
            return
        self._play_clip(item)

    def _play_clip(self, iid: str):
        """Calcula tc inicio ‑ fin y prepara _clip_*"""
        if not self.v_audio.get():
            messagebox.showwarning("Falta audio", "Selecciona archivo de audio")
            return
        try:
            start = float(_parse_tc(self.tree.set(iid, "tc")))
        except ValueError:
            return
        children = list(self.tree.get_children())
        idx = children.index(iid)
        end = None
        for next_iid in children[idx + 1:]:
            try:
                t = float(_parse_tc(self.tree.set(next_iid, "tc")))
            except ValueError:
                continue
            if t > start:
                end = t
                break
        self._clip_item, self._clip_start, self._clip_end = iid, start, end
        self._clip_offset = 0.0
        self._show_text_popup(iid)
        self._play_current_clip()

    def _play_current_clip(self):
        if self._clip_item and self.v_audio.get():
            start = self._clip_start + self._clip_offset
            # Asegurar que no queden reproducciones previas activas
            self._stop_all_audio()
            # Si se pidió velocidad >1x y no soportamos fast, degradar a 1x con aviso
            if abs(self._play_rate - 1.0) > 1e-6 and not self._supports_fast:
                self._play_rate = 1.0
                self._warn_fast_unavailable()
            if (getattr(self, "_ffplay_path", None)
                    and getattr(self, "_audio_engine", "") == "ffplay"):
                self._play_ffplay(start, self._clip_end)
            else:
                # Sin ffplay: siempre 1x, sin hack de saltos
                self._cancel_rate_job()
                play_interval(self.v_audio.get(), start, self._clip_end)

    def _seek_clip(self, offset: float) -> None:
        """Move playback head to ``offset`` seconds within current clip."""
        self._clip_offset = max(0.0, offset)
        # Reinicia reproducción limpiamente con el nuevo offset
        self._stop_all_audio()
        self._play_current_clip()

    # ----------------------------- motor FFPLAY -------------------------------------
    def _play_ffplay(self, start: float, end: float | None) -> None:
        # Matar cualquier reproducción previa (ffplay/pygame/vlc)
        self._stop_all_audio()
        # Construir comando
        rate = max(0.5, min(2.0, float(self._play_rate)))
        cmd = [
            self._ffplay_path, "-hide_banner", "-loglevel", "error",
            "-nodisp", "-autoexit", "-vn"
        ]
        if start and start > 0:
            cmd += ["-ss", f"{start:.3f}"]
        if end is not None and end > start:
            dur = max(0.0, end - start + PLAYBACK_PAD)
            cmd += ["-t", f"{dur:.3f}"]
        if abs(rate - 1.0) > 1e-6:
            cmd += ["-af", f"atempo={rate:.3f}"]
        cmd += [self.v_audio.get()]
        try:
            self._ffplay_proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._rate_wall_start = time.monotonic()
            self._rate_pos_start = start
        except Exception:
            # Fallback a pygame
            self._audio_engine = "pygame"
            if abs(self._play_rate - 1.0) < 1e-6:
                play_interval(self.v_audio.get(), start, end)
            else:
                self._start_rate_playback()

    # ----------------------------- motor VLC ---------------------------------------
    def _play_vlc(self, start: float, end: float | None) -> None:
        try:
            # detener reproducción previa
            if self._vlc_player is not None:
                try:
                    self._vlc_player.stop()
                except Exception:
                    pass
            media = self._vlc_instance.media_new_path(self.v_audio.get())
            opts = [f":start-time={start}"]
            if end is not None:
                opts.append(f":stop-time={end}")
            try:
                media.add_options(*opts)
            except Exception:
                for o in opts:
                    try:
                        media.add_option(o)
                    except Exception:
                        pass
            player = self._vlc_instance.media_player_new()
            player.set_media(media)
            player.play()
            try:
                player.set_rate(float(self._play_rate))
            except Exception:
                pass
            self._vlc_player = player
        except Exception:
            # Fallback a pygame
            self._audio_engine = "pygame"
            if abs(self._play_rate - 1.0) > 1e-6:
                self._start_rate_playback()
            else:
                play_interval(self.v_audio.get(), start, end)

    def _cancel_rate_job(self) -> None:
        if getattr(self, "_rate_job", None):
            try:
                self.after_cancel(self._rate_job)
            except Exception:
                pass
            self._rate_job = None

    def _start_rate_playback(self) -> None:
        # Punto de inicio deseado
        self._stop_all_audio()
        pos = self._clip_start + self._clip_offset
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(self.v_audio.get())
            pygame.mixer.music.play(start=pos)
        except Exception:
            # Fallback
            play_interval(self.v_audio.get(), pos, self._clip_end)
        # Anclas para calcular posición deseada según la pared de tiempo
        self._rate_wall_start = time.monotonic()
        self._rate_pos_start = pos
        self._schedule_rate_tick()

    def _schedule_rate_tick(self) -> None:
        self._cancel_rate_job()
        self._rate_job = self.after(150, self._rate_tick)

    def _rate_tick(self) -> None:
        # Si no hay velocidad acelerada, no continuar
        if abs(self._play_rate - 1.0) < 1e-6 or not self._supports_fast:
            self._cancel_rate_job()
            return
        if self._rate_wall_start is None or self._rate_pos_start is None:
            return
        # Calcular posición deseada
        now = time.monotonic()
        elapsed = now - self._rate_wall_start
        rate = max(self._play_rate, 0.1)
        desired = self._rate_pos_start + elapsed * rate
        clip_end = self._clip_end if self._clip_end is not None else float("inf")
        if desired >= clip_end + PLAYBACK_PAD:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
            self._cancel_rate_job()
            return

        # Posición real
        actual = desired
        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pos_ms = pygame.mixer.music.get_pos()
                if pos_ms >= 0:
                    actual = self._clip_start + self._clip_offset + pos_ms / 1000.0
        except Exception:
            pass

        # Si hay desviación, re-lanzar al punto deseado
        if abs(desired - actual) > 0.3:
            try:
                pygame.mixer.music.play(start=desired)
            except Exception:
                pass

        self._schedule_rate_tick()

    def _warn_fast_unavailable(self) -> None:
        if not self._warned_fast:
            self._warned_fast = True
            self._log("Velocidad >1x requiere FFmpeg/ffplay en el PATH. Reproduciendo a 1x.")
            try:
                messagebox.showinfo(
                    "Velocidad no disponible",
                    "Para reproducir a 1.5x o 2x, instala FFmpeg y asegúrate de "
                    "tener ffplay en el PATH.\n\nSe usa 1x por ahora.",
                )
            except Exception:
                pass

    def _stop_all_audio(self) -> None:
        """Detiene cualquier motor de audio activo y cancela timers."""
        self._cancel_rate_job()
        # ffplay
        try:
            if getattr(self, "_ffplay_proc", None) is not None:
                self._ffplay_proc.kill()
                self._ffplay_proc = None
        except Exception:
            pass
        # vlc
        try:
            if getattr(self, "_vlc_player", None) is not None:
                self._vlc_player.stop()
                self._vlc_player = None
        except Exception:
            pass
        # pygame
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass

    def _show_text_popup(self, iid: str) -> None:
        """Display full text for Original and ASR with timeline controls."""
        original = self.tree.set(iid, "Original")
        asr = self.tree.set(iid, "ASR")

        win = tk.Toplevel(self)
        win.title("Texto completo")

        text_frame = ttk.Frame(win)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        st_orig = scrolledtext.ScrolledText(text_frame, width=50, height=10, wrap="word")
        st_asr = scrolledtext.ScrolledText(text_frame, width=50, height=10, wrap="word")
        st_orig.insert("1.0", original)
        st_asr.insert("1.0", asr)
        st_orig.configure(state="disabled")
        st_asr.configure(state="disabled")
        st_orig.pack(side="left", fill="both", expand=True, padx=(0, 5))
        st_asr.pack(side="left", fill="both", expand=True, padx=(5, 0))

        dur = (self._clip_end or self._clip_start) - self._clip_start
        scale = ttk.Scale(win, from_=0.0, to=max(dur, 0.0), orient="horizontal", length=400)
        scale.set(self._clip_offset)
        scale.pack(fill="x", padx=10, pady=(0, 10))

        def _seek(event=None):
            self._seek_clip(float(scale.get()))

        scale.bind("<ButtonRelease-1>", _seek)

        def _update():
            if getattr(self, "_audio_engine", "") == "ffplay":
                if self._rate_wall_start is not None and self._rate_pos_start is not None:
                    wall_elapsed = time.monotonic() - self._rate_wall_start
                    rate = max(self._play_rate, 0.1)
                    rel = (self._rate_pos_start - self._clip_start) + wall_elapsed * rate
                    rel = max(0.0, min(max(dur, 0.0), rel))
                    scale.set(rel)
            elif (getattr(self, "_audio_engine", "") == "vlc"
                    and getattr(self, "_vlc_player", None) is not None):
                try:
                    cur_ms = self._vlc_player.get_time()
                    if cur_ms >= 0:
                        rel = (cur_ms / 1000.0) - self._clip_start
                        rel = max(0.0, min(max(dur, 0.0), rel))
                        scale.set(rel)
                except Exception:
                    pass
            else:
                if abs(self._play_rate - 1.0) < 1e-6:
                    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                        pos = pygame.mixer.music.get_pos()
                        if pos >= 0:
                            scale.set(self._clip_offset + pos / 1000)
                else:
                    if self._rate_wall_start is not None and self._rate_pos_start is not None:
                        wall_elapsed = time.monotonic() - self._rate_wall_start
                        rel = (self._rate_pos_start - self._clip_start)
                        rel += wall_elapsed * self._play_rate
                        rel = max(0.0, min(max(dur, 0.0), rel))
                        scale.set(rel)
            win.after(100, _update)

        _update()

        # Controles de velocidad
        rate_frame = ttk.Frame(win)
        rate_frame.pack(pady=(0, 6))

        def _set_rate(r: float):
            # Si no hay soporte fast, degradar a 1x y avisar
            if abs(r - 1.0) > 1e-6 and not self._supports_fast:
                self._play_rate = 1.0
                self._warn_fast_unavailable()
                return
            self._play_rate = r
            if getattr(self, "_audio_engine", "") == "ffplay":
                # Reiniciar en el punto actual con nueva velocidad
                self._clip_offset = float(scale.get())
                self._stop_all_audio()
                self._play_current_clip()
            elif (getattr(self, "_audio_engine", "") == "vlc"
                    and getattr(self, "_vlc_player", None) is not None):
                try:
                    self._vlc_player.set_rate(float(r))
                except Exception:
                    pass
            else:
                # Sin soporte de fast, mantener 1x
                self._play_rate = 1.0
                self._warn_fast_unavailable()

        ttk.Label(rate_frame, text="Velocidad:").pack(side="left", padx=(0, 6))
        ttk.Button(rate_frame, text="1x", command=lambda: _set_rate(1.0)).pack(
            side="left", padx=2)
        ttk.Button(rate_frame, text="1.5x", command=lambda: _set_rate(1.5)).pack(
            side="left", padx=2)
        ttk.Button(rate_frame, text="2x", command=lambda: _set_rate(2.0)).pack(
            side="left", padx=2)

        btns = ttk.Frame(win)
        btns.pack(pady=(0, 10))
        ttk.Button(btns, text="OK", command=lambda: self._popup_mark_ok(iid, win)).pack(
            side="left", padx=4)
        ttk.Button(
            btns, text="Marcar",
            command=lambda: self.add_audacity_marker(
                self._clip_start + float(scale.get()))
        ).pack(side="left", padx=4)

        def _close():
            self._stop_all_audio()
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", _close)
        ttk.Button(btns, text="Cerrar", command=_close).pack(side="left", padx=4)

    def _toggle_ok(self, item: str) -> None:
        current = self.tree.set(item, "OK")
        new_val = "" if current == "OK" else "OK"
        self.tree.set(item, "OK", new_val)
        try:
            line_id = int(self.tree.set(item, "ID"))
        except Exception:
            return
        if new_val:
            self.ok_rows.add(line_id)
        else:
            self.ok_rows.discard(line_id)
        self.save_json()

    def _popup_mark_ok(self, iid: str, win: tk.Toplevel) -> None:
        self.tree.set(iid, "OK", "OK")
        try:
            line_id = int(self.tree.set(iid, "ID"))
            self.ok_rows.add(line_id)
        except Exception:
            pass
        win.destroy()
        self.save_json()

    def _update_metrics(self, iid: str, *, save: bool = True) -> None:
        """Recalculate flag and WER after modifying text."""
        original = self.tree.set(iid, "Original")
        asr = self.tree.set(iid, "ASR")
        ref_t = normalize(original, strip_punct=False).split()
        hyp_t = normalize(asr, strip_punct=False).split()
        if hyp_t:
            wer_val = Levenshtein.normalized_distance(ref_t, hyp_t)
            base_ref = [t.strip(".,;!") for t in ref_t]
            base_hyp = [t.strip(".,;!") for t in hyp_t]
            base_wer = Levenshtein.normalized_distance(base_ref, base_hyp)
        else:
            wer_val = 1.0
            base_wer = 1.0
        if base_wer <= 0.05:
            flag = "✅"
        else:
            thr = 0.20 if len(ref_t) < 5 else WARN_WER
            flag = "✅" if wer_val <= thr else ("⚠️" if wer_val <= 0.20 else "❌")
        self.tree.set(iid, "✓", flag)
        self.tree.set(iid, "WER", f"{wer_val*100:.1f}")
        if save:
            self.save_json()

    def _recompute_tc(self) -> None:
        """Normalise time codes while keeping large jumps visible for review."""
        last: float | None = None
        drift: list[str] = []
        for iid in self.tree.get_children():
            raw = self.tree.set(iid, "tc")
            try:
                tc = float(_parse_tc(raw))
            except (TypeError, ValueError):
                tc = last if last is not None else 0.0
            if last is None:
                last = tc
            elif tc < last - 0.35:
                if last - tc <= 1.0:
                    tc = last
                else:
                    drift.append(iid)
                    last = tc
            else:
                last = tc
            self.tree.set(iid, "tc", _format_tc(tc))
        if drift:
            self._log(f"[TC] {len(drift)} filas mantienen saltos de tiempo; revisar.")

    def _clip_ok(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "ok")
            self._update_row_in_all_rows(self._clip_item, "AI", "ok")
            self._apply_filter()
        self._hide_clip()

    def _clip_bad(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "mal")
            self._update_row_in_all_rows(self._clip_item, "AI", "mal")
            self._apply_filter()
        self._hide_clip()

    def _hide_clip(self) -> None:
        self._stop_all_audio()
        self._clip_item = None

    def _next_bad_row(self):
        """Jump to the next row marked ``"mal"`` and not already ``"OK"``."""
        children = list(self.tree.get_children())
        if not children:
            return
        start = 0
        if self._clip_item and self._clip_item in children:
            start = children.index(self._clip_item) + 1
        sequence = children[start:] + children[:start]
        for iid in sequence:
            if (
                self.tree.set(iid, "AI") == "mal"
                and self.tree.set(iid, "OK") != "OK"
            ):
                self.tree.see(iid)
                self._play_clip(iid)
                return

    def _prev_bad_row(self):
        """Jump to the previous row marked ``"mal"`` and not ``"OK"``."""
        children = list(self.tree.get_children())
        if not children:
            return
        start = len(children) - 1
        if self._clip_item and self._clip_item in children:
            start = children.index(self._clip_item) - 1
        first_part = list(reversed(children[: start + 1]))
        second_part = list(reversed(children[start + 1:]))
        for iid in first_part + second_part:
            if (
                self.tree.set(iid, "AI") == "mal"
                and self.tree.set(iid, "OK") != "OK"
            ):
                self.tree.see(iid)
                self._play_clip(iid)
                return

    # --------------------------------------------------------------- cell utils
    def _cell_click(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        for iid in self.tree.get_children():
            tags = list(self.tree.item(iid, "tags"))
            if self.tree_tag in tags:
                tags.remove(self.tree_tag)
                self.tree.item(iid, tags=tuple(tags))
        if col not in ("#7", "#8") or not item:
            self.selected_cell = None
            return
        tags = list(self.tree.item(item, "tags"))
        if self.tree_tag not in tags:
            tags.append(self.tree_tag)
            self.tree.item(item, tags=tuple(tags))
        self.selected_cell = (item, col)

    def _popup_menu(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        self._cell_click(event)
        sel = self.tree.selection()
        self.menu.entryconfig(
            "Fusionar filas seleccionadas",
            state="normal" if len(sel) > 1 else "disabled",
        )
        if item:
            self.menu.tk_popup(event.x_root, event.y_root)

    def _move_cell(self, direction: str) -> None:
        if not self.selected_cell:
            return
        item, col_id = self.selected_cell
        children = self.tree.get_children()
        idx = children.index(item)
        dst_idx = idx - 1 if direction == "up" else idx + 1
        if dst_idx < 0 or dst_idx >= len(children):
            return
        dst_item = children[dst_idx]
        col = "Original" if col_id == "#7" else "ASR"
        src_text = self.tree.set(item, col)
        if not src_text:
            return
        dst_text = self.tree.set(dst_item, col)
        fused = (
            (dst_text.rstrip().rstrip(".") + " " + src_text).strip()
            if direction == "up"
            else (src_text.rstrip(".") + " " + dst_text).strip()
        )
        self._snapshot()
        self.tree.set(dst_item, col, fused)
        self.tree.set(item, col, "")
        other_col = "ASR" if col == "Original" else "Original"
        if not self.tree.set(item, col) and not self.tree.set(item, other_col):
            self.tree.delete(item)
        self.selected_cell = None
        for iid in self.tree.get_children():
            tags = list(self.tree.item(iid, "tags"))
            if self.tree_tag in tags:
                tags.remove(self.tree_tag)
                self.tree.item(iid, tags=tuple(tags))
        self._recompute_tc()
        self._update_metrics(dst_item)
        self._update_metrics(item)
        self.save_json()

    def _move_word(self, direction: str, which: str) -> None:
        if not self.selected_cell:
            return
        item, col_id = self.selected_cell
        children = self.tree.get_children()
        idx = children.index(item)
        dst_idx = idx - 1 if direction == "up" else idx + 1
        if dst_idx < 0 or dst_idx >= len(children):
            return
        dst_item = children[dst_idx]
        col = "Original" if col_id == "#7" else "ASR"
        src_text = self.tree.set(item, col).strip()
        if not src_text:
            return
        words = src_text.split()
        if not words:
            return
        word = words.pop(0 if which == "first" else -1)
        self._snapshot()
        self.tree.set(item, col, " ".join(words))
        dst_text = self.tree.set(dst_item, col).strip()
        fused = (
            (dst_text + " " + word).strip()
            if direction == "up"
            else (word + " " + dst_text).strip()
        )
        self.tree.set(dst_item, col, fused)
        other_col = "ASR" if col == "Original" else "Original"
        if not self.tree.set(item, col) and not self.tree.set(item, other_col):
            self.tree.delete(item)
        self.selected_cell = None
        for iid in self.tree.get_children():
            tags = list(self.tree.item(iid, "tags"))
            if self.tree_tag in tags:
                tags.remove(self.tree_tag)
                self.tree.item(iid, tags=tuple(tags))
        self._recompute_tc()
        self._update_metrics(dst_item)
        self._update_metrics(item)
        self.save_json()

    # ------------------------------------------------------------- position bar
    def _on_pos_change(self, value: str) -> None:
        if not self.tree.get_children():
            return
        idx = int(float(value)) - 1
        children = self.tree.get_children()
        idx = max(0, min(len(children) - 1, idx))
        iid = children[idx]
        self.tree.see(iid)
        self.tree.selection_set(iid)
        self._update_position()

    def _update_position(self, event: tk.Event | None = None) -> None:
        if not self.tree.get_children():
            self.pos_label.config(text="0/0")
            return
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0]) + 1
        total = len(self.tree.get_children())
        self.pos_label.config(text=f"{idx}/{total}")
        if self.pos_scale:
            self.pos_scale.configure(to=total)
        self.pos_scale.set(idx)

    def _update_scale_range(self) -> None:
        total = len(self.tree.get_children())
        if self.pos_scale:
            self.pos_scale.configure(to=max(total, 1))
        if self.pos_label:
            sel = self.tree.selection()
            idx = self.tree.index(sel[0]) + 1 if sel else 0
            self.pos_label.config(text=f"{idx}/{total}")

    # ------------------------------------------------------------- marker utils
    def set_marker(self) -> None:
        if not self.v_json.get():
            return
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        self.marker_path = Path(self.v_json.get()).with_suffix(".marker")
        self.marker_path.write_text(str(idx), encoding="utf8")
        self._log(f"✔ Marcador en fila {idx + 1}")

    def _get_audacity_session(self) -> AudacityLabelSession | None:
        if not self.v_audio.get():
            return None
        if self.audio_session and self.audio_session.audio_path == Path(self.v_audio.get()):
            return self.audio_session
        try:
            self.audio_session = AudacityLabelSession(self.v_audio.get())
        except Exception as exc:
            self._log(str(exc))
            self.audio_session = None
        return self.audio_session

    def add_audacity_marker(self, time_sec: float) -> None:
        session = self._get_audacity_session()
        if not session:
            return
        session.add_marker(time_sec)
        self._log(f"✔ Marker Audacity {time_sec:.2f}s")

    # ------------------------------------------------------------- bookmark utils
    def _bookmark_path(self) -> Path | None:
        if not self.v_json.get():
            return None
        return Path(self.v_json.get()).with_suffix(".bookmark.json")

    def save_bookmark(self, abs_time: float | None = None) -> None:
        """Guarda un 'punto' para retomar. Si ``abs_time`` no se pasa, usa
        el inicio de la fila seleccionada. Persiste en un archivo junto al JSON.
        """
        if not self.v_json.get():
            messagebox.showwarning("Sin JSON", "Carga un JSON primero")
            return
        p = self._bookmark_path()
        if p is None:
            return
        sel = self.tree.selection()
        if sel:
            iid = sel[0]
        else:
            children = self.tree.get_children()
            iid = self._clip_item or (children[0] if children else None)
        if iid is None:
            messagebox.showwarning("Sin selección", "Selecciona una fila para guardar el punto")
            return
        idx = self.tree.index(iid)
        if abs_time is None:
            # Intentar capturar la posición actual de reproducción si hay clip activo
            if self._clip_item:
                try:
                    if (getattr(self, "_audio_engine", "") == "ffplay"
                            and self._rate_wall_start is not None
                            and self._rate_pos_start is not None):
                        rate = max(self._play_rate, 0.1)
                        abs_time = self._rate_pos_start + (
                            time.monotonic() - self._rate_wall_start) * rate
                    elif pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                        pos = pygame.mixer.music.get_pos()
                        if pos >= 0:
                            abs_time = self._clip_start + self._clip_offset + pos / 1000.0
                    elif (getattr(self, "_audio_engine", "") == "vlc"
                            and getattr(self, "_vlc_player", None) is not None):
                        cur_ms = self._vlc_player.get_time()
                        if cur_ms >= 0:
                            abs_time = cur_ms / 1000.0
                except Exception:
                    pass
            if abs_time is None:
                try:
                    abs_time = float(_parse_tc(self.tree.set(iid, "tc")))
                except Exception:
                    abs_time = 0.0
        data = {
            "row_index": int(idx),
            "time": float(abs_time),
            "rate": float(self._play_rate),
        }
        try:
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf8")
            self._log(f"Punto guardado (fila {idx + 1}, t={abs_time:.2f}s)")
        except Exception as exc:
            show_error("Error", exc)

    def goto_bookmark(self) -> None:
        p = self._bookmark_path()
        if p is None or not p.exists():
            messagebox.showinfo("Punto", "No hay punto guardado")
            return
        try:
            data = json.loads(p.read_text(encoding="utf8"))
            idx = int(data.get("row_index", 0))
            t = float(data.get("time", 0.0))
            rate = float(data.get("rate", self._play_rate))
        except Exception as exc:
            show_error("Error", exc)
            return
        children = list(self.tree.get_children())
        if not children:
            return
        idx = max(0, min(len(children) - 1, idx))
        iid = children[idx]
        self.tree.selection_set(iid)
        self.tree.see(iid)
        self._update_position()
        self._play_rate = rate
        if self.v_audio.get():
            try:
                row_tc = float(_parse_tc(self.tree.set(iid, "tc")))
            except Exception:
                row_tc = t
            self._clip_item = iid
            self._clip_start = row_tc
            self._clip_offset = max(0.0, t - row_tc)
            end = None
            for next_iid in children[idx + 1:]:
                try:
                    tt = float(_parse_tc(self.tree.set(next_iid, "tc")))
                except Exception:
                    continue
                if tt > row_tc:
                    end = tt
                    break
            self._clip_end = end
            self._show_text_popup(iid)
            self._play_current_clip()

    def _load_marker(self) -> None:
        if not self.v_json.get():
            return
        self.marker_path = Path(self.v_json.get()).with_suffix(".marker")
        if self.marker_path.exists():
            try:
                idx = int(self.marker_path.read_text())
                children = self.tree.get_children()
                if 0 <= idx < len(children):
                    iid = children[idx]
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
                    self._update_position()
            except Exception:
                pass

    def _load_bookmark_selection(self) -> None:
        p = self._bookmark_path()
        if p is None or not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf8"))
            idx = int(data.get("row_index", 0))
        except Exception:
            return
        children = self.tree.get_children()
        if not children:
            return
        if 0 <= idx < len(children):
            iid = children[idx]
            self.tree.selection_set(iid)
            self.tree.see(iid)
            self._update_position()

    def _merge_selected_rows(self) -> None:
        sel = list(self.tree.selection())
        if len(sel) < 2:
            return
        sel.sort(key=lambda iid: self.tree.index(iid))
        first = sel[0]
        originals: list[str] = []
        asrs: list[str] = []

        for iid in sel:
            originals.append(self.tree.set(iid, "Original").strip())
            asrs.append(self.tree.set(iid, "ASR").strip())

        fuse = lambda parts: " ".join(p.rstrip(".,;") for p in parts if p)

        self._snapshot()
        self.merged_rows[first] = [list(self.tree.item(i)["values"]) for i in sel]

        self.tree.set(first, "Original", fuse(originals))
        self.tree.set(first, "ASR", fuse(asrs))
        self.tree.set(first, "WER", "")

        for iid in sel[1:]:
            self.tree.delete(iid)
            self.merged_rows.pop(iid, None)

        tags = list(self.tree.item(first, "tags"))
        if self.merged_tag not in tags:
            tags.append(self.merged_tag)
        self.tree.item(first, tags=tuple(tags))

        start_idx = self.tree.index(first)
        for new_id, iid in enumerate(self.tree.get_children()[start_idx:], start_idx):
            self.tree.set(iid, "ID", new_id)
        self._update_scale_range()
        self._recompute_tc()
        self._update_metrics(first)
        self.save_json()

    def _unmerge_row(self) -> None:
        sel = list(self.tree.selection())
        if len(sel) != 1:
            return
        item = sel[0]
        if item not in self.merged_rows:
            return
        rows = self.merged_rows.pop(item)
        idx = self.tree.index(item)
        self._snapshot()
        self.tree.delete(item)
        for r in rows:
            iid = self.tree.insert("", idx, values=r)
            idx += 1
            self._update_metrics(iid)
        for new_id, iid in enumerate(self.tree.get_children()):
            self.tree.set(iid, "ID", new_id)
        self._update_scale_range()
        self._recompute_tc()
        self.save_json()

    # ---------------------------------------------------------------------------------
    # hilo worker (alinear) -----------------------------------------------------------

    def _update_row_in_all_rows(self, iid: str, col_name: str, new_value: str) -> None:
        """Actualiza un valor en la lista `self.all_rows` basado en el iid de la tabla."""
        try:
            row_id = self.tree.set(iid, "ID")
            col_idx_map = {name: i for i, name in enumerate(self.tree["columns"])}
            col_idx = col_idx_map[col_name]

            for row in self.all_rows:
                if str(row[0]) == str(row_id):
                    row[col_idx] = new_value
                    break
        except (KeyError, ValueError, IndexError):
            # No es crítico si falla, el estado se resincronizará al guardar/cargar.
            pass

    def _worker(self):
        def debug(msg: str) -> None:
            print(msg)
            self.q.put(msg)

        try:
            from alignment import set_debug_logger
            set_debug_logger(debug)

            self.q.put("→ Leyendo guion…")
            debug(f"DEBUG: Leyendo guion {self.v_ref.get()}")
            ref = read_script(self.v_ref.get())

            debug(f"DEBUG: Guion cargado ({len(ref)} chars)")
            self.q.put(f"→ DEBUG: Primeros 200 chars: {ref[:200]}")

            asr_path = Path(self.v_asr.get())
            debug(f"DEBUG: Leyendo ASR {asr_path}")
            if asr_path.suffix.lower() == ".csv":
                from utils.resync_python_v2 import load_words_csv
                csv_words, csv_tcs = load_words_csv(asr_path)
                hyp = " ".join(csv_words)
                use_csv = True
                debug("DEBUG: ASR en formato CSV cargado")
            else:
                hyp = asr_path.read_text(
                    encoding="utf8", errors="ignore"
                )
                use_csv = False
                debug("DEBUG: ASR en texto cargado")
            self.q.put("→ TXT externo cargado")

            # ═══ DEBUG TEMPORAL ═══
            self.q.put(f"→ DEBUG: Longitud ASR: {len(hyp)} caracteres")
            self.q.put(f"→ DEBUG: Primeros 200 chars ASR: {hyp[:200]}")
            debug("DEBUG: ASR cargado")
            # ═══ FIN DEBUG ═══

            self.q.put("→ Alineando…")
            debug("DEBUG: Iniciando alineacion")

            if use_csv:
                # ASR llegó como CSV de palabras: usar alineado palabra-a-palabra
                rows = build_rows_from_words(ref, csv_words, csv_tcs)
            else:
                # ASR llegó como texto: alinear por tokens…
                rows = build_rows(ref, hyp)
                # …pero si hay CSV al lado (o lo seleccionás), preferir palabra-a-palabra
                csv_path: Path | None = None
                audio_val = self.v_audio.get()
                if audio_val:
                    candidate = Path(audio_val).with_suffix(".words.csv")
                    if candidate.exists():
                        csv_path = candidate
                if csv_path is None:
                    from tkinter import filedialog
                    p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*")])
                    if p:
                        csv_path = Path(p)
                if csv_path and csv_path.exists():
                    try:
                        from utils.resync_python_v2 import load_words_csv
                        csv_words, csv_tcs = load_words_csv(csv_path)
                        rows = build_rows_from_words(ref, csv_words, csv_tcs)
                        debug(f"DEBUG: CSV usado {csv_path}")
                    except Exception as exc:
                        self.q.put(f"CSV fallback error: {exc}")

            rows = [canonical_row(r) for r in rows]

            # ═══ DEBUG TEMPORAL ═══
            self.q.put(f"→ DEBUG: Se generaron {len(rows)} filas")
            if rows:
                self.q.put(f"→ DEBUG: Primera fila: {rows[0]}")
            debug("DEBUG: alineacion completada")
            # ═══ FIN DEBUG ═══

            out = Path(self.v_asr.get()).with_suffix(".qc.json")
            if out.exists():
                try:
                    old = json.loads(out.read_text(encoding="utf8"))
                    from qc_utils import merge_qc_metadata

                    rows = merge_qc_metadata(old, rows)
                    rows = [canonical_row(r) for r in rows]
                except Exception:
                    pass
            out.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8"
            )

            debug(f"DEBUG: JSON guardado en {out}")

            self.all_rows = rows
            self.q.put(("ROWS_READY", None))
            self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            err = buf.getvalue()
            debug(f"DEBUG: error en worker:\n{err}")
            self.q.put(("ERROR", "Error en el procesamiento", err))
        finally:
            self.q.put(("CLOSE_PROGRESS", None))
            from alignment import set_debug_logger
            set_debug_logger(print)

    # ---------------------------------------------------------------------------------
    # cola de mensajes Tk -------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()

                if isinstance(msg, tuple) and msg[0] == "ROWS_READY":
                    self._apply_filter()
                    self._close_progress()
                    self._snapshot()

                elif isinstance(msg, tuple) and msg[0] == "SET_ASR":
                    self.v_asr.set(msg[1])
                    self._close_progress()

                elif isinstance(msg, tuple) and msg[0] == "PROGRESS":
                    pct = int(msg[1])
                    # soporta tuplas ("PROGRESS", pct) o ("PROGRESS", pct, eta)
                    self._update_progress(pct)
                    if len(msg) >= 3 and self._prog_label:
                        def _fmt_eta(secs: float) -> str:
                            try:
                                secs = float(secs)
                            except Exception:
                                return ""
                            m = int(secs // 60)
                            s = int(secs % 60)
                            return f" — ETA {m}:{s:02d}"
                        self._prog_label["text"] = f"{pct}%" + _fmt_eta(msg[2])

                elif isinstance(msg, tuple) and msg[0] == "RELOAD":
                    self.load_json()

                elif isinstance(msg, tuple) and msg[0] == "AI_START":
                    iid = msg[1]
                    tags = list(self.tree.item(iid, "tags"))
                    if "processing" not in tags:
                        tags.append("processing")
                        self.tree.item(iid, tags=tuple(tags))
                    self.tree.see(iid)
                    self.tree.selection_set(iid)

                elif isinstance(msg, tuple) and msg[0] == "AI_ROW_ID":
                    row_id, verdict, ok = msg[1]

                    col_map = {name: i for i, name in enumerate(self.tree["columns"])}
                    ai_idx = col_map["AI"]
                    ok_idx = col_map["OK"]

                    for row in self.all_rows:
                        if str(row[0]) == str(row_id):
                            row[ai_idx] = verdict
                            if ok:
                                row[ok_idx] = ok
                                self.ok_rows.add(int(row_id))
                            break
                    self._apply_filter()

                elif isinstance(msg, tuple) and msg[0] == "AI_CORRECTION_SUPERVISED_ID":
                    (
                        row_id,
                        original_text,
                        original_asr,
                        final_asr,
                        verdict,
                        proposed_asr,
                    ) = msg[1]

                    log_correction_metadata(
                        self.v_json.get(),
                        row_id,
                        original_asr,
                        proposed_asr,
                        verdict,
                    )

                    self.correction_stats[verdict] = self.correction_stats.get(verdict, 0) + 1

                    if original_asr.strip() != final_asr.strip():

                        asr_idx = self.tree["columns"].index("ASR")
                        for row in self.all_rows:
                            if str(row[0]) == str(row_id):
                                row[asr_idx] = final_asr
                                break

                        self._apply_filter()

                        for iid in self.tree.get_children():
                            if self.tree.set(iid, "ID") == str(row_id):
                                self._update_metrics(iid)
                                break

                        self.tree.set(iid, "ASR", final_asr)
                        self._update_metrics(iid)  # This also saves

                        self._log(f"  - Fila {row_id} actualizada. Veredicto supervisor: {verdict}")
                    else:
                        self._log(
                            f"  - Fila {row_id} no modificada. Veredicto supervisor: {verdict}")

                elif isinstance(msg, tuple) and msg[0] == "AI_START_ID":
                    row_id = msg[1]
                    for iid in self.tree.get_children():
                        if self.tree.set(iid, "ID") == row_id:
                            tags = list(self.tree.item(iid, "tags"))
                            if "processing" not in tags:
                                tags.append("processing")
                                self.tree.item(iid, tags=tuple(tags))
                            self.tree.see(iid)
                            break

                elif isinstance(msg, tuple) and msg[0] == "ADVANCED_AI_REVIEW_DONE_ID":
                    row_id, verdict, comment = msg[1]
                    col_idx_map = {name: i for i, name in enumerate(self.tree["columns"])}
                    ai_col_idx = col_idx_map["AI"]
                    ok_col_idx = col_idx_map["OK"]

                    for row in self.all_rows:
                        if str(row[0]) == str(row_id):
                            row[ai_col_idx] = verdict
                            if verdict == "OK":
                                row[ok_col_idx] = "OK"
                            break

                    self._apply_filter()
                    self._log(f"Revisión avanzada Fila {row_id}: {verdict}. Comentario: {comment}")
                    self.save_json()

                elif isinstance(msg, tuple) and msg[0] == "AI_DONE_ID":
                    row_id = msg[1]
                    for iid in self.tree.get_children():
                        if self.tree.set(iid, "ID") == row_id:
                            tags = list(self.tree.item(iid, "tags"))
                            if "processing" in tags:
                                tags.remove("processing")
                                self.tree.item(iid, tags=tuple(tags))
                            break

                elif isinstance(msg, tuple) and msg[0] == "AI_DONE":
                    iid = msg[1]
                    tags = list(self.tree.item(iid, "tags"))
                    if "processing" in tags:
                        tags.remove("processing")
                        self.tree.item(iid, tags=tuple(tags))

                elif isinstance(msg, tuple) and msg[0] == "ERROR":
                    self._close_progress()
                    show_error(msg[1], msg[2])

                elif isinstance(msg, tuple) and msg[0] == "CLOSE_PROGRESS":
                    self._close_progress()

                else:
                    self._log(str(msg))

        except queue.Empty:
            pass

        self.after(250, self._poll)

    # ------------------------------------------------------------- undo/redo --
    def _snapshot(self) -> None:
        rows = [list(self.tree.item(i)["values"]) for i in self.tree.get_children()]
        self.undo_stack.append(json.dumps(rows, ensure_ascii=False))
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _restore(self, data: str) -> None:
        self.all_rows = json.loads(data)
        self._apply_filter()

    def undo(self, event: tk.Event | None = None) -> None:
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        current = json.dumps([
            list(self.tree.item(i)["values"]) for i in self.tree.get_children()
        ], ensure_ascii=False)
        self.redo_stack.append(current)
        self._restore(state)
        self.save_json()

    def redo(self, event: tk.Event | None = None) -> None:
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        current = json.dumps([
            list(self.tree.item(i)["values"]) for i in self.tree.get_children()
        ], ensure_ascii=False)
        self.undo_stack.append(current)
        self._restore(state)
        self.save_json()

    def create_edl(self) -> None:
        from tkinter import filedialog
        if not self.v_ref.get():
            messagebox.showerror("Error", "Selecciona un guion")
            return
        json_path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not json_path:
            return
        try:
            script_text = read_script(self.v_ref.get())
            edl = build_intervals(script_text, json_path)
            out = Path(json_path).with_suffix(".edl.json")
            Path(out).write_text(json.dumps(edl, indent=2), encoding="utf8")
            messagebox.showinfo("EDL", f"Guardado {out}")
        except Exception as exc:
            show_error("Error", exc)

    def second_pass_sync(self) -> None:
        """
        Runs a second pass to fix word alignment issues by shifting words
        between adjacent rows that are not marked as correct (✅).
        """
        self._log("⏳ Iniciando segunda pasada de sincronización...")
        self._snapshot()

        children = list(self.tree.get_children())
        if not children:
            self._log("La tabla está vacía.")
            return

        changes_count = 0
        for i in range(1, len(children)):
            prev_iid = children[i - 1]
            curr_iid = children[i]

            # Skip if either row is already perfect or merged.
            if self.tree.set(prev_iid, "✓") == "✅" or self.tree.set(curr_iid, "✓") == "✅":
                continue
            if (self.merged_tag in self.tree.item(prev_iid, "tags")
                    or self.merged_tag in self.tree.item(curr_iid, "tags")):
                continue

            # --- Store original state ---
            prev_orig = self.tree.set(prev_iid, "Original")
            prev_asr = self.tree.set(prev_iid, "ASR")
            curr_orig = self.tree.set(curr_iid, "Original")
            curr_asr = self.tree.set(curr_iid, "ASR")

            if not all([prev_orig, prev_asr, curr_orig, curr_asr]):
                continue

            prev_orig_words = prev_orig.split()
            prev_asr_words = prev_asr.split()
            curr_orig_words = curr_orig.split()
            curr_asr_words = curr_asr.split()

            if not all([
                prev_orig_words, prev_asr_words, curr_orig_words, curr_asr_words
            ]):
                continue

            def get_wer(ref, hyp):
                ref_t = normalize(ref, strip_punct=False).split()
                hyp_t = normalize(hyp, strip_punct=False).split()
                if not hyp_t:
                    return 1.0
                return Levenshtein.normalized_distance(ref_t, hyp_t)

            original_wer1 = get_wer(prev_orig, prev_asr)
            original_wer2 = get_wer(curr_orig, curr_asr)
            original_combined_wer = original_wer1 + original_wer2

            best_wer = original_combined_wer
            best_config = None

            # --- Scenario 1: Move last word from previous to current ---
            if len(prev_orig_words) > 1 and len(prev_asr_words) > 1:
                p_orig_last = prev_orig_words[-1]
                new_prev_orig_s1 = " ".join(prev_orig_words[:-1])
                new_curr_orig_s1 = f"{p_orig_last} {curr_orig}"
                p_asr_last = prev_asr_words[-1]
                new_prev_asr_s1 = " ".join(prev_asr_words[:-1])
                new_curr_asr_s1 = f"{p_asr_last} {curr_asr}"
                wer1 = get_wer(new_prev_orig_s1, new_prev_asr_s1)
                wer2 = get_wer(new_curr_orig_s1, new_curr_asr_s1)
                if wer1 + wer2 < best_wer:
                    best_wer = wer1 + wer2
                    best_config = (
                        new_prev_orig_s1, new_prev_asr_s1,
                        new_curr_orig_s1, new_curr_asr_s1)

            # --- Scenario 2: Move first word from current to previous ---
            if len(curr_orig_words) > 1 and len(curr_asr_words) > 1:
                c_orig_first = curr_orig_words[0]
                new_curr_orig_s2 = " ".join(curr_orig_words[1:])
                new_prev_orig_s2 = f"{prev_orig} {c_orig_first}"
                c_asr_first = curr_asr_words[0]
                new_curr_asr_s2 = " ".join(curr_asr_words[1:])
                new_prev_asr_s2 = f"{prev_asr} {c_asr_first}"
                wer1 = get_wer(new_prev_orig_s2, new_prev_asr_s2)
                wer2 = get_wer(new_curr_orig_s2, new_curr_asr_s2)
                if wer1 + wer2 < best_wer:
                    best_wer = wer1 + wer2
                    best_config = (
                        new_prev_orig_s2, new_prev_asr_s2,
                        new_curr_orig_s2, new_curr_asr_s2)

            # --- Apply best change if any ---
            if best_config:
                changes_count += 1
                (new_prev_orig_txt, new_prev_asr_txt,
                 new_curr_orig_txt, new_curr_asr_txt) = best_config
                self.tree.set(prev_iid, "Original", new_prev_orig_txt)
                self.tree.set(prev_iid, "ASR", new_prev_asr_txt)
                self.tree.set(curr_iid, "Original", new_curr_orig_txt)
                self.tree.set(curr_iid, "ASR", new_curr_asr_txt)
                self._update_metrics(prev_iid, save=False)
                self._update_metrics(curr_iid, save=False)
                prev_id = self.tree.set(prev_iid, "ID")
                curr_id = self.tree.set(curr_iid, "ID")
                self._log(
                    f"  - Corrección entre filas {prev_id} y {curr_id} "
                    f"(WER: {original_combined_wer:.2f} → {best_wer:.2f})")

        if changes_count > 0:
            self.save_json()
            self._log(f"✔ Segunda pasada completada. Se aplicaron {changes_count} correcciones.")
        else:
            self._log("✔ Segunda pasada completada. No se encontraron correcciones obvias.")

    def generate_correction_report(self) -> None:
        """Generates and saves a summary report of AI correction stats."""
        if not self.v_json.get():
            messagebox.showwarning("Sin JSON", "Carga un JSON para generar un informe.")
            return
        if not self.correction_stats:
            messagebox.showinfo("Sin datos", "No hay estadísticas de corrección para informar.")
            return

        total = sum(self.correction_stats.values())
        plausible = self.correction_stats.get("plausible", 0)
        implausible = self.correction_stats.get("implausible", 0)
        no_change = self.correction_stats.get("no_change", 0)

        report_lines = [
            "Informe de Corrección de Transcripción con IA",
            "==============================================",
            f"Total de filas analizadas: {total}",
            f"  - Correcciones aplicadas (Plausible): {plausible} ({plausible/total:.1%})",
            f"  - Correcciones revocadas (Implausible): {implausible} ({implausible/total:.1%})",
            f"  - Sin cambios propuestos: {no_change} ({no_change/total:.1%})",
        ]
        report_str = "\n".join(report_lines)

        try:
            report_path = Path(self.v_json.get()).with_suffix(".correction_report.txt")
            report_path.write_text(report_str, encoding="utf-8")
            self._log(f"✔ Informe de corrección guardado en: {report_path}")
            messagebox.showinfo("Informe Guardado", f"El informe se ha guardado en:\n{report_path}")
        except Exception as e:
            show_error("Error al guardar informe", e)

    def advanced_ai_review(self) -> None:

        """Runs an advanced, contextual AI review on selected or all relevant rows."""
        # Single row mode
        if self.ai_one.get():
            sel = self.tree.selection()
            if not sel:
                messagebox.showwarning("Selección inválida", "Selecciona una única fila para la revisión avanzada.")
                return

            iid = sel[0]
            row_id = self.tree.set(iid, "ID")

            row_data = next((r for r in self.all_rows if str(r[0]) == row_id), None)
            if not row_data:
                return

            ai_status = row_data[3].lower()
            flag_status = row_data[1]
            if ai_status != 'mal' and flag_status != '⚠️':
                messagebox.showinfo("No es necesario", "La revisión avanzada es para filas marcadas como 'mal' o '⚠️'.")
                return

            self._start_advanced_review_thread_for_id(row_id)

            return

        # Batch mode
        rows_to_review = [
            r for r in self.all_rows
            if r[3].lower() == 'mal' or r[1] == '⚠️'
        ]

        if not rows_to_review:
            messagebox.showinfo("Sin filas para revisar", "No se encontraron filas marcadas como 'mal' o '⚠️'.")
            return

        self._log(f"⏳ Iniciando Revisión AI Avanzada para {len(rows_to_review)} filas…")
        threading.Thread(
            target=self._batch_advanced_ai_review_worker,
            args=(rows_to_review,),
            daemon=True
        ).start()

    def _get_row_context(self, row_id: str) -> dict | None:
        """Builds the context for a given row ID from self.all_rows."""
        try:
            row_index = next(i for i, r in enumerate(self.all_rows) if str(r[0]) == row_id)
        except StopIteration:
            return None

        context = {}
        current_row = self.all_rows[row_index]
        context["current"] = {
            "id": current_row[0], "original": current_row[6], "asr": current_row[7]
        }
        if row_index > 0:
            prev_row = self.all_rows[row_index - 1]
            context["previous"] = {"id": prev_row[0], "original": prev_row[6], "asr": prev_row[7]}
        if row_index < len(self.all_rows) - 1:
            next_row = self.all_rows[row_index + 1]
            context["next"] = {"id": next_row[0], "original": next_row[6], "asr": next_row[7]}
        return context

    def _start_advanced_review_thread_for_id(self, row_id: str) -> None:
        context = self._get_row_context(row_id)
        if not context:
            return

        self._log(f"⏳ Revisión AI Avanzada para fila {row_id}…")
        self.q.put(("AI_START_ID", row_id))
        threading.Thread(
            target=self._advanced_ai_review_worker,
            args=(context,),
            daemon=True
        ).start()

    def _batch_advanced_ai_review_worker(self, rows_to_review: list) -> None:
        """Worker for batch advanced AI review."""
        for row_data in rows_to_review:
            row_id = str(row_data[0])
            context = self._get_row_context(row_id)
            if not context:
                continue

            self.q.put(("AI_START_ID", row_id))
            try:
                from ai_review import get_advanced_review_verdict
                verdict, comment = get_advanced_review_verdict(context)
                self.q.put(("ADVANCED_AI_REVIEW_DONE_ID", (row_id, verdict, comment)))
            except Exception:
                buf = io.StringIO()
                traceback.print_exc(file=buf)
                self.q.put(buf.getvalue())
            finally:
                self.q.put(("AI_DONE_ID", row_id))

    def _advanced_ai_review_worker(self, context: dict) -> None:
        """Worker for single advanced AI review."""
        row_id = str(context["current"]["id"])
        try:
            from ai_review import get_advanced_review_verdict
            verdict, comment = get_advanced_review_verdict(context)
            self.q.put(("ADVANCED_AI_REVIEW_DONE_ID", (row_id, verdict, comment)))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())
        finally:
            self.q.put(("AI_DONE_ID", row_id))


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    App().mainloop()

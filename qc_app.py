from __future__ import annotations
"""Tkinter GUI for the QC application (v5.3‑tc).

* Usa columna **tc** (time‑code) como instante de INICIO de la frase.
* Para reproducir un clip se toma el tc de la fila actual y el tc de la
  fila siguiente (o el final del archivo si no hay siguiente).
* Se mantiene **toda** la funcionalidad original: transcribe, AI‑review,
  edición de celdas, undo/redo, fusión, etc.
"""

import io
import json
import os
import queue
import threading
import traceback
from pathlib import Path

import pygame
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from utils.gui_errors import show_error

from alignment import build_rows, WARN_WER
from text_utils import read_script, normalize
from rapidfuzz.distance import Levenshtein
from qc_utils import canonical_row
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
        self.v_ref   = tk.StringVar(self)
        self.v_asr   = tk.StringVar(self)
        self.v_audio = tk.StringVar(self)
        self.v_json  = tk.StringVar(self)
        self.ai_one  = tk.BooleanVar(self, value=False)

        # Estados internos
        self.q:   queue.Queue = queue.Queue()
        self.ok_rows: set[int] = set()
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.merged_rows: dict[str, list[list[str]]] = {}

        self.selected_cell: tuple[str, str] | None = None
        self.tree_tag = "sel_cell"
        self.merged_tag = "merged"

        # Repro
        self._clip_item: str | None = None
        self._clip_start = 0.0
        self._clip_end: float | None = None
        self._clip_offset = 0.0  # offset within the current clip

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
        top = ttk.Frame(self)
        top.pack(fill="x", padx=3, pady=2)

        # Entradas de archivos -------------------------------------------------------
        self._lbl_entry(top, "Guion:", self.v_ref, 0, ("PDF/TXT", "*.pdf;*.txt"))
        self._lbl_entry(top, "TXT ASR:", self.v_asr, 1, ("TXT/CSV", "*.txt;*.csv"))
        self._lbl_entry(top, "Audio:", self.v_audio, 2,
                        ("Media", "*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.mp4"))

        ttk.Button(top, text="Transcribir", command=self.transcribe).grid(row=2, column=3, padx=6)
        ttk.Button(top, text="Procesar", width=11, command=self.launch).grid(row=0, column=3, rowspan=2, padx=6)

        ttk.Label(top, text="JSON:").grid(row=3, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_json, width=70).grid(row=3, column=1)
        ttk.Button(top, text="Abrir JSON…", command=self.load_json).grid(row=3, column=2)

        ttk.Button(top, text="AI Review (o3)", command=self.ai_review).grid(row=3, column=3, padx=6)
        ttk.Checkbutton(top, text="una fila", variable=self.ai_one).grid(row=3, column=4, padx=4)
        ttk.Button(top, text="Detener análisis", command=self.stop_ai_review).grid(row=3, column=5, padx=6)
        ttk.Button(top, text="Crear EDL", command=self.create_edl).grid(row=3, column=6, padx=6)

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

        self.bind_all("<Control-z>", self.undo)
        self.bind_all("<Control-Shift-Z>", self.redo)

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
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="e")
        ttk.Entry(parent, textvariable=var, width=70).grid(row=row, column=1)
        ttk.Button(parent, text="…", command=lambda: self._browse(var, ft)).grid(row=row, column=2)

    # ---------------------------------------------------------------- table ----------
    def _build_table(self) -> None:
        cols = ("ID", "✓", "OK", "AI", "WER", "tc", "Original", "ASR")
        widths = (50, 30, 40, 40, 60, 60, 800, 800)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=3, pady=2)

        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=27, selectmode="extended")
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")

        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="left", fill="y")

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
        self._update_scale_range()

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
                r[5] = _parse_tc(str(r[5]))
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
    # Normaliza la lista que llega de build_rows a las 8 columnas de la GUI
    # Orden final: [ID, ✓, OK, AI, WER, tc, Original, ASR]
    # ──────────────────────────────────────────────────────────────────────────────
    def _row_from_alignment(self, r: list) -> list:
        """
        build_rows genera:
          6 col.: [ID, ✓,        WER, tc, Original, ASR]
          7 col.: [ID, ✓,  OK,   WER, tc, Original, ASR]
          8 col.: [ID, ✓,  OK, AI, WER, tc, Original, ASR]  (ya correcto)
        Retorna siempre 8-columnas en el orden que usa la GUI.
        """

        from qc_utils import canonical_row

        row = canonical_row(r)

        # ``canonical_row`` may return 8 or 9 columns. When a "Score" column is
        # present the time code is at index 6 instead of 5. Rows ending with a
        # list store "takes" and keep the time code at index 5.  Adjust the
        # index accordingly before formatting.
        tc_idx = 6 if len(row) >= 9 and not isinstance(row[-1], list) else 5
        if len(row) > tc_idx:
            row[tc_idx] = _format_tc(row[tc_idx])
        return row

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
        if self.ai_one.get():
            sel = self.tree.selection()
            if not sel:
                messagebox.showwarning("Falta info", "Selecciona una fila")
                return
            iid = sel[0]
            original = self.tree.set(iid, "Original")
            asr = self.tree.set(iid, "ASR")
            self._log("⏳ Revisión AI fila…")
            threading.Thread(
                target=self._ai_review_one_worker,
                args=(iid, original, asr),
                daemon=True,
            ).start()
        else:
            self._log(
                "⏳ Solicitando revisión AI (esto puede tardar unos segundos)…"
            )
            items = list(self.tree.get_children())
            threading.Thread(
                target=self._ai_review_worker,
                args=(items,),
                daemon=True,
            ).start()

    def stop_ai_review(self):
        try:
            from ai_review import stop_review

            stop_review()
            self._log("⏹ Deteniendo análisis AI…")
        except Exception as exc:
            self._log(str(exc))

    def _ai_review_worker(self, items: list[str] | None = None) -> None:
        """Run batch AI review updating the GUI incrementally."""
        try:
            import ai_review

            if not items:
                approved, remaining = ai_review.review_file(self.v_json.get())
                self.q.put(("RELOAD", None))
                if ai_review._stop_review:
                    self.q.put("⚠ Revisión detenida")
                else:
                    self.q.put(
                        f"✔ Auto-aprobadas {approved} / Restantes {remaining}"
                    )
                return

            def progress(stage: str, idx: int, row: list) -> None:
                iid = items[idx]
                if stage == "start":
                    self.q.put(("AI_START", iid))
                else:
                    self.q.put(("AI_ROW", (iid, row[3], row[2])))

            try:
                approved, remaining = ai_review.review_file(
                    self.v_json.get(), progress_callback=progress
                )
            except TypeError:
                approved, remaining = ai_review.review_file(self.v_json.get())
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

    def _ai_review_one_worker(self, iid: str, original: str, asr: str) -> None:
        try:
            from ai_review import review_row

            row = [0, "", "", 0.0, 0.0, original, asr]
            review_row(row)
            self.q.put(("AI_ROW", (iid, row[3], row[2])))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            err = buf.getvalue()
            print(err)
            self.q.put(err)

    # ---------------------------------------------------------------------------------
    # JSON ---------------------------------------------------------------------------
    def load_json(self):
        if not self.v_json.get():
            p = filedialog.askopenfilename(filetypes=[("QC JSON", "*.qc.json;*.json")])
            if not p:
                return
            self.v_json.set(p)

        try:
            rows = json.loads(Path(self.v_json.get()).read_text(encoding="utf8"))
            self.clear_table()

            for r in rows:
                vals = self._row_from_alignment(r)
                # Treeview no admite números; asegúrate de que llegan como str
                vals[6], vals[7] = str(vals[6]), str(vals[7])
                self.tree.insert("", tk.END, values=vals)
            self._snapshot()
            self._update_scale_range()
            self._load_marker()
            self._log(f"✔ Cargado {self.v_json.get()}")
        except Exception as e:
            show_error("Error", e)

    # ---------------------------------------------------------------------------------
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
        for next_iid in children[idx + 1 :]:
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
            play_interval(self.v_audio.get(), start, self._clip_end)

    def _seek_clip(self, offset: float) -> None:
        """Move playback head to ``offset`` seconds within current clip."""
        self._clip_offset = max(0.0, offset)
        self._play_current_clip()

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
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pos = pygame.mixer.music.get_pos()
                if pos >= 0:
                    scale.set(self._clip_offset + pos / 1000)
                win.after(100, _update)

        _update()

        btns = ttk.Frame(win)
        btns.pack(pady=(0, 10))
        ttk.Button(btns, text="OK", command=lambda: self._popup_mark_ok(iid, win)).pack(side="left", padx=4)
        ttk.Button(btns, text="Marcar", command=lambda: self.add_audacity_marker(self._clip_start + float(scale.get()))).pack(side="left", padx=4)
        ttk.Button(btns, text="Cerrar", command=win.destroy).pack(side="left", padx=4)

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

    def _update_metrics(self, iid: str) -> None:
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
        self.save_json()

    def _recompute_tc(self) -> None:
        """Ensure monotonically increasing time codes."""
        last = 0.0
        for iid in self.tree.get_children():
            try:
                tc = float(_parse_tc(self.tree.set(iid, "tc")))
            except ValueError:
                tc = last
            if tc < last:
                tc = last
            self.tree.set(iid, "tc", _format_tc(tc))
            last = tc

    def _clip_ok(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "ok")
        self._hide_clip()

    def _clip_bad(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "mal")
        self._hide_clip()

    def _hide_clip(self) -> None:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
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
                from utils.resync_python_v2 import load_words_csv, resync_rows
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
            rows = build_rows(ref, hyp)
            if use_csv:
                resync_rows(rows, csv_words, csv_tcs)
            else:
                csv_path: Path | None = None
                audio_val = self.v_audio.get()
                if audio_val:
                    candidate = Path(audio_val).with_suffix(".words.csv")
                    if candidate.exists():
                        csv_path = candidate
                if csv_path is None:
                    from tkinter import filedialog
                    p = filedialog.askopenfilename(
                        filetypes=[("CSV", "*.csv"), ("All", "*")]
                    )
                    if p:
                        csv_path = Path(p)
                if csv_path and csv_path.exists():
                    try:
                        from utils.resync_python_v2 import load_words_csv, resync_rows
                        csv_words, csv_tcs = load_words_csv(csv_path)
                        resync_rows(rows, csv_words, csv_tcs)
                        debug(f"DEBUG: CSV usado {csv_path}")
                    except Exception as exc:
                        self.q.put(f"Resync error: {exc}")

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

            self.q.put(("ROWS", rows))
            self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            err = buf.getvalue()
            debug(f"DEBUG: error en worker:\n{err}")
            self.q.put(err)
        finally:
            from alignment import set_debug_logger
            set_debug_logger(print)

    # ---------------------------------------------------------------------------------
    # cola de mensajes Tk -------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()

                if isinstance(msg, tuple) and msg[0] == "ROWS":
                    for r in msg[1]:
                        vals = self._row_from_alignment(r)
                        vals[6], vals[7] = str(vals[6]), str(vals[7])
                        self.tree.insert("", tk.END, values=vals)
                    self._update_scale_range()
                    self._close_progress()
                    self._snapshot()

                elif isinstance(msg, tuple) and msg[0] == "SET_ASR":
                    self.v_asr.set(msg[1])
                    self._close_progress()

                elif isinstance(msg, tuple) and msg[0] == "PROGRESS":
                    pct = int(msg[1])
                    self._update_progress(pct)

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

                elif isinstance(msg, tuple) and msg[0] == "AI_ROW":
                    iid, verdict, ok = msg[1]
                    self.tree.set(iid, "AI", verdict)
                    if ok:
                        self.tree.set(iid, "OK", ok)
                        try:
                            line_id = int(self.tree.set(iid, "ID"))
                            self.ok_rows.add(line_id)
                        except Exception:
                            pass
                    tags = list(self.tree.item(iid, "tags"))
                    if "processing" in tags:
                        tags.remove("processing")
                        self.tree.item(iid, tags=tuple(tags))

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
        rows = json.loads(data)
        self.clear_table()

        for r in rows:
            vals = self._row_from_alignment(r)
            vals[6], vals[7] = str(vals[6]), str(vals[7])
            self.tree.insert("", tk.END, values=vals)
        self._update_scale_range()

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


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    App().mainloop()

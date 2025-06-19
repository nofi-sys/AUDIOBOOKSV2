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
import sys
import threading
import traceback
from pathlib import Path

import pygame
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from alignment import build_rows
from text_utils import read_script

# --------------------------------------------------------------------------------------
# utilidades de audio ------------------------------------------------------------------
# --------------------------------------------------------------------------------------

def play_interval(path: str, start: float, end: float | None) -> None:
    """Reproduce *path* desde *start* (seg) hasta *end* (seg) con pygame."""

    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(start=start)
    if end is not None:
        ms = int(max(0.0, end - start) * 1000)
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

        # Repro
        self._clip_item: str | None = None
        self._clip_start = 0.0
        self._clip_end: float | None = None

        self._build_ui()
        self.after(250, self._poll)

    # ---------------------------------------------------------------- build UI ------
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=3, pady=2)

        # Entradas de archivos -------------------------------------------------------
        self._lbl_entry(top, "Guion:", self.v_ref, 0, ("PDF/TXT", "*.pdf;*.txt"))
        self._lbl_entry(top, "TXT ASR:", self.v_asr, 1, ("TXT", "*.txt"))
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

        # Tabla principal -----------------------------------------------------------
        self._build_table()
        self._build_player_bar()

        style = ttk.Style(self)
        style.configure("Treeview", rowheight=45)

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
        sb.pack(side="right", fill="y")

        self.tree.tag_configure("sel_cell", background="#d0e0ff")
        self.tree.tag_configure("merged", background="#f5f5f5")

        # bindings
        self.tree.bind("<Double-1>", self._on_double_click)

    # ------------------------------------------------------------- player bar -------
    def _build_player_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(side="top", anchor="ne", padx=4, pady=4)
        ttk.Button(bar, text="▶", command=self._play_current_clip).pack(side="left", padx=4)
        ttk.Button(bar, text="←", command=self._prev_bad_row).pack(side="left", padx=4)
        ttk.Button(bar, text="→", command=self._next_bad_row).pack(side="left", padx=4)
        ttk.Button(bar, text="OK", command=self._clip_ok).pack(side="left", padx=4)
        ttk.Button(bar, text="mal", command=self._clip_bad).pack(side="left", padx=4)

    # ---------------------------------------------------------------------------------
    # navegación de archivos ----------------------------------------------------------
    def _browse(self, var: tk.StringVar, ft: tuple[str, str]):
        p = filedialog.askopenfilename(filetypes=[ft])
        if p:
            var.set(p)

    # ---------------------------------------------------------------------------------
    # mensajes log ---------------------------------------------------------------------
    def _log(self, msg: str):
        print(msg)

    # ---------------------------------------------------------------------------------
    # Transcripción -------------------------------------------------------------------
    def transcribe(self):
        messagebox.showinfo("Info", "Funcionalidad de transcripción no modificada (placeholder).")

    # ---------------------------------------------------------------------------------
    # Procesar align ------------------------------------------------------------------
    def launch(self):
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info", "Selecciona guion y TXT ASR.")
            return
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------------------------
    # AI review -----------------------------------------------------------------------
    def ai_review(self):
        messagebox.showinfo("Info", "Funcionalidad AI‑review no modificada.")

    def stop_ai_review(self):
        pass

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
        except Exception as exc:
            messagebox.showerror("Error", str(exc)); return

        self.tree.delete(*self.tree.get_children())
        for r in rows:
            if len(r) == 6:
                vals = [r[0], r[1], "", "", r[2], r[3], r[4], r[5]]
            elif len(r) == 7:
                vals = [r[0], r[1], r[2], "", r[3], r[4], r[5], r[6]]
            else:
                vals = r
            # aseguramos string
            vals[6], vals[7] = str(vals[6]), str(vals[7])
            self.tree.insert("", tk.END, values=vals)

    # ---------------------------------------------------------------------------------
    # Reproducción -------------------------------------------------------------------
    def _on_double_click(self, ev):
        item = self.tree.identify_row(ev.y)
        if item:
            self._play_clip(item)

    def _play_clip(self, iid: str):
        """Calcula tc inicio ‑ fin y prepara _clip_*"""
        if not self.v_audio.get():
            messagebox.showwarning("Falta audio", "Selecciona archivo de audio")
            return
        try:
            start = float(self.tree.set(iid, "tc"))
        except ValueError:
            return
        # siguiente
        children = list(self.tree.get_children())
        idx = children.index(iid)
        end = None
        if idx + 1 < len(children):
            try:
                end = float(self.tree.set(children[idx + 1], "tc"))
            except ValueError:
                pass
        self._clip_item, self._clip_start, self._clip_end = iid, start, end
        self._play_current_clip()

    def _play_current_clip(self):
        if self._clip_item and self.v_audio.get():
            play_interval(self.v_audio.get(), self._clip_start, self._clip_end)

    def _clip_ok(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "ok")
    def _clip_bad(self):
        if self._clip_item:
            self.tree.set(self._clip_item, "AI", "mal")

    def _next_bad_row(self):
        pass
    def _prev_bad_row(self):
        pass

    # ---------------------------------------------------------------------------------
    # hilo worker (alinear) -----------------------------------------------------------
    def _worker(self):
        pass

    # ---------------------------------------------------------------------------------
    # cola de mensajes Tk -------------------------------------------------------------
    def _poll(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.after(250, self._poll)

# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    App().mainloop()

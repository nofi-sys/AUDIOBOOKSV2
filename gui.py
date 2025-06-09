"""Tkinter GUI for the QC application."""

import io
import os
import json
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from alignment import build_rows
from text_utils import read_script
import textwrap


class App(tk.Tk):
    def __init__(self) -> None:
        use_tk = bool(os.environ.get("DISPLAY"))
        super().__init__(useTk=use_tk)
        if use_tk:
            self.title("QC-Audiolibro  v5.1")
            self.geometry("1850x760")

        self.v_ref = tk.StringVar(self)
        self.v_asr = tk.StringVar(self)
        self.v_audio = tk.StringVar(self)
        self.v_json = tk.StringVar(self)
        self.q: queue.Queue = queue.Queue()
        self.ok_rows: set[int] = set()
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []

        self.selected_cell: tuple[str, str] | None = None
        self.tree_tag = "cell_sel"


        top = ttk.Frame(self)
        top.pack(fill="x", padx=3, pady=2)
        ttk.Label(top, text="Guion:").grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_ref, width=70).grid(row=0, column=1)
        ttk.Button(
            top,
            text="…",
            command=lambda: self.browse(
                self.v_ref, ("PDF/TXT", "*.pdf;*.txt")
            ),
        ).grid(row=0, column=2)

        ttk.Label(top, text="TXT ASR:").grid(row=1, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_asr, width=70).grid(row=1, column=1)
        ttk.Button(
            top,
            text="…",
            command=lambda: self.browse(self.v_asr, ("TXT", "*.txt")),
        ).grid(row=1, column=2)

        ttk.Label(top, text="Audio:").grid(row=2, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_audio, width=70).grid(row=2, column=1)
        ttk.Button(
            top,
            text="…",
            command=lambda: self.browse(
                self.v_audio,
                ("Media", "*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.mp4"),
            ),
        ).grid(row=2, column=2)
        ttk.Button(top, text="Transcribir", command=self.transcribe).grid(
            row=2, column=3, padx=6
        )

        ttk.Button(top, text="Procesar", width=11, command=self.launch).grid(
            row=0, column=3, rowspan=2, padx=6
        )
        ttk.Label(top, text="JSON:").grid(row=3, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_json, width=70).grid(row=3, column=1)
        ttk.Button(top, text="Abrir JSON…", command=self.load_json).grid(
            row=3, column=2
        )

        style = ttk.Style(self)
        style.configure("Treeview", rowheight=45)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=3, pady=2)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("ID", "✓", "OK", "WER", "dur", "Original", "ASR"),
            show="headings",
            height=27,
        )
        for c, w in zip(
            ("ID", "✓", "OK", "WER", "dur", "Original", "ASR"),
            (50, 30, 40, 60, 60, 800, 800),
        ):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")


        self.tree.tag_configure(self.tree_tag, background="#d0e0ff")

        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(
            label="Mover ↑",
            command=lambda: self._move_cell("up"),
        )
        self.menu.add_command(
            label="Mover ↓",
            command=lambda: self._move_cell("down"),
        )
        self.menu.add_separator()
        self.menu.add_command(
            label="↑ última palabra",
            command=lambda: self._move_word("up", "last"),
        )
        self.menu.add_command(
            label="↓ última palabra",
            command=lambda: self._move_word("down", "last"),
        )
        self.menu.add_command(
            label="↑ primera palabra",
            command=lambda: self._move_word("up", "first"),
        )
        self.menu.add_command(
            label="↓ primera palabra",
            command=lambda: self._move_word("down", "first"),
        )
        self.tree.bind("<Button-1>", self._cell_click)
        self.tree.bind("<Button-3>", self._popup_menu)
        self.bind_all("<Control-z>", self.undo)
        self.bind_all("<Control-Shift-Z>", self.redo)


        self.tree.bind("<Double-1>", self._toggle_ok)

        self.log_box = scrolledtext.ScrolledText(
            self, height=5, state="disabled"
        )
        self.log_box.pack(fill="x", padx=3, pady=2)

        self.after(250, self._poll)

    def browse(self, var: tk.StringVar, ft: tuple[str, str]) -> None:
        path = filedialog.askopenfilename(filetypes=[ft])
        if path:
            var.set(path)

    def log_msg(self, msg: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def clear_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        self.ok_rows.clear()

    def _toggle_ok(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if col != "#3" or not item:
            return
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

    def _cell_click(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        for iid in self.tree.get_children():
            tags = list(self.tree.item(iid, "tags"))
            if self.tree_tag in tags:
                tags.remove(self.tree_tag)
                self.tree.item(iid, tags=tuple(tags))
        if col not in ("#6", "#7") or not item:
            self.selected_cell = None
            return
        tags = list(self.tree.item(item, "tags"))
        if self.tree_tag not in tags:
            tags.append(self.tree_tag)
            self.tree.item(item, tags=tuple(tags))
        self.selected_cell = (item, col)

    def _popup_menu(self, event: tk.Event) -> None:
        self._cell_click(event)
        if self.selected_cell:
            self.menu.tk_popup(event.x_root, event.y_root)

    def _move_cell(self, direction: str) -> None:
        """Fuse the selected cell with the row above or below.

        ``direction`` should be ``"up"`` or ``"down"``. Text from the source
        cell is appended to the destination cell, respecting the desired order.
        The source cell is cleared and the entire row removed if both columns
        become empty. A snapshot of the table is taken so the operation can be
        undone.
        """
        if not self.selected_cell:
            return
        item, col_id = self.selected_cell
        children = self.tree.get_children()
        idx = children.index(item)
        dst_idx = idx - 1 if direction == "up" else idx + 1
        if dst_idx < 0 or dst_idx >= len(children):
            return
        dst_item = children[dst_idx]
        col = "Original" if col_id == "#6" else "ASR"
        src_text = self.tree.set(item, col)
        if not src_text:
            return
        dst_text = self.tree.set(dst_item, col)
        if direction == "up":
            fused = (dst_text.rstrip().rstrip(".") + " " + src_text).strip()
        else:
            fused = (src_text.rstrip(".") + " " + dst_text).strip()
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

        self.save_json()

    def _move_word(self, direction: str, which: str) -> None:
        """Move the first or last word to the adjacent row."""
        if not self.selected_cell:
            return
        item, col_id = self.selected_cell
        children = self.tree.get_children()
        idx = children.index(item)
        dst_idx = idx - 1 if direction == "up" else idx + 1
        if dst_idx < 0 or dst_idx >= len(children):
            return
        dst_item = children[dst_idx]
        col = "Original" if col_id == "#6" else "ASR"
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
        if direction == "up":
            fused = (dst_text + " " + word).strip()
        else:
            fused = (word + " " + dst_text).strip()
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

        self.save_json()

    def transcribe(self) -> None:
        if not self.v_audio.get():
            messagebox.showwarning("Falta info", "Selecciona archivo de audio")
            return
        if not self.v_ref.get():
            messagebox.showwarning("Falta info", "Selecciona guion para guiar la transcripción")
            return
        self.log_msg("⏳ Transcribiendo…")
        threading.Thread(target=self._transcribe_worker, daemon=True).start()

    def _transcribe_worker(self) -> None:
        try:
            from transcriber import transcribe_file

            out = transcribe_file(
                self.v_audio.get(), script_path=self.v_ref.get()
            )
            self.q.put(("SET_ASR", str(out)))
            self.q.put(f"✔ Transcripción guardada en {out}")
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())

    def launch(self) -> None:
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info", "Selecciona guion y TXT ASR.")
            return
        self.clear_table()
        self.log_msg("⏳ Iniciando…")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            self.q.put("→ Leyendo guion…")
            ref = read_script(self.v_ref.get())

            self.q.put("→ TXT externo cargado")
            hyp = Path(self.v_asr.get()).read_text(
                encoding="utf8", errors="ignore"
            )

            self.q.put("→ Alineando…")
            rows = build_rows(ref, hyp)

            out = Path(self.v_asr.get()).with_suffix(".qc.json")
            out.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8"
            )

            self.q.put(("ROWS", rows))
            self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())

    def save_json(self) -> None:
        """Save current table rows to ``v_json`` path."""

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
            Path(self.v_json.get()).write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8"
            )
            self.log_msg(f"✔ Guardado {self.v_json.get()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_json(self) -> None:
        if not self.v_json.get():
            p = filedialog.askopenfilename(
                filetypes=[("QC JSON", "*.qc.json;*.json")]
            )
            if not p:
                return
            self.v_json.set(p)
        try:
            rows = json.loads(
                Path(self.v_json.get()).read_text(encoding="utf8")
            )
            self.clear_table()
            for r in rows:
                if len(r) == 6:
                    vals = [r[0], r[1], "", r[2], r[3], r[4], r[5]]
                else:
                    vals = r
                vals[5] = textwrap.fill(str(vals[5]), width=80)
                vals[6] = textwrap.fill(str(vals[6]), width=80)
                self.tree.insert("", tk.END, values=vals)
            self._snapshot()
            self.log_msg(f"✔ Cargado {self.v_json.get()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _poll(self) -> None:
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "ROWS":
                    for r in msg[1]:
                        vals = [r[0], r[1], "", r[2], r[3], r[4], r[5]]
                        vals[5] = textwrap.fill(str(vals[5]), width=80)
                        vals[6] = textwrap.fill(str(vals[6]), width=80)
                        self.tree.insert("", tk.END, values=vals)
                    self._snapshot()
                elif isinstance(msg, tuple) and msg[0] == "SET_ASR":
                    self.v_asr.set(msg[1])
                else:
                    self.log_msg(str(msg))
        except queue.Empty:
            pass
        self.after(250, self._poll)

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
            if len(r) == 6:
                vals = [r[0], r[1], "", r[2], r[3], r[4], r[5]]
            else:
                vals = r
            vals[5] = textwrap.fill(str(vals[5]), width=80)
            vals[6] = textwrap.fill(str(vals[6]), width=80)
            self.tree.insert("", tk.END, values=vals)

    def undo(self, event: tk.Event | None = None) -> None:
        if not self.undo_stack:
            return
        state = self.undo_stack.pop()
        current = json.dumps(
            [list(self.tree.item(i)["values"]) for i in self.tree.get_children()],
            ensure_ascii=False,
        )
        self.redo_stack.append(current)
        self._restore(state)
        self.save_json()

    def redo(self, event: tk.Event | None = None) -> None:
        if not self.redo_stack:
            return
        state = self.redo_stack.pop()
        current = json.dumps(
            [list(self.tree.item(i)["values"]) for i in self.tree.get_children()],
            ensure_ascii=False,
        )
        self.undo_stack.append(current)
        self._restore(state)
        self.save_json()


if __name__ == "__main__":
    App().mainloop()

"""Tkinter GUI for the QC application."""

import io
import os
import json
import queue
import threading
import traceback
from pathlib import Path
import sys

import pygame

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from alignment import build_rows
from text_utils import read_script


class ClipWindow(tk.Toplevel):
    """Simple audio clip player with approval buttons."""

    def __init__(
        self,
        master: tk.Tk,
        audio_path: str,
        start: float,
        duration: float,
        on_ok: callable,
        on_bad: callable,
    ) -> None:
        super().__init__(master)
        self.title("Revisar audio")
        self.geometry("300x120")
        self.audio_path = audio_path
        self.start = start
        self.duration = duration
        self.on_ok = on_ok
        self.on_bad = on_bad

        ttk.Button(self, text="▶", command=self.play).pack(pady=4)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=4)
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="mal", command=self._bad).pack(side="left", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._close)

    def play(self) -> None:
        pygame.mixer.init()
        pygame.mixer.music.load(self.audio_path)
        pygame.mixer.music.play(start=self.start)
        self.after(int(self.duration * 1000), pygame.mixer.music.stop)

    def _ok(self) -> None:
        self.on_ok()
        self._close()

    def _bad(self) -> None:
        self.on_bad()
        self._close()

    def _close(self) -> None:
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self.destroy()


class App(tk.Tk):
    def __init__(self) -> None:
        use_tk = True
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            use_tk = False
        super().__init__(useTk=use_tk)
        if use_tk:
            self.title("QC-Audiolibro  v5.1")
            self.geometry("1850x760")

        self.v_ref = tk.StringVar(self)
        self.v_asr = tk.StringVar(self)
        self.v_audio = tk.StringVar(self)
        self.v_json = tk.StringVar(self)
        self.ai_one = tk.BooleanVar(self, value=False)
        self.q: queue.Queue = queue.Queue()
        self.ok_rows: set[int] = set()
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []

        self.merged_rows: dict[str, list[list[str]]] = {}

        self.selected_cell: tuple[str, str] | None = None
        self.tree_tag = "cell_sel"
        self.merged_tag = "merged"


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
        ttk.Button(top, text="AI Review (o3)", command=self.ai_review).grid(
            row=3, column=3, padx=6
        )
        ttk.Checkbutton(
            top,
            text="una fila",
            variable=self.ai_one,
        ).grid(row=3, column=4, padx=4)
        ttk.Button(
            top,
            text="Detener análisis",
            command=self.stop_ai_review,
        ).grid(row=3, column=5, padx=6)

        style = ttk.Style(self)
        style.configure("Treeview", rowheight=45)

        table_frame = ttk.Frame(self)
        table_frame.pack(fill="both", expand=True, padx=3, pady=2)

        self.tree = ttk.Treeview(
            table_frame,
            columns=("ID", "✓", "OK", "AI", "WER", "dur", "Original", "ASR"),
            show="headings",
            height=27,
            selectmode="extended",
        )
        for c, w in zip(
            ("ID", "✓", "OK", "AI", "WER", "dur", "Original", "ASR"),
            (50, 30, 40, 40, 60, 60, 800, 800),
        ):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")


        self.tree.tag_configure(self.tree_tag, background="#d0e0ff")
        self.tree.tag_configure(
            self.merged_tag,
            background="#f5f5f5",
        )

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
        self.menu.add_separator()
        self.menu.add_command(
            label="Fusionar filas seleccionadas",
            command=self._merge_selected_rows,
        )
        self.menu.add_command(
            label="Desagrupar fila",
            command=self._unmerge_row,
        )
        self.tree.bind("<Button-1>", self._cell_click)
        self.tree.bind("<Button-3>", self._popup_menu)
        self.bind_all("<Control-z>", self.undo)
        self.bind_all("<Control-Shift-Z>", self.redo)


        self.tree.bind("<Double-1>", self._handle_double)

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

    def _handle_double(self, event: tk.Event) -> None:
        item = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not item:
            return
        if col == "#3":
            self._toggle_ok(item)
            return
        self._play_clip(item)

    def _play_clip(self, item: str) -> None:
        if not self.v_audio.get():
            messagebox.showwarning("Falta info", "Selecciona archivo de audio")
            return
        start = 0.0
        children = self.tree.get_children()
        for iid in children:
            if iid == item:
                break
            try:
                start += float(self.tree.set(iid, "dur"))
            except ValueError:
                pass
        try:
            dur = float(self.tree.set(item, "dur"))
        except ValueError:
            dur = 0.0

        def ok_cb() -> None:
            self.tree.set(item, "AI", "ok")
            self.save_json()

        def bad_cb() -> None:
            self.tree.set(item, "AI", "mal")
            self.save_json()

        ClipWindow(self, self.v_audio.get(), start, dur, ok_cb, bad_cb)

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
        col = "Original" if col_id == "#7" else "ASR"
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

    def _merge_selected_rows(self) -> None:
        sel = list(self.tree.selection())
        if len(sel) < 2:
            return
        sel.sort(key=lambda iid: self.tree.index(iid))
        first = sel[0]
        originals: list[str] = []
        asrs: list[str] = []
        total_dur = 0.0

        for iid in sel:
            originals.append(self.tree.set(iid, "Original").strip())
            asrs.append(self.tree.set(iid, "ASR").strip())
            try:
                total_dur += float(self.tree.set(iid, "dur"))
            except ValueError:
                pass

        fuse = lambda parts: " ".join(p.rstrip(".,;") for p in parts if p)

        self._snapshot()
        self.merged_rows[first] = [list(self.tree.item(i)["values"]) for i in sel]

        self.tree.set(first, "Original", fuse(originals))
        self.tree.set(first, "ASR", fuse(asrs))
        self.tree.set(first, "dur", f"{total_dur:.2f}")
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
        for new_id, iid in enumerate(self.tree.get_children()):
            self.tree.set(iid, "ID", new_id)
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
            err = buf.getvalue()
            print(err)
            self.q.put(err)

    def launch(self) -> None:
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info", "Selecciona guion y TXT ASR.")
            return
        self.clear_table()
        self.log_msg("⏳ Iniciando…")
        threading.Thread(target=self._worker, daemon=True).start()

    def ai_review(self) -> None:
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
            self.log_msg("⏳ Revisión AI fila…")
            threading.Thread(
                target=self._ai_review_one_worker,
                args=(iid, original, asr),
                daemon=True,
            ).start()
        else:
            self.log_msg(
                "⏳ Solicitando revisión AI (esto puede tardar unos segundos)…"
            )
            threading.Thread(target=self._ai_review_worker, daemon=True).start()

    def stop_ai_review(self) -> None:
        """Signal the AI review thread to stop."""
        try:
            from ai_review import stop_review

            stop_review()
            self.log_msg("⏹ Deteniendo análisis AI…")
        except Exception as exc:
            self.log_msg(str(exc))

    def _ai_review_worker(self) -> None:
        try:
            import ai_review

            approved, remaining = ai_review.review_file(self.v_json.get())
            self.q.put(("RELOAD", None))
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
                    vals = [r[0], r[1], "", "", r[2], r[3], r[4], r[5]]
                elif len(r) == 7:
                    vals = [r[0], r[1], r[2], "", r[3], r[4], r[5], r[6]]
                else:
                    vals = r
                vals[6] = str(vals[6])
                vals[7] = str(vals[7])
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
                        vals = [r[0], r[1], "", "", r[2], r[3], r[4], r[5]]
                        vals[6] = str(vals[6])
                        vals[7] = str(vals[7])
                        self.tree.insert("", tk.END, values=vals)
                    self._snapshot()
                elif isinstance(msg, tuple) and msg[0] == "RELOAD":
                    self.load_json()
                elif isinstance(msg, tuple) and msg[0] == "SET_ASR":
                    self.v_asr.set(msg[1])
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
                    self.save_json()
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
                vals = [r[0], r[1], "", "", r[2], r[3], r[4], r[5]]
            elif len(r) == 7:
                vals = [r[0], r[1], r[2], "", r[3], r[4], r[5], r[6]]
            else:
                vals = r
            vals[6] = str(vals[6])
            vals[7] = str(vals[7])
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

"""Tkinter GUI for the QC application."""

import io
import json
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from alignment import build_rows
from text_utils import read_script


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QC-Audiolibro  v5.1")
        self.geometry("1850x760")

        self.v_ref = tk.StringVar()
        self.v_asr = tk.StringVar()
        self.v_json = tk.StringVar()
        self.q: queue.Queue = queue.Queue()
        self.ok_rows: set[int] = set()

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

        ttk.Button(top, text="Procesar", width=11, command=self.launch).grid(
            row=0, column=3, rowspan=2, padx=6
        )

        ttk.Label(top, text="JSON:").grid(row=2, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_json, width=70).grid(
            row=2, column=1
        )
        ttk.Button(top, text="Abrir JSON…", command=self.load_json).grid(
            row=2, column=2
        )

        self.tree = ttk.Treeview(
            self,
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
        self.tree.pack(fill="both", expand=True, padx=3, pady=2)

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
                self.tree.insert("", tk.END, values=vals)
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
                        self.tree.insert("", tk.END, values=vals)
                else:
                    self.log_msg(str(msg))
        except queue.Empty:
            pass
        self.after(250, self._poll)


if __name__ == "__main__":
    App().mainloop()

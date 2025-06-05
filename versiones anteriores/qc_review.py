#!/usr/bin/env python3
"""
qc_review_gui.py – Revisor gráfico de discrepancias texto ↔ audio
Autor: 2025 – MIT
"""

import json, subprocess, shutil, tempfile, webbrowser
from pathlib import Path
from tkinter import Tk, ttk, filedialog, messagebox, scrolledtext, StringVar, BOTH, END
import difflib, re

import pandas as pd
from rapidfuzz.distance import Levenshtein       # pip install rapidfuzz
# ─────────────────────────── parámetros ────────────────────────────
DUP_RE = re.compile(r"\b(\w+)\s+\1\b", flags=re.I)

# ──────────────────────────── helpers ──────────────────────────────
def load_json(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf8"))
    df = pd.DataFrame(data)
    df["dur"] = df["dur"].astype(float)
    df["start"] = df["dur"].cumsum().shift(fill_value=0)
    df["WER%"] = df.apply(lambda r: round(Levenshtein.normalized_distance(r.Original, r.ASR)*100, 1), axis=1)
    df["dup?"] = df.ASR.str.contains(DUP_RE)
    return df

def diff_html(a: str, b: str) -> str:
    """Devuelve string con <span style='color:red/green'> para mostrar en navegador."""
    seq = difflib.SequenceMatcher(None, a.split(), b.split())
    html = []
    for op,i1,i2,j1,j2 in seq.get_opcodes():
        if op == "equal":
            html.extend(seq.a[i1:i2])
        elif op == "delete":
            html.extend(f"<span style='color:#d33'>{w}</span>" for w in seq.a[i1:i2])
        elif op == "insert":
            html.extend(f"<span style='color:#080'>{w}</span>" for w in seq.b[j1:j2])
        elif op == "replace":
            html.extend(f"<span style='color:#d33'>{w}</span>" for w in seq.a[i1:i2])
            html.extend(f"<span style='color:#080'>{w}</span>" for w in seq.b[j1:j2])
    return " ".join(html)

def export_clips(wav: Path, df: pd.DataFrame, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for _, r in df[df["✓"]!="✅"].iterrows():
        out = folder / f"{r.ID:04d}_{r['✓']}.wav"
        cmd = ["ffmpeg","-y","-i",str(wav),
               "-ss",f"{r.start:.2f}","-t",f"{r.dur:.2f}",
               "-acodec","copy",str(out)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    messagebox.showinfo("Listo", f"Clips guardados en\n{folder}")

# ──────────────────────────── GUI ─────────────────────────────────
class Reviewer:
    def __init__(self, root: Tk):
        self.root = root
        root.title("QC-Audiolibro – Revisor")
        root.geometry("1080x640")

        self.json_path = StringVar()
        self.audio_path = StringVar()
        self.df = None

        # barra superior
        frm = ttk.Frame(root, padding=6)
        frm.pack(fill='x')
        ttk.Label(frm, text="JSON:").pack(side='left')
        ttk.Entry(frm, textvariable=self.json_path, width=60).pack(side='left', padx=2)
        ttk.Button(frm, text="Abrir…", command=self.open_json).pack(side='left')
        ttk.Label(frm, text="Audio (opcional):").pack(side='left', padx=(20,0))
        ttk.Entry(frm, textvariable=self.audio_path, width=45).pack(side='left')
        ttk.Button(frm, text="Seleccionar…", command=self.open_audio).pack(side='left')
        ttk.Button(frm, text="Generar clips", command=self.make_clips).pack(side='right')

        # tabla
        cols = ("ID","✓","WER","dup","Original","ASR")
        self.tree = ttk.Treeview(root, columns=cols, show='headings', height=18)
        for c,w in zip(cols,(50,40,60,40,400,400)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor='w')
        self.tree.pack(fill=BOTH, expand=True, padx=6, pady=4)
        self.tree.bind("<<TreeviewSelect>>", self.show_diff)

        # diff viewer
        self.txt = scrolledtext.ScrolledText(root, height=8, font=("Consolas",10))
        self.txt.pack(fill=BOTH, expand=False, padx=6, pady=(0,6))
        self.txt.tag_configure("del", foreground="#d33")
        self.txt.tag_configure("add", foreground="#080")

    # ───── eventos ─────
    def open_json(self):
        p = filedialog.askopenfilename(filetypes=(("JSON","*.json"),))
        if not p: return
        self.json_path.set(p)
        self.df = load_json(Path(p))
        self.populate()

    def populate(self):
        self.tree.delete(*self.tree.get_children())
        for _, r in self.df.iterrows():
            self.tree.insert("", END, values=(r.ID, r["✓"], r["WER%"],
                                              "🔁" if r["dup?"] else "",
                                              r.Original[:120], r.ASR[:120]))

    def open_audio(self):
        p = filedialog.askopenfilename(filetypes=(("Audio","*.wav;*.mp3;*.flac;*.m4a;*.ogg"),))
        if p: self.audio_path.set(p)

    def show_diff(self, _):
        sel = self.tree.selection()
        if not sel: return
        row = self.tree.item(sel[0])["values"][0]   # ID
        r = self.df.loc[self.df.ID==row].iloc[0]
        html = diff_html(r.Original, r.ASR)
        # como Text no renderiza HTML, abrimos en navegador local
        tmp = Path(tempfile.gettempdir())/f"diff_{row}.html"
        tmp.write_text(f"<meta charset='utf8'><body style='font-family:sans-serif'>"
                       f"<h3>ID {row} – dur {r.dur:.2f}s – WER {r['WER%']}%</h3>"
                       f"<p>{html}</p></body>", encoding="utf8")
        webbrowser.open(tmp.as_uri())

    def make_clips(self):
        if self.df is None:
            messagebox.showwarning("Sin datos","Carga primero un JSON.")
            return
        if not self.audio_path.get():
            messagebox.showwarning("Falta audio","Selecciona el archivo de audio original.")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror("FFmpeg","Agrega FFmpeg al PATH para exportar clips.")
            return
        outdir = Path(filedialog.askdirectory(title="Carpeta destino de clips") or "")
        if not outdir: return
        export_clips(Path(self.audio_path.get()), self.df, outdir)

# ──────────────────────────── run ─────────────────────────────────
if __name__ == "__main__":
    root = Tk()
    Reviewer(root)
    root.mainloop()

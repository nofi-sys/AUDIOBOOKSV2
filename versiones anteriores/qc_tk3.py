#!/usr/bin/env python
# qc_tk2.py – QC audiolibros (Tkinter) – 2025 • MIT

###############################################################################
#  IMPORTS & SMALL UTILS
###############################################################################

import io, json, math, queue, threading, traceback, tempfile, subprocess
import re, sys
from pathlib import Path
from difflib import SequenceMatcher

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

try:
    from rapidfuzz.distance import Levenshtein
    _RF = True
except ImportError:
    _RF = False

import unidecode, pandas as pd

# ---------- normalización ----------------------------------------------------
_DIG = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()
def norm(txt: str) -> str:
    txt = unidecode.unidecode(txt.lower())
    txt = re.sub(r"\b([0-9])\b", lambda m: _DIG[int(m.group(1))], txt)
    txt = re.sub(r"[^\w\s]", " ", txt)          # sin puntuación
    return re.sub(r"\s+", " ", txt).strip()

###############################################################################
#  LECTURA DE SCRIPT (PDF / TXT)
###############################################################################
def _clean_lines(raw: str) -> str:
    # quedate solo con líneas que contengan ≥4 letras seguidas
    good = [ln.strip() for ln in raw.splitlines()
            if re.search(r"[a-záéíóúüñ]{4}", ln, re.I)]
    return "\n".join(good)

def _from_pdfplumber(p: Path) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            return _clean_lines(
                "\n".join(page.extract_text() or "" for page in pdf.pages))
    except Exception:
        return ""

def _from_pymupdf(p: Path) -> str:
    try:
        import fitz
    except ImportError:
        return ""
    out = []
    with fitz.open(p) as doc:
        for page in doc:
            out.append(page.get_text("text"))
    return _clean_lines("\n".join(out))

def _from_ocr(p: Path) -> str:
    try:
        import fitz, pytesseract, PIL.Image
    except ImportError:
        return ""
    out = []
    with fitz.open(p) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang="spa+eng"))
    return _clean_lines("\n".join(out))

def read_script(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() != ".pdf":
        return Path(path).read_text(encoding="utf8", errors="ignore")

    for extractor in (_from_pdfplumber, _from_pymupdf, _from_ocr):
        txt = extractor(p)
        if len(txt.strip()) > 50:
            return txt
    raise RuntimeError("No se pudo extraer texto del PDF; usa un TXT plano.")

###############################################################################
#  ALINEADO ROBUSTO
###############################################################################
STOP = set("de la y el los las un una en que a por con del se para es".split())

def smart_pairs(a_words, b_words):
    sm = SequenceMatcher(a=a_words, b=b_words, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                if a_words[i1+k] not in STOP:
                    pairs.append((i1+k, j1+k))
    return pairs

def chunked_rows(words, size=12):
    rows=[]
    for i in range(0, len(words), size):
        seg=" ".join(words[i:i+size])
        rows.append(seg)
    return rows

def make_blocks(ref_raw, hyp_raw, chunk_size=12):
    ref = norm(ref_raw).split()
    hyp = norm(hyp_raw).split()

    pairs = smart_pairs(ref, hyp)
    if not pairs:
        # fallback: trocear simplemente
        ref_chunks = chunked_rows(ref, chunk_size)
        hyp_chunks = chunked_rows(hyp, chunk_size)
        return _rows_from_chunks(ref_chunks, hyp_chunks)

    rmap = [-1]*len(ref); hmap = [-1]*len(hyp)
    bid  = 0
    for i,j in pairs:
        if rmap[i]==hmap[j]==-1:
            rmap[i]=hmap[j]=bid; bid+=1

    rows=[]
    for b in range(bid):
        r_seg=" ".join(w for w,m in zip(ref,rmap) if m==b)
        h_seg=" ".join(w for w,m in zip(hyp,hmap) if m==b)
        rows.append(_row(b,r_seg,h_seg))
    return rows

def _row(idx, r_seg, h_seg):
    if _RF:
        wer = Levenshtein.normalized_distance(r_seg,h_seg)*100
    else:
        wer=(1-SequenceMatcher(None,r_seg,h_seg).ratio())*100
    dur=len(h_seg.split())/3.0
    mark="✅" if wer<10 else ("⚠️" if wer<35 else "❌")
    return [idx,mark,round(wer,1),round(dur,2),r_seg,h_seg]

def _rows_from_chunks(ref_chunks,hyp_chunks):
    rows=[]
    for idx,(r,h) in enumerate(zip(ref_chunks,hyp_chunks)):
        rows.append(_row(idx,r,h))
    return rows

###############################################################################
#  GUI: Tkinter
###############################################################################
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro – Revisor")
        self.geometry("1280x720")

        self.v_audio=tk.StringVar(); self.v_script=tk.StringVar(); self.v_txt=tk.StringVar()
        top=ttk.Frame(self,padding=4); top.pack(fill="x")

        def add_row(lbl,var,cmd,r):
            ttk.Label(top,text=lbl).grid(row=r,column=0,sticky="e")
            ttk.Entry(top,textvariable=var,width=95).grid(row=r,column=1)
            ttk.Button(top,text="…",command=cmd).grid(row=r,column=2)

        add_row("Audio", self.v_audio,  self.pick_audio,  0)
        add_row("Guion", self.v_script, self.pick_script, 1)
        add_row("TXT  ", self.v_txt,    self.pick_txt,    2)

        ttk.Button(top,text="▶ Procesar",width=16,command=self.start
                  ).grid(row=0,column=3,rowspan=3,padx=8)

        cols=("ID","✓","WER","dur","Original","ASR")
        self.tree=ttk.Treeview(self,columns=cols,show="headings",height=18)
        for c,w in zip(cols,(40,30,60,60,640,640)):
            self.tree.heading(c,text=c); self.tree.column(c,width=w,anchor="w")
        self.tree.pack(fill="both",expand=True)

        self.log=scrolledtext.ScrolledText(self,height=6,state="disabled")
        self.log.pack(fill="x",padx=4,pady=3)

        self.queue=queue.Queue(); self.after(200,self._tick)

    # file pickers ------------------------------------------------------------
    def pick_audio(self):
        f=filedialog.askopenfilename(filetypes=[("Audio","*.mp3;*.wav;*.m4a;*.mp4")])
        if f:self.v_audio.set(f)
    def pick_script(self):
        f=filedialog.askopenfilename(filetypes=[("Guion","*.txt;*.pdf")])
        if f:self.v_script.set(f)
    def pick_txt(self):
        f=filedialog.askopenfilename(filetypes=[("TXT","*.txt")])
        if f:self.v_txt.set(f)

    # worker ------------------------------------------------------------------
    def start(self):
        if not self.v_audio.get() or not self.v_script.get():
            messagebox.showwarning("Falta info","Selecciona audio y guion.")
            return
        self.tree.delete(*self.tree.get_children()); self._log_clear()
        threading.Thread(target=self.worker,daemon=True).start()

    def worker(self):
        q=self.queue.put
        try:
            q("→ Preparando audio…")  # placeholder
            q("→ Leyendo guion…")
            ref=read_script(self.v_script.get())

            if self.v_txt.get():
                q("→ TXT externo…")
                hyp=Path(self.v_txt.get()).read_text(encoding="utf8")
            else:
                q("→ (demo) usamos guion como hipótesis")
                hyp=ref

            rows=make_blocks(ref,hyp)
            out=Path(self.v_audio.get()).with_suffix(".qc.json")
            pd.DataFrame(rows,columns=["ID","✓","WER","dur","Original","ASR"]
                        ).to_json(out,orient="records",force_ascii=False,indent=2)
            q(f"✔ JSON guardado → {out}")
            q(rows)
        except Exception:
            buf=io.StringIO(); traceback.print_exc(file=buf); q(buf.getvalue())

    # cola → GUI --------------------------------------------------------------
    def _tick(self):
        try:
            while True:
                msg=self.queue.get_nowait()
                if isinstance(msg,str):
                    self._log(msg)
                else:
                    for r in msg:self.tree.insert("",tk.END,values=r)
        except queue.Empty:
            pass
        self.after(200,self._tick)

    # log helpers -------------------------------------------------------------
    def _log(self,txt):
        self.log.configure(state="normal")
        self.log.insert("end",txt+"\n"); self.log.configure(state="disabled")
        self.log.yview("end")
    def _log_clear(self):
        self.log.configure(state="normal"); self.log.delete("1.0","end")
        self.log.configure(state="disabled")

# main ------------------------------------------------------------------------
if __name__=="__main__":
    App().mainloop()

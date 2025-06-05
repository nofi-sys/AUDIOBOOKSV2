#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_tk.py – Revisor Audiolibros (líneas de 12 palabras, DTW robusto)
MIT • 2025
"""

import threading, queue, tempfile, subprocess, json, re, io, traceback, sys, os
from pathlib import Path
from typing import List, Tuple, Dict

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import unidecode, pdfplumber
from rapidfuzz.distance import Levenshtein

# ---------- parámetros ----------
LINE_LEN   = 12        # palabras por fila en la tabla
COARSE_W   = 40        # ventana DTW gruesa
FINE_W     = 8         # ventana DTW fina (subalineación)
WARN_WER   = 0.08      # > 8 % → ⚠️

SPANISH_STOP = {
    # lista mínima; amplíala si quieres
    "de","la","el","y","que","en","a","los","se","del","por","con","las","un",
    "para","una","su","al","lo","como","más","o","pero","sus","le","ya","fue",
}

# ---------- utilidades texto ----------
def normalize(txt:str)->str:
    txt = unidecode.unidecode(txt.lower())
    txt = re.sub(r"[^\w\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def read_script(path:str)->str:
    path = Path(path)
    if path.suffix.lower()==".pdf":
        with pdfplumber.open(path) as pdf:
            raw = "\n".join(p.extract_text() or "" for p in pdf.pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa un TXT.")
        return raw
    return path.read_text(encoding="utf8", errors="ignore")

# ---------- DTW con ventana ----------
def dtw_band(a:List[str], b:List[str], w:int)->List[Tuple[int,int]]:
    n, m   = len(a), len(b)
    W      = max(w, abs(n-m))
    D: Dict[Tuple[int,int], Tuple[int,int]]  = {}
    back: Dict[Tuple[int,int], Tuple[int,int]] = {}

    for i in range(-1, n):
        for j in range(-1, m):
            D[(i, j)] = (10**9, 1)    # (coste acumulado, dist local)
    D[(-1, -1)] = (0, 0)

    for i in range(n):
        for j in range(max(0, i-W), min(m, i+W+1)):
            cost = 0 if a[i]==b[j] else 1
            best, prev = min(
                (D[(i-1, j  )][0]+1,   (i-1, j  )),
                (D[(i  , j-1)][0]+1,   (i  , j-1)),
                (D[(i-1, j-1)][0]+cost,(i-1, j-1)),
                key=lambda x:x[0]
            )
            D[(i, j)]    = (best, cost)
            back[(i, j)] = prev

    if (n-1, m-1) not in back:
        raise RuntimeError("sin camino DTW")

    path=[]
    i,j = n-1, m-1
    while (i,j)!=(-1,-1):
        path.append((i,j))
        i,j = back[(i,j)]
    return path[::-1]

def safe_dtw(a,b,w)->List[Tuple[int,int]]:
    try:
        return dtw_band(a,b,w)
    except RuntimeError:
        # sin límite (pleno) – más lento pero garantiza alineación
        return dtw_band(a,b,max(len(a),len(b)))

# ---------- generación filas ----------
def make_lines(tokens:List[str], length:int)->List[str]:
    out,cur=[],[]
    for t in tokens:
        cur.append(t)
        if len(cur)>=length:
            out.append(" ".join(cur)); cur=[]
    if cur: out.append(" ".join(cur))
    return out

def build_rows(ref:str, hyp:str) -> List[List]:
    ref_tok = normalize(ref).split()
    hyp_tok = normalize(hyp).split()

    # alineación “rápida” sobre tokens sin stop-words
    ref_sw = [t for t in ref_tok if t not in SPANISH_STOP]
    hyp_sw = [t for t in hyp_tok if t not in SPANISH_STOP]
    coarse = safe_dtw(ref_sw, hyp_sw, COARSE_W)

    # reconstruir mapa IDX completo (palabra→palabra)
    # muy simple: para cada palabra “clave” alineada, igualamos posiciones
    # y usamos desplazamiento local ±FINE_W para rellenar huecos
    map_h = [-1]*len(ref_tok)
    ptr_ref = ptr_hyp = 0
    sw_i = [i for i,t in enumerate(ref_tok) if t not in SPANISH_STOP]
    sw_j = [j for j,t in enumerate(hyp_tok) if t not in SPANISH_STOP]
    for (ci,cj) in coarse:
        map_h[ sw_i[ci] ] = sw_j[cj]

    # huecos: rellenar hacia delante si la distancia es razonable
    last=-1
    for i in range(len(map_h)):
        if map_h[i]!=-1:
            last=map_h[i]
        else:
            last+=1
            if last<len(hyp_tok):
                map_h[i]=last

    # dividir guion en líneas de 12 palabras
    ref_lines = make_lines(ref_tok, LINE_LEN)
    rows=[]
    idx=0
    for lid,line in enumerate(ref_lines):
        words=line.split()
        span=(idx, idx+len(words)-1)
        idx_h=[map_h[k] for k in range(*span,1)]
        # texto ASR correspondiente
        if idx_h:
            s=min(idx_h); e=max(idx_h)+1
            asr_line=" ".join(hyp_tok[s:e])
        else:
            asr_line=""

        wer=Levenshtein.normalized_distance(line,asr_line)
        flag="✅" if wer<=WARN_WER else ("⚠️" if wer<=0.20 else "❌")
        dur=round(len(asr_line.split())/3,2)
        rows.append([lid,flag,wer*100,dur,line,asr_line])
        idx+=len(words)
    return rows

# ---------- GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro – Revisor")
        self.geometry("1850x750")

        # vars
        self.v_ref  = tk.StringVar()
        self.v_asr  = tk.StringVar()
        self.queue  = queue.Queue()

        top = ttk.Frame(self); top.pack(fill="x", pady=2, padx=2)
        ttk.Label(top,text="Guion:").grid(row=0,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_ref,width=90).grid(row=0,column=1)
        ttk.Button(top,text="…",command=self.sel_ref,width=3).grid(row=0,column=2)

        ttk.Label(top,text="TXT ASR:").grid(row=1,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_asr,width=90).grid(row=1,column=1)
        ttk.Button(top,text="…",command=self.sel_asr,width=3).grid(row=1,column=2)

        ttk.Button(top,text="Procesar",command=self.launch,width=12).grid(row=0,column=3,rowspan=2,padx=10)

        # tabla
        cols=("ID","✓","WER","dur","Original","ASR")
        self.tree=ttk.Treeview(self,columns=cols,show="headings",height=25)
        for c,w in zip(cols,(50,30,60,60,800,800)):
            self.tree.heading(c,text=c)
            self.tree.column(c,width=w,anchor="w")
        self.tree.pack(fill="both",expand=True,padx=2,pady=2)

        self.log=scrolledtext.ScrolledText(self,height=5,state="disabled")
        self.log.pack(fill="x",padx=2,pady=2)

        self.after(200,self.poll)

    # --- helpers GUI
    def sel_ref(self):  self._sel(self.v_ref, [("PDF/TXT","*.pdf;*.txt")])
    def sel_asr(self):  self._sel(self.v_asr, [("TXT","*.txt")])
    def _sel(self,var,ft):
        p=filedialog.askopenfilename(filetypes=ft)
        if p: var.set(p)

    def launch(self):
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info","Selecciona guion y TXT ASR.")
            return
        self.tree.delete(*self.tree.get_children())
        self.log_clear()
        threading.Thread(target=self.worker,daemon=True).start()

    def worker(self):
        try:
            self.queue.put("→ Leyendo guion…")
            ref=read_script(self.v_ref.get())

            self.queue.put("→ TXT externo cargado")
            hyp=Path(self.v_asr.get()).read_text(encoding="utf8",errors="ignore")

            self.queue.put("→ Alineando…")
            rows=build_rows(ref,hyp)

            self.queue.put(("ROWS",rows))
            self.queue.put("✔ Listo.")
        except Exception as e:
            buf=io.StringIO(); traceback.print_exc(file=buf)
            self.queue.put(buf.getvalue())

    # --- comunicación hilo→GUI
    def poll(self):
        try:
            while True:
                msg=self.queue.get_nowait()
                if isinstance(msg,tuple) and msg[0]=="ROWS":
                    for row in msg[1]:
                        self.tree.insert("",tk.END,values=row)
                else:
                    self.log_print(str(msg))
        except queue.Empty:
            pass
        self.after(200,self.poll)

    def log_print(self,msg):
        self.log.configure(state="normal")
        self.log.insert("end",msg+"\n")
        self.log.configure(state="disabled")
        self.log.see("end")
    def log_clear(self):
        self.log.configure(state="normal"); self.log.delete("1.0","end"); self.log.configure(state="disabled")

# ---------- main ----------
if __name__=="__main__":
    App().mainloop()

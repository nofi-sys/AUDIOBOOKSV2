#!/usr/bin/env python
"""
qc_tk.py – QC audiolibros • GUI Tkinter
Autor 2025 • MIT
"""

import threading, queue, tempfile, subprocess, io, traceback, re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import torch, whisperx, unidecode, pdfplumber, pandas as pd
from rapidfuzz.distance import Levenshtein

# ───────────────────────── utilidades ─────────────────────────
DIGITS = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()

def normalize(t:str)->str:
    t = unidecode.unidecode(t.lower())
    t = re.sub(r"\b([0-9])\b", lambda m: DIGITS[int(m.group(1))], t)
    t = re.sub(r"[^a-z0-9\s\.\,\!\?\:;¿¡]", "", t)
    return re.sub(r"\s+", " ", t).strip()

def segment(txt:str, max_words=12):
    words, out, cur = txt.split(), [], []
    for w in words:
        cur.append(w)
        if len(cur)>=max_words or w.endswith((".", "!", "?")):
            out.append(" ".join(cur)); cur=[]
    if cur: out.append(" ".join(cur))
    return out

def ffmpeg_wav(src:str)->str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg","-y","-i",src,"-ar","16000","-ac","1","-vn",tmp],
                   check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    return tmp

def read_text(p:str)->str:
    if p.lower().endswith(".pdf"):
        with pdfplumber.open(p) as pdf:
            return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    return Path(p).read_text(encoding="utf8", errors="ignore")

# ───────────────────── alineación global ─────────────────────
def sim(a,b): return 1-Levenshtein.normalized_distance(a,b)

def align(orig, hyp, gap=0.8):
    n,m=len(orig),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    bt=[[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=dp[i-1][0]+gap; bt[i][0]=(i-1,0)
    for j in range(1,m+1): dp[0][j]=dp[0][j-1]+gap; bt[0][j]=(0,j-1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            d=dp[i-1][j-1]+(1-sim(orig[i-1],hyp[j-1]))
            u=dp[i-1][j]+gap
            l=dp[i][j-1]+gap
            dp[i][j],bt[i][j]=min((d,(i-1,j-1)),(u,(i-1,j)),(l,(i,j-1)),key=lambda x:x[0])
    out,i,j=[],n,m
    while i or j:
        pi,pj=bt[i][j]
        if pi==i-1 and pj==j-1: out.append((i-1,j-1))
        elif pi==i-1:           out.append((i-1,None))
        else:                   out.append((None,j-1))
        i,j=pi,pj
    return list(reversed(out))

def build_df(o,h):
    pairs=align(o,h)
    rows=[]
    for idx,(oi,hj) in enumerate(pairs):
        o_txt=o[oi] if oi is not None else ""
        h_txt=h[hj] if hj is not None else ""
        wer=Levenshtein.normalized_distance(o_txt,h_txt) if o_txt and h_txt else 1.0
        flag="✅" if wer<=.05 else ("⚠️" if wer<=.30 else "❌")
        dur=len(h_txt.split())/3.0 if h_txt else 0.0        # ≈ segundos
        rows.append([idx,flag,round(wer*100,1),f"{dur:.2f}",o_txt,h_txt])
    return pd.DataFrame(rows,columns=["ID","✓","WER%","dur","Original","ASR"])

# ─────────────────────── worker thread ───────────────────────
def worker(audio,text,txt_asr,model,q):
    try:
        device="cuda" if torch.cuda.is_available() else "cpu"
        q.put("→ Convirtiendo audio…"); wav=ffmpeg_wav(audio)
        q.put("→ Procesando guion…");   ref=segment(normalize(read_text(text)))

        if txt_asr:
            q.put("→ Usando TXT transcrito…")
            hyp_full=normalize(Path(txt_asr).read_text(encoding="utf8"))
        else:
            q.put(f"→ WhisperX-{model} … (paciencia)")
            wx=whisperx.load_model(model,device)
            segs=wx.transcribe(wav,batch_size=16)["segments"]
            mdl,meta=whisperx.load_align_model("es",device)
            segs=whisperx.align(segs,mdl,meta,wav,device)
            hyp_full=normalize(" ".join(s["text"] for s in segs))

        hyp=segment(hyp_full,12)
        q.put("→ Alineando…"); df=build_df(ref,hyp)

        out=Path(audio).with_suffix(".qc.json")
        df.to_json(out,orient="records",force_ascii=False,indent=2)
        q.put(f"✔ Terminado → {out}")
    except Exception:
        buf=io.StringIO(); traceback.print_exc(file=buf)
        q.put("⚠️ ERROR:\n"+buf.getvalue())

# ─────────────────────────── GUI ────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro – Revisor")
        self.resizable(False,False)

        self.v_a=tk.StringVar(); self.v_t=tk.StringVar()
        self.v_x=tk.StringVar(); self.v_m=tk.StringVar(value="small")
        self.q=queue.Queue()

        frm=ttk.Frame(self,padding=10); frm.grid()
        def row(r,l,var,cmd):
            ttk.Label(frm,text=l).grid(row=r,column=0,sticky="e")
            ttk.Entry(frm,textvariable=var,width=58).grid(row=r,column=1)
            ttk.Button(frm,text="…",command=cmd,width=3).grid(row=r,column=2)
        row(0,"Audio", self.v_a, self.open_a)
        row(1,"Guion PDF/TXT", self.v_t, self.open_t)
        row(2,"TXT opc.", self.v_x, self.open_x)

        ttk.Label(frm,text="Modelo").grid(row=3,column=0,sticky="e")
        ttk.Combobox(frm,textvariable=self.v_m,values=("tiny","base","small","medium","large"),
                     width=8,state="readonly").grid(row=3,column=1,sticky="w")
        ttk.Button(frm,text="▶ Ejecutar",command=self.run,width=15)\
            .grid(row=3,column=2,pady=4)

        self.log=scrolledtext.ScrolledText(frm,width=90,height=20,state="disabled",font=("Consolas",9))
        self.log.grid(row=4,column=0,columnspan=3,pady=6)

        self.after(200,self.poll)

    def open_a(self): self._pick(self.v_a,"*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.aac;*.mp4")
    def open_t(self): self._pick(self.v_t,"*.pdf;*.txt")
    def open_x(self): self._pick(self.v_x,"*.txt")
    def _pick(self,var,types):
        f=filedialog.askopenfilename(filetypes=[("Archivos",types)])
        if f: var.set(f)

    def run(self):
        if not self.v_a.get() or not self.v_t.get():
            messagebox.showwarning("Falta info","Selecciona audio y guion.")
            return
        self._clr()
        threading.Thread(target=worker,args=(self.v_a.get(),self.v_t.get(),
                     self.v_x.get() or None,self.v_m.get(),self.q),daemon=True).start()

    def poll(self):
        try:
            while True: self._print(self.q.get_nowait())
        except queue.Empty: pass
        self.after(200,self.poll)

    def _print(self,msg):
        self.log.config(state="normal"); self.log.insert("end",msg+"\n")
        self.log.config(state="disabled"); self.log.yview("end")
    def _clr(self):
        self.log.config(state="normal"); self.log.delete("1.0","end")
        self.log.config(state="disabled")

# ─────────────────────────── main ───────────────────────────
if __name__=="__main__":
    App().mainloop()

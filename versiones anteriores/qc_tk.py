#!/usr/bin/env python
"""
qc_tk.py – generación de QC JSON con alineación “inteligente”
Autor 2025 • MIT
Requiere: torch, whisperx, pandas, rapidfuzz, unidecode, pdfplumber
"""

import re, io, json, queue, threading, subprocess, tempfile, traceback, sys
from pathlib import Path

import torch, whisperx, pandas as pd, pdfplumber, unidecode
from rapidfuzz.distance import Levenshtein

# ───── normalizado ────────────────────────────────────────────
DIGITS = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()
ABBR = {
    "dra": "doctora", "dr": "doctor",
    "sra": "señora",  "sr": "señor",
    "av": "avenida",  "kg": "kilos",
    "bsas": "buenos aires",
}

def expand_abbr(word: str) -> str:
    w = word.rstrip(".")
    return ABBR.get(w, w)

def normalize(text: str) -> str:
    txt = unidecode.unidecode(text.lower())
    txt = re.sub(r"\b(\d)\b", lambda m: DIGITS[int(m.group(1))], txt)
    txt = re.sub(r"[^a-z0-9¿¡\?\!\.,:\s]", " ", txt)
    txt = " ".join(expand_abbr(w) for w in txt.split())
    return txt

# ───── segmentador “sentencias cortas” (ignora abreviaturas) ─
ABBR_RE = r"(?<!\b(?:dr|dra|sr|sra|mr|mrs|av|etc)\.)"

def smart_segment(txt: str, max_words=15):
    # 1) cortar en “puntos no abreviatura”
    chunks = re.split(ABBR_RE + r"\s*[\.!\?]+\s*", txt)
    out, cur = [], []
    for ch in chunks:
        for w in ch.split():
            cur.append(w)
            if len(cur) >= max_words:
                out.append(" ".join(cur)); cur=[]
        if ch.endswith((".", "!", "?")) and cur:
            out.append(" ".join(cur)); cur=[]
    if cur: out.append(" ".join(cur))
    return out

# ───── FFMPEG helper ──────────────────────────────────────────
def ffmpeg_wav(src: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg","-y","-i",src,"-ar","16000","-ac","1","-vn",tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp

def read_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower()==".pdf":
        with pdfplumber.open(p) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    return p.read_text(encoding="utf8", errors="ignore")

# ───── alineación dinámica de frases ──────────────────────────
def align_blocks(ref, hyp, win=2):
    i=j=0; pairs=[]
    while i<len(ref) and j<len(hyp):
        best=(999, 1,1)   # wer, di, dj
        for di in range(1,win+1):
            for dj in range(1,win+1):
                if i+di<=len(ref) and j+dj<=len(hyp):
                    r=" ".join(ref[i:i+di]); h=" ".join(hyp[j:j+dj])
                    wer=Levenshtein.normalized_distance(r,h)
                    if wer<best[0]:
                        best=(wer,di,dj)
        wer,di,dj=best
        pairs.append( ( " ".join(ref[i:i+di]),  " ".join(hyp[j:j+dj]), wer) )
        i+=di; j+=dj
    # arrastre de sobrantes
    for r in ref[i:]:
        pairs.append( (r,"",1.0) )
    for h in hyp[j:]:
        pairs.append( ("",h,1.0) )
    return pairs

# ───── clasificación ─────────────────────────────────────────
def classify(wer: float, dur: float):
    if not dur or dur<0.2:   return "❌"
    if wer  <=0.06:          return "✅"
    if wer  <=0.35:          return "⚠️"
    return "❌"

# ───── worker hilo ───────────────────────────────────────────
def build_json(audio, script, txt_asr, model, q):
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        q.put("→ Convirtiendo audio…")
        wav = ffmpeg_wav(audio)

        q.put("→ Analizando guion…")
        ref_sent = smart_segment(normalize(read_text(script)))

        # — ASR —
        if txt_asr:
            q.put("→ Leyendo TXT externo…")
            hyp_full = normalize(Path(txt_asr).read_text(encoding="utf8"))
        else:
            q.put("→ Transcribiendo con WhisperX… (paciencia)")
            wx = whisperx.load_model(model, dev)
            segs = wx.transcribe(wav, batch_size=16)["segments"]
            m_align, meta = whisperx.load_align_model("es", dev)
            segs = whisperx.align(segs, m_align, meta, wav, dev)
            hyp_full = normalize(" ".join(s["text"] for s in segs))

        hyp_sent = smart_segment(hyp_full)

        q.put("→ Alineando bloques…")
        pairs = align_blocks(ref_sent, hyp_sent, win=2)

        rows=[]
        for idx,(r,h,wer) in enumerate(pairs):
            dur = len(h.split())/3 if h else 0
            rows.append([idx, classify(wer,dur), round(wer*100,1), f"{dur:.2f}", r, h])

        df = pd.DataFrame(rows, columns=["ID","✓","WER","dur","Original","ASR"])
        out = Path(audio).with_suffix(".qc.json")
        df.to_json(out, orient="records", force_ascii=False, indent=2)
        q.put(f"✔ JSON generado: {out}")
    except Exception:
        buf=io.StringIO(); traceback.print_exc(file=buf)
        q.put(buf.getvalue())

# ───────────────────────── GUI simple ─────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro • Generar JSON")
        self.resizable(False,False)

        v_a,v_s,v_x,v_m = (tk.StringVar() for _ in range(4))
        v_m.set("small")
        self.vars = v_a,v_s,v_x,v_m
        self.q=queue.Queue()

        frm=ttk.Frame(self,padding=10); frm.grid()
        def row(r,l,var,cmd):
            ttk.Label(frm,text=l).grid(row=r,column=0,sticky="e")
            ttk.Entry(frm,textvariable=var,width=60).grid(row=r,column=1)
            ttk.Button(frm,text="…",command=cmd,width=3).grid(row=r,column=2)
        row(0,"Audio",v_a, lambda:self.sel(v_a,"*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.aac;*.mp4"))
        row(1,"Guion PDF/TXT",v_s, lambda:self.sel(v_s,"*.pdf;*.txt"))
        row(2,"TXT opc.",v_x, lambda:self.sel(v_x,"*.txt"))

        ttk.Label(frm,text="Modelo").grid(row=3,column=0,sticky="e")
        ttk.Combobox(frm,textvariable=v_m,values=("tiny","base","small","medium","large"),
                     width=8,state="readonly").grid(row=3,column=1,sticky="w")
        ttk.Button(frm,text="▶ Ejecutar",command=self.run,width=15)\
            .grid(row=3,column=2,pady=4)

        self.log=scrolledtext.ScrolledText(frm,width=90,height=18,state="disabled",font=("Consolas",9))
        self.log.grid(row=4,column=0,columnspan=3,pady=6)

        self.after(200,self.poll)

    def sel(self,var,types):
        f=filedialog.askopenfilename(filetypes=[("Archivos",types)])
        if f: var.set(f)

    def run(self):
        v_a,v_s,v_x,v_m=self.vars
        if not v_a.get() or not v_s.get():
            messagebox.showwarning("Falta info","Selecciona audio y guion."); return
        self.clr()
        threading.Thread(target=worker,args=(v_a.get(),v_s.get(),
                        v_x.get() or None,v_m.get(),self.q),daemon=True).start()

    def poll(self):
        try:
            while True: self.out(self.q.get_nowait())
        except queue.Empty: pass
        self.after(200,self.poll)

    def out(self,msg):
        self.log.config(state="normal"); self.log.insert("end",msg+"\n")
        self.log.config(state="disabled"); self.log.yview("end")
    def clr(self):
        self.log.config(state="normal"); self.log.delete("1.0","end")
        self.log.config(state="disabled")

if __name__=="__main__":
    App().mainloop()

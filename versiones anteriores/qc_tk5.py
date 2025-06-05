#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_tk.py  –  Revisor Audiolibros   v4.4
• líneas de 12 palabras
• guarda / abre JSON   (.qc.json)
• corrige la pérdida de la última palabra en la columna ASR
MIT • 2025
"""
import threading, queue, json, re, io, traceback
from pathlib import Path
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import unidecode, pdfplumber
from rapidfuzz.distance import Levenshtein

# ---------- parámetros ----------
LINE_LEN   = 12
COARSE_W   = 40
WARN_WER   = 0.08
STOP = {
    "de","la","el","y","que","en","a","los","se","del","por","con","las",
    "un","para","una","su","al","lo","como","más","o","pero","sus","le",
    "ya","fue"
}

# ---------- utilidades texto ----------
def normalize(t:str)->str:
    t = unidecode.unidecode(t.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def read_script(p:str)->str:
    p = Path(p)
    if p.suffix.lower()==".pdf":
        with pdfplumber.open(p) as pdf:
            raw="\n".join(pg.extract_text() or "" for pg in pdf.pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa TXT.")
        return raw
    return p.read_text(encoding="utf8",errors="ignore")

# ---------- DTW ventana ----------
def dtw_band(a:List[str],b:List[str],w:int)->List[Tuple[int,int]]:
    n,m=len(a),len(b); W=max(w,abs(n-m))
    D:Dict[Tuple[int,int],Tuple[int,int]]={}
    back={}
    BIG=10**9
    for i in range(-1,n):
        for j in range(-1,m):
            D[(i,j)]=(BIG,1)
    D[(-1,-1)]=(0,0)
    for i in range(n):
        for j in range(max(0,i-W),min(m,i+W+1)):
            cost=0 if a[i]==b[j] else 1
            best,pr=min( (D[(i-1,j)][0]+1,(i-1,j)),
                         (D[(i,j-1)][0]+1,(i,j-1)),
                         (D[(i-1,j-1)][0]+cost,(i-1,j-1)),
                         key=lambda x:x[0])
            D[(i,j)]=(best,cost); back[(i,j)]=pr
    if (n-1,m-1) not in back: raise RuntimeError
    path=[]; i,j=n-1,m-1
    while (i,j)!=(-1,-1):
        path.append((i,j)); i,j=back[(i,j)]
    return path[::-1]

def safe_dtw(a,b,w):
    try: return dtw_band(a,b,w)
    except: return dtw_band(a,b,max(len(a),len(b)))

# ---------- filas ----------
def chunks(tok:List[str],n:int)->List[str]:
    out,cur=[],[]
    for t in tok:
        cur.append(t)
        if len(cur)>=n:
            out.append(" ".join(cur)); cur=[]
    if cur: out.append(" ".join(cur))
    return out

def build_rows(ref:str,hyp:str)->List[List]:
    ref_tok=normalize(ref).split()
    hyp_tok=normalize(hyp).split()

    ref_sw=[t for t in ref_tok if t not in STOP]
    hyp_sw=[t for t in hyp_tok if t not in STOP]
    coarse=safe_dtw(ref_sw,hyp_sw,COARSE_W)

    idx_r=[i for i,t in enumerate(ref_tok) if t not in STOP]
    idx_h=[j for j,t in enumerate(hyp_tok) if t not in STOP]
    map_h=[-1]*len(ref_tok)
    for (i,j) in coarse:
        map_h[idx_r[i]]=idx_h[j]

    last=-1
    for i in range(len(map_h)):
        if map_h[i]!=-1:
            last=map_h[i]
        elif last+1<len(hyp_tok):
            last+=1; map_h[i]=last

    rows=[]; pos=0
    for line_id,line in enumerate(chunks(ref_tok,LINE_LEN)):
        span_start=pos
        span_end  =pos+len(line.split())    # fin exclusivo  ✅
        pos       =span_end

        h_idx=[map_h[k] for k in range(span_start,span_end) if map_h[k]!=-1]
        asr_line=" ".join(hyp_tok[min(h_idx):max(h_idx)+1]) if h_idx else ""
        wer=Levenshtein.normalized_distance(line,asr_line)
        flag="✅" if wer<=WARN_WER else ("⚠️" if wer<=0.20 else "❌")
        dur=round(len(asr_line.split())/3,2)
        rows.append([line_id,flag,round(wer*100,1),dur,line,asr_line])
    return rows

# ---------- GUI (igual que v 4.3 salvo atributo/función log) ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro")
        self.geometry("1850x760")
        self.v_ref=tk.StringVar(); self.v_asr=tk.StringVar(); self.v_json=tk.StringVar()
        self.q=queue.Queue()

        top=ttk.Frame(self); top.pack(fill="x",padx=3,pady=2)
        ttk.Label(top,text="Guion:").grid(row=0,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_ref,width=70).grid(row=0,column=1)
        ttk.Button(top,text="…",command=lambda:self.browse(self.v_ref,("PDF/TXT","*.pdf;*.txt"))).grid(row=0,column=2)

        ttk.Label(top,text="TXT ASR:").grid(row=1,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_asr,width=70).grid(row=1,column=1)
        ttk.Button(top,text="…",command=lambda:self.browse(self.v_asr,("TXT","*.txt"))).grid(row=1,column=2)

        ttk.Button(top,text="Procesar",width=11,command=self.launch).grid(row=0,column=3,rowspan=2,padx=6)

        ttk.Label(top,text="JSON:").grid(row=2,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_json,width=70).grid(row=2,column=1)
        ttk.Button(top,text="Abrir JSON…",command=self.load_json).grid(row=2,column=2)

        self.tree=ttk.Treeview(self,columns=("ID","✓","WER","dur","Original","ASR"),
                               show="headings",height=27)
        for c,w in zip(("ID","✓","WER","dur","Original","ASR"),
                       (50,30,60,60,800,800)):
            self.tree.heading(c,text=c); self.tree.column(c,width=w,anchor="w")
        self.tree.pack(fill="both",expand=True,padx=3,pady=2)

        self.log_box=scrolledtext.ScrolledText(self,height=5,state="disabled")
        self.log_box.pack(fill="x",padx=3,pady=2)

        self.after(250,self._poll)

    def browse(self,var,ft):
        p=filedialog.askopenfilename(filetypes=[ft]);
        if p: var.set(p)

    def log_msg(self,msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end",msg+"\n")
        self.log_box.configure(state="disabled"); self.log_box.see("end")

    def clear_table(self): self.tree.delete(*self.tree.get_children())

    def launch(self):
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info","Selecciona guion y TXT ASR."); return
        self.clear_table(); self.log_msg("⏳ Iniciando…")
        threading.Thread(target=self.worker,daemon=True).start()

    def worker(self):
        try:
            self.q.put("→ Leyendo guion…"); ref=read_script(self.v_ref.get())
            self.q.put("→ TXT externo cargado")
            hyp=Path(self.v_asr.get()).read_text(encoding="utf8",errors="ignore")
            self.q.put("→ Alineando…")
            rows=build_rows(ref,hyp)
            out=Path(self.v_asr.get()).with_suffix(".qc.json")
            out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf8")
            self.q.put(("ROWS",rows)); self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf=io.StringIO(); traceback.print_exc(file=buf); self.q.put(buf.getvalue())

    def load_json(self):
        if not self.v_json.get():
            p=filedialog.askopenfilename(filetypes=[("QC JSON","*.qc.json;*.json")]);
            if not p: return; self.v_json.set(p)
        try:
            rows=json.loads(Path(self.v_json.get()).read_text(encoding="utf8"))
            self.clear_table()
            for r in rows: self.tree.insert("",tk.END,values=r)
            self.log_msg(f"✔ Cargado {self.v_json.get()}")
        except Exception as e: messagebox.showerror("Error",str(e))

    def _poll(self):
        try:
            while True:
                msg=self.q.get_nowait()
                if isinstance(msg,tuple) and msg[0]=="ROWS":
                    for r in msg[1]: self.tree.insert("",tk.END,values=r)
                else: self.log_msg(str(msg))
        except queue.Empty: pass
        self.after(250,self._poll)

if __name__=="__main__":
    App().mainloop()

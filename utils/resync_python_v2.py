# resync_json_gui.py – re-sincroniza tiempos ‘tc’ en un QC-JSON usando un CSV palabra-tiempo
# ------------------------------------------------------------------------------------------
# ▸ Selecciona (GUI) un archivo .qc.json y el .words.csv con los time-codes.
# ▸ Encuentra anclas (n-gramas exactos 5→4→3→2→1) entre ASR y CSV.
# ▸ Interpola linealmente entre anclas: cada palabra_json hereda tc de la palabra_csv alineada.
# ▸ Para cada fila ASR coloca tc = tiempo de su PRIMERA palabra y ajusta fila siguiente
# ▸ Guarda <nombre>.resync.json junto al original y muestra progreso / avisos.
# ------------------------------------------------------------------------------------------
from __future__ import annotations
import json, re, sys, threading, tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk, messagebox

from utils.gui_errors import show_error
from typing import List, Tuple
import unicodedata
from difflib import SequenceMatcher

###########################################################################################
# ▸ 1. utilidades de normalización y tokenización
###########################################################################################
_tok_re = re.compile(r"\w+['-]?\w*")

def norm(txt: str) -> str:
    txt = unicodedata.normalize("NFD", txt.lower())
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9'\-\s]", " ", txt).strip()

def tokenize(text: str) -> List[str]:
    return _tok_re.findall(norm(text))

###########################################################################################
# ▸ 2. leer CSV palabra-tiempo  (acepta ; , tab)
###########################################################################################
def load_words_csv(path: Path) -> Tuple[List[str], List[float]]:
    words, tcs = [], []
    with path.open("r", encoding="utf8", errors="ignore") as fh:
        for ln, line in enumerate(fh, 1):
            line=line.strip()
            if not line: continue
            # separador flexible
            if ";" in line:   t_str, w_raw = line.split(";",1)
            elif "," in line: t_str, w_raw = line.split(",",1)
            else:
                parts = re.split(r"\s+", line, 1)
                if len(parts)<2: continue
                t_str, w_raw = parts
            try:
                t = float(t_str.replace(",",".")); w = tokenize(w_raw)
            except ValueError: continue
            for wtok in w:                   # puede haber varias palabras en un token csv
                words.append(wtok)
                tcs.append(round(t,2))
    return words, tcs

###########################################################################################
# ▸ 3. localizar ANCLAS 5→4→3→2 (bloques únicos)
###########################################################################################
def find_anchors(csv_words: List[str], json_words: List[str]) -> List[Tuple[int,int,int]]:
    anchors: List[Tuple[int,int,int]]=[]
    used_csv=set(); used_json=set()
    joined_csv=" ".join(csv_words)
    for n in (5,4,3,2):
        # índice rápido csv n-gram -> idx
        csv_map = {" ".join(csv_words[i:i+n]): i
                   for i in range(len(csv_words)-n+1)}
        j=0
        while j<=len(json_words)-n:
            if any((j+k) in used_json for k in range(n)):
                j+=1; continue
            key = " ".join(json_words[j:j+n])
            i = csv_map.get(key)
            if i is not None and not any((i+k) in used_csv for k in range(n)):
                anchors.append((j,i,n))
                for k in range(n): used_json.add(j+k); used_csv.add(i+k)
                j+=n; continue
            j+=1
    return sorted(anchors, key=lambda a:a[0])

###########################################################################################
# ▸ 4. alineación simple entre dos listas (uso de difflib)
###########################################################################################
def align_chunk(j_words: List[str], c_words: List[str]) -> List[int]:
    """
    Devuelve para cada posición de j_words el índice en c_words que
    mejor coincide, o -1 si gap (cuando no hay match).
    Se basa en SequenceMatcher (O(n*m) pero los trozos son pequeños).
    """
    sm = SequenceMatcher(None, j_words, c_words, autojunk=False)
    mapping=[-1]*len(j_words)
    for tag, i1,i2,j1,j2 in sm.get_opcodes():
        if tag=="equal":
            for k in range(i2-i1):
                mapping[i1+k]=j1+k
    # rellenar huecos linealmente
    last=-1
    prev_idx=-1
    for idx,val in enumerate(mapping):
        if val!=-1:
            # propagar hacia atrás
            if last==-1:
                for k in range(idx):
                    mapping[k]=val
            else:
                step=(val-last)/(idx-prev_idx)
                for k in range(prev_idx+1, idx):
                    mapping[k]=round(last+step*(k-prev_idx))
            prev_idx, last = idx, val
    # rellenar cola
    if mapping and mapping[-1]==-1:
        last_idx=max([i for i,v in enumerate(mapping) if v!=-1], default=None)
        if last_idx is not None:
            for k in range(last_idx+1,len(mapping)):
                mapping[k]=mapping[last_idx]
    return mapping

###########################################################################################
# ▸ 5. proceso completo de resync
###########################################################################################
def resync_rows(rows: List[List], csv_words: List[str], csv_tcs: List[float],
                log_cb=lambda *_:None, progress_cb=lambda *_:None):

    # a) explotar json en lista plana de palabras
    j_tokens: List[str]=[]
    tok2row=[]               # idx_json -> row_id
    for ridx,row in enumerate(rows):
        toks = tokenize(row[-1])
        j_tokens.extend(toks)
        tok2row.extend([ridx]*len(toks))

    # b) anclas
    anchors=find_anchors(csv_words, j_tokens)
    log_cb(f"Encontradas {len(anchors)} anclas")
    real_anchors = anchors[:]

    # c) default mapping
    mapping=[-1]*len(j_tokens)

    # d) copiar anclas directas
    for jidx,cidx,n in anchors:
        for k in range(n):
            mapping[jidx+k]=cidx+k

    # e) recorrer intervalos y alinear interior
    segs=[]
    prev_j=prev_c=0
    anchors.append( (len(j_tokens), len(csv_words), 0) )    # ancla final artificial
    for j,c,n in anchors:
        j_chunk=j_tokens[prev_j:j]
        c_chunk=csv_words[prev_c:c]
        if j_chunk and c_chunk:
            local_map=align_chunk(j_chunk,c_chunk)
            for off,cm in enumerate(local_map):
                mapping[prev_j+off]=prev_c+cm if cm!=-1 else prev_c
        prev_j, prev_c = j+n, c+n

    # f) asignar tc a cada fila (inicio = primera palabra de esa fila)
    row_tc=[None]*len(rows)
    for jidx, cidx in enumerate(mapping):
        ridx=tok2row[jidx]
        if row_tc[ridx] is None and cidx!=-1:
            row_tc[ridx]=csv_tcs[cidx]

    unmapped = sum(1 for tc in row_tc if tc is None)
    if unmapped / len(row_tc) > 0.3:
        log_cb(f"WARNING: {unmapped} of {len(row_tc)} rows could not be mapped")

    # g) propagar faltantes linealmente
    last_tc=0.0
    for i in range(len(row_tc)):
        if row_tc[i] is None:
            row_tc[i]=last_tc
        else:
            last_tc=row_tc[i]

    # filas tras la última ancla: interpolar hasta el último tiempo CSV
    if real_anchors:
        j_last = real_anchors[-1][0] + real_anchors[-1][2] - 1
        last_row = tok2row[j_last]
    else:
        last_row = -1
    if last_row < len(row_tc) - 1:
        start = row_tc[last_row] if last_row >= 0 else 0.0
        end = csv_tcs[-1]
        n = len(row_tc) - last_row - 1
        if real_anchors:
            if n > 0 and end > start:
                step = (end - start) / n
                for idx in range(1, n + 1):
                    row_tc[last_row + idx] = start + step * idx
        else:
            if n > 1 and end > start:
                step = (end - start) / (n - 1)
                for idx in range(n):
                    row_tc[idx] = start + step * idx

    # h) escribir tc en la columna adecuada  / progress
    tc_idx = 5 if len(rows[0]) > 5 else len(rows[0])
    for i, row in enumerate(rows):
        if len(row) <= tc_idx:
            row.extend([""] * (tc_idx - len(row) + 1))
        row[tc_idx] = f"{row_tc[i]:.2f}"
        if i % 10 == 0:
            progress_cb(i / len(rows))


def resync_file(json_path: str | Path, csv_path: str | Path) -> List[List]:
    """Return rows from ``json_path`` with updated ``tc`` using ``csv_path``."""

    rows = json.loads(Path(json_path).read_text(encoding="utf8"))
    csv_words, csv_tcs = load_words_csv(Path(csv_path))
    resync_rows(rows, csv_words, csv_tcs)
    return rows

###########################################################################################
# ▸ 6. GUI  (sin cambios visuales)
###########################################################################################
class ResyncApp(tk.Tk):
    def __init__(self)->None:
        super().__init__()
        self.title("Re-sincronizar QC-JSON con CSV word-timings")
        self.geometry("650x430")
        self.v_json=tk.StringVar(); self.v_csv=tk.StringVar()

        frm=ttk.Frame(self); frm.pack(fill="x", padx=10, pady=8)
        ttk.Label(frm,text="QC JSON:").grid(row=0,column=0,sticky="e")
        ttk.Entry(frm,textvariable=self.v_json,width=60).grid(row=0,column=1)
        ttk.Button(frm,text="…",command=self.pick_json).grid(row=0,column=2)

        ttk.Label(frm,text="CSV words:").grid(row=1,column=0,sticky="e")
        ttk.Entry(frm,textvariable=self.v_csv,width=60).grid(row=1,column=1)
        ttk.Button(frm,text="…",command=self.pick_csv).grid(row=1,column=2)

        ttk.Button(frm,text="Re-sincronizar",command=self.launch).grid(row=2,column=1,pady=8)
        self.pbar=ttk.Progressbar(self,length=600); self.pbar.pack(padx=10,pady=4)
        self.log=scrolledtext.ScrolledText(self,height=13,state="disabled")
        self.log.pack(fill="both",expand=True,padx=10,pady=(0,10))

    # UI helpers
    def pick_json(self):
        p=filedialog.askopenfilename(filetypes=[("QC JSON","*.json;*.qc.json")])
        if p: self.v_json.set(p)
    def pick_csv(self):
        p=filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if p: self.v_csv.set(p)
    def log_msg(self,txt:str):
        self.log["state"]="normal"; self.log.insert("end",txt+"\n")
        self.log["state"]="disabled"; self.log.see("end")

    # main
    def launch(self):
        pj,pc=self.v_json.get(),self.v_csv.get()
        if not (pj and pc):
            show_error("Falta info", ValueError("Selecciona JSON y CSV")); return
        threading.Thread(target=self.worker,args=(Path(pj),Path(pc)),daemon=True).start()

    def worker(self,json_path:Path,csv_path:Path):
        try:
            self.log_msg("Leyendo JSON…"); rows=json.loads(json_path.read_text(encoding="utf8"))
            total=len(rows); self.log_msg(f"→ {total} filas")

            self.log_msg("Leyendo CSV word-timings…")
            csv_words,csv_tcs=load_words_csv(csv_path)
            if not csv_words: show_error("Error", ValueError("CSV vacío")); return
            self.log_msg(f"→ {len(csv_words)} palabras en CSV")

            resync_rows(rows,csv_words,csv_tcs,
                        log_cb=self.log_msg,
                        progress_cb=lambda v:self.pbar.configure(value=v*100))

            out=json_path.with_suffix(".resync.json")
            out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),"utf8")
            self.pbar["value"]=100
            self.log_msg(f"✔ Terminado. Guardado en {out}")
        except Exception as exc:
            self.log_msg(f"ERROR: {exc}")
            raise

###########################################################################################
if __name__=="__main__":
    ResyncApp().mainloop()

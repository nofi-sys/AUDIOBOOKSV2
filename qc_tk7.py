#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_tk.py  –  Revisor Audiolibros   v5.1
· Segmentación por “anclas” léxicas de 3 palabras (trigramas)
· DTW interno en cada tramo delimitado por anclas
· Penalización temporal aproximada (timestamps proxy)
· líneas de 12 palabras   · curva abre / guarda .qc.json
MIT • 2025
"""
import threading, queue, json, re, io, traceback, math
from pathlib import Path
from typing import List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import unidecode, pdfplumber
from rapidfuzz.distance import Levenshtein


# ──────────────────────── parámetros globales ────────────────────────
LINE_LEN   = 12         # palabras por fila
COARSE_W   = 40         # ventana DTW inicial
WARN_WER   = 0.08
ANCHOR_MAX_FREQ = 2     # frecuencia máxima para trigramas ancla
GAMMA_TIME = 0.3        # peso de la “penalización temporal” en DTW

STOP = {"de","la","el","y","que","en","a","los","se","del","por","con","las",
        "un","para","una","su","al","lo","como","más","o","pero","sus","le",
        "ya","fue"}

DIGIT_NAMES = {
    "0":"cero","1":"uno","2":"dos","3":"tres","4":"cuatro","5":"cinco","6":"seis",
    "7":"siete","8":"ocho","9":"nueve","10":"diez","11":"once","12":"doce",
    "13":"trece","14":"catorce","15":"quince","16":"dieciseis","17":"diecisiete",
    "18":"dieciocho","19":"diecinueve","20":"veinte"
}


# ───────────────────────── utilidades texto ──────────────────────────
def normalize(t:str, strip_punct:bool=True)->str:
    """
    - Minúsculas, quita acentos
    - Si strip_punct=True, reemplaza toda puntuación por espacio
    - Colapsa espacios contiguos
    """
    t = unidecode.unidecode(t.lower())
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def read_script(p:str)->str:
    """
    Lee texto de PDF o TXT. Si es PDF, lo abre con pdfplumber.
    Si no extrae nada, lanza RuntimeError para que el usuario use TXT.
    """
    p = Path(p)
    if p.suffix.lower()==".pdf":
        with pdfplumber.open(p) as pdf:
            pages = [pg.extract_text() or "" for pg in pdf.pages]
        raw = "\n".join(pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa un TXT.")
        return raw
    return p.read_text(encoding="utf8",errors="ignore")


# ─────────── comparación difusa de tokens + equivalencias ────────────
def token_equal(a:str,b:str)->bool:
    """
    Devuelve True si 'a' y 'b' son muy parecidos:
      - idénticos
      - distancia Levenshtein normalizada ≤ 0.2
      - uno es dígito y el otro su forma textual (0↔"cero", etc.)
    """
    if a == b:
        return True
    if Levenshtein.normalized_distance(a, b) <= 0.2:
        return True
    if a.isdigit() and DIGIT_NAMES.get(a) == b:
        return True
    if b.isdigit() and DIGIT_NAMES.get(b) == a:
        return True
    return False


# ─────────────── localizar “anclas” de trigramas ───────────────────
def find_anchor_trigrams(ref_tok:List[str], hyp_tok:List[str]) -> List[Tuple[int,int]]:
    """
    1) Construye todos los trigramas (3 tokens consecutivos) en ref_tok y en hyp_tok.
    2) Cuenta su frecuencia. Solo sobreviven trigramas cuya frecuencia ≤ ANCHOR_MAX_FREQ en AMBOS textos.
    3) Recorre cada trigram en ref_tok en orden ascendente de índice i; busca su primera aparición en hyp_tok a partir de j_last.
    4) Devuelve la lista (i_ref, j_hyp) sobre INDICES INICIALES del trigram.
    5) Filtra para que sean estrictamente monótonos en i y j.
    """
    # 1) generar todos los trigramas
    ref_trigs: List[Tuple[str,str,str]] = []
    hyp_trigs: List[Tuple[str,str,str]] = []

    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i+1], ref_tok[i+2])
        # ignorar trigramas que incorporen stopwords
        if tri[0] in STOP or tri[1] in STOP or tri[2] in STOP:
            continue
        ref_trigs.append(tri)

    for j in range(len(hyp_tok) - 2):
        tri = (hyp_tok[j], hyp_tok[j+1], hyp_tok[j+2])
        if tri[0] in STOP or tri[1] in STOP or tri[2] in STOP:
            continue
        hyp_trigs.append(tri)

    # 2) contar frecuencias
    freq_ref: Dict[Tuple[str,str,str],int] = {}
    freq_hyp: Dict[Tuple[str,str,str],int] = {}
    for tri in ref_trigs:
        freq_ref[tri] = freq_ref.get(tri, 0) + 1
    for tri in hyp_trigs:
        freq_hyp[tri] = freq_hyp.get(tri, 0) + 1

    # 3) seleccionar trigramas “poco frecuentes” en ambos
    lowfreq_ref = {tri for tri,count in freq_ref.items() if count <= ANCHOR_MAX_FREQ}
    lowfreq_hyp = {tri for tri,count in freq_hyp.items() if count <= ANCHOR_MAX_FREQ}
    candidate_trigs = lowfreq_ref.intersection(lowfreq_hyp)

    # 4) emparejar en orden (primera aparición en hyp_tok)
    anchors: List[Tuple[int,int]] = []
    j_last = 0
    for i in range(len(ref_tok) - 2):
        tri = (ref_tok[i], ref_tok[i+1], ref_tok[i+2])
        if tri not in candidate_trigs:
            continue
        # buscar primera aparición en hyp_tok desde j_last
        for j in range(j_last, len(hyp_tok) - 2):
            if (hyp_tok[j], hyp_tok[j+1], hyp_tok[j+2]) == tri:
                anchors.append((i, j))
                j_last = j + 3   # saltamos toda la longitud del trigram para no solaparnos
                break

    # 5) filtrar monotonicidad estricta
    filtered: List[Tuple[int,int]] = []
    last_i, last_j = -1, -1
    for (i,j) in anchors:
        if i > last_i and j > last_j:
            filtered.append((i,j))
            last_i, last_j = i, j

    return filtered


# ────────────────────── DTW con penalización temporal ─────────────────
def dtw_band(a:List[str], b:List[str], w:int) -> List[Tuple[int,int]]:
    """
    Igual que antes, pero sumamos un término pos_cost = GAMMA_TIME * |(i/n) - (j/m)|
    para empujar la trayectoria cerca de la diagonal (proxy de timestamps aproximados).
    """
    n, m = len(a), len(b)
    W = max(w, abs(n-m))
    BIG = 1e9

    D   : Dict[Tuple[int,int], Tuple[float, Tuple[int,int] or None]] = {}
    back: Dict[Tuple[int,int], Tuple[int,int]] = {}

    D[(-1,-1)] = (0, None)

    for i in range(n):
        lo = max(0, i - W)
        hi = min(m-1, i + W)
        for j in range(lo, hi + 1):
            best_cost = BIG
            best_prev = None
            # vecinos: arriba / izquierda / diagonal
            for (di, dj, move_cost) in ((-1,0,1), (0,-1,1), (-1,-1,0)):
                prev = (i+di, j+dj)
                if prev in D:
                    c = D[prev][0] + move_cost
                    if c < best_cost:
                        best_cost, best_prev = c, prev

            # coste match/ mismatch en token
            match_cost = 0 if token_equal(a[i], b[j]) else 1
            # penalización temporal (distancia relativa entre índices)
            pos_cost = GAMMA_TIME * abs((i / n) - (j / m))
            total = best_cost + match_cost + pos_cost

            D[(i,j)] = (total, best_prev)
            back[(i,j)] = best_prev

    if (n-1, m-1) not in back:
        raise RuntimeError("DTW fuera de ventana")
    path: List[Tuple[int,int]] = []
    i, j = n-1, m-1
    while (i,j) != (-1, -1):
        path.append((i,j))
        prev = back[(i,j)]
        if prev is None:
            break
        i, j = prev
    return path[::-1]

def safe_dtw(a:List[str], b:List[str], w:int) -> List[Tuple[int,int]]:
    """
    Prueba dtw_band(a,b,w), duplicando w hasta 2× máximo.
    Si todo falla, aplica fallback_monótono.
    """
    band = w
    max_band = max(len(a), len(b)) * 2
    while band <= max_band:
        try:
            return dtw_band(a, b, band)
        except RuntimeError:
            band *= 2
    # fallback monótono
    return fallback_pairs(a, b)

def fallback_pairs(ref_tokens:List[str], hyp_tokens:List[str]) -> List[Tuple[int,int]]:
    """
    Para cada token de ref_tokens busca la primera coincidencia token_equal
    en hyp_tokens a partir de j_last. Monótono en j.
    """
    pairs = []
    j_last = 0
    for i, t in enumerate(ref_tokens):
        for j in range(j_last, len(hyp_tokens)):
            if token_equal(t, hyp_tokens[j]):
                pairs.append((i,j))
                j_last = j + 1
                break
    return pairs


# ───────────────── construir filas con anclas de trigramas ───────────
def build_rows(ref:str, hyp:str) -> List[List]:
    """
    1) Tokeniza y normaliza SIN quitar puntuación (strip_punct=False) para conservar
       comas/puntos como separación de tokens.
    2) Crea listas ref_tok, hyp_tok (tokens con posibles comas).
    3) Llama a find_anchor_trigrams() y obtiene lista monótona de (i_ref, j_hyp).
    4) Para cada intervalo entre anclas consecutivas, realiza DTW interno usando safe_dtw
       sobre sublistas FILTRADAS de stopwords. Mapea índices back a índices globales.
    5) Rellena huecos monótonos en map_h para que ninguna palabra quede sin par (asignación
       secuencial del siguiente token libre).
    6) Divide ref_tok en bloques de LINE_LEN y para cada bloque arma la línea Original,
       calcula min/max en mapa para extraer la línea ASR, luego WER y marca (✅⚠️❌).
    """
    # 1) Tokenización b-granular (puntos/comas como tokens)
    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = normalize(hyp, strip_punct=False).split()

    # 2) hallar anclas de trigramas
    anchor_pairs = find_anchor_trigrams(ref_tok, hyp_tok)

    # 3) recorrer entre anclas y alinear cada tramo
    full_pairs: List[Tuple[int,int]] = []
    # insertar pseudo-ancla inicio y fin
    seg_starts = [(-1, -1)] + anchor_pairs + [(len(ref_tok)-1, len(hyp_tok)-1)]
    for (prev_i, prev_j), (next_i, next_j) in zip(seg_starts[:-1], seg_starts[1:]):
        # sub-rango exclusivo de next_i/next_j
        if next_i > prev_i + 1 and next_j > prev_j + 1:
            sub_ref = ref_tok[prev_i+1 : next_i]
            sub_hyp = hyp_tok[prev_j+1 : next_j]
            # filtrar STOP para DTW grueso
            sub_r_sw = [t for t in sub_ref if t not in STOP]
            sub_h_sw = [t for t in sub_hyp if t not in STOP]
            if sub_r_sw and sub_h_sw:
                pairs_sw = safe_dtw(sub_r_sw, sub_h_sw, COARSE_W)
                # reconstruir índices globales
                idx_r = [k for k,t in enumerate(sub_ref) if t not in STOP]
                idx_h = [k for k,t in enumerate(sub_hyp) if t not in STOP]
                for ri, hj in pairs_sw:
                    orig_i = prev_i + 1 + idx_r[ri]
                    hyp_i  = prev_j + 1 + idx_h[hj]
                    full_pairs.append((orig_i, hyp_i))
        # añadir la propia ancla (si existe en rangos válidos)
        if 0 <= next_i < len(ref_tok) and 0 <= next_j < len(hyp_tok):
            full_pairs.append((next_i, next_j))

    # 4) map_h global, rellenar monótono
    map_h = [-1] * len(ref_tok)
    for (ri, hj) in full_pairs:
        if 0 <= ri < len(ref_tok) and 0 <= hj < len(hyp_tok) and map_h[ri] == -1:
            map_h[ri] = hj
    last = -1
    for i in range(len(map_h)):
        if map_h[i] != -1:
            last = map_h[i]
        else:
            if last + 1 < len(hyp_tok):
                last += 1
                map_h[i] = last
            # si last+1 está fuera, queda como -1 (vacío)

    # 5) dividir en líneas de LINE_LEN
    rows = []
    pos = 0
    line_id = 0
    while pos < len(ref_tok):
        block = ref_tok[pos : pos + LINE_LEN]
        span_start = pos
        span_end   = pos + len(block)
        pos        = span_end

        h_idxs = [map_h[k] for k in range(span_start, span_end) if map_h[k] != -1]
        if h_idxs:
            h_start = min(h_idxs)
            h_end   = max(h_idxs) + 1
            asr_line = " ".join(hyp_tok[h_start:h_end])
        else:
            asr_line = ""

        orig_line = " ".join(block)
        wer_val   = Levenshtein.normalized_distance(orig_line, asr_line) if asr_line else 1.0
        flag      = "✅" if wer_val <= WARN_WER else ("⚠️" if wer_val <= 0.20 else "❌")
        dur       = round(len(asr_line.split()) / 3.0, 2)

        rows.append([line_id, flag, round(wer_val * 100, 1), dur, orig_line, asr_line])
        line_id += 1

    return rows


# ──────────────────────────── GUI (sin cambios) ──────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro  v5.1")
        self.geometry("1850x760")

        self.v_ref  = tk.StringVar()
        self.v_asr  = tk.StringVar()
        self.v_json = tk.StringVar()
        self.q      = queue.Queue()

        top = ttk.Frame(self); top.pack(fill="x", padx=3, pady=2)
        ttk.Label(top,text="Guion:").grid(row=0,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_ref,width=70).grid(row=0,column=1)
        ttk.Button(top,text="…",command=lambda: self.browse(self.v_ref,("PDF/TXT","*.pdf;*.txt"))).grid(row=0,column=2)

        ttk.Label(top,text="TXT ASR:").grid(row=1,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_asr,width=70).grid(row=1,column=1)
        ttk.Button(top,text="…",command=lambda: self.browse(self.v_asr,("TXT","*.txt"))).grid(row=1,column=2)

        ttk.Button(top,text="Procesar",width=11,command=self.launch).grid(row=0,column=3,rowspan=2,padx=6)

        ttk.Label(top,text="JSON:").grid(row=2,column=0,sticky="e")
        ttk.Entry(top,textvariable=self.v_json,width=70).grid(row=2,column=1)
        ttk.Button(top,text="Abrir JSON…",command=self.load_json).grid(row=2,column=2)

        self.tree = ttk.Treeview(self,columns=("ID","✓","WER","dur","Original","ASR"),
                                 show="headings",height=27)
        for c,w in zip(("ID","✓","WER","dur","Original","ASR"),
                       (50,30,60,60,800,800)):
            self.tree.heading(c,text=c)
            self.tree.column(c,width=w,anchor="w")
        self.tree.pack(fill="both",expand=True,padx=3,pady=2)

        self.log_box = scrolledtext.ScrolledText(self,height=5,state="disabled")
        self.log_box.pack(fill="x",padx=3,pady=2)

        self.after(250,self._poll)

    # ───────────── helpers GUI ─────────────
    def browse(self,var,ft):
        p = filedialog.askopenfilename(filetypes=[ft])
        if p:
            var.set(p)

    def log_msg(self,msg:str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end",msg+"\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def clear_table(self):
        self.tree.delete(*self.tree.get_children())

    # ───────────────── worker ─────────────────
    def launch(self):
        if not(self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info","Selecciona guion y TXT ASR."); return
        self.clear_table()
        self.log_msg("⏳ Iniciando…")
        threading.Thread(target=self._worker,daemon=True).start()

    def _worker(self):
        try:
            self.q.put("→ Leyendo guion…")
            ref = read_script(self.v_ref.get())

            self.q.put("→ TXT externo cargado")
            hyp = Path(self.v_asr.get()).read_text(encoding="utf8",errors="ignore")

            self.q.put("→ Alineando…")
            rows = build_rows(ref, hyp)

            out = Path(self.v_asr.get()).with_suffix(".qc.json")
            out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf8")

            self.q.put(("ROWS",rows))
            self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())

    def load_json(self):
        if not self.v_json.get():
            p = filedialog.askopenfilename(filetypes=[("QC JSON","*.qc.json;*.json")])
            if not p: return
            self.v_json.set(p)
        try:
            rows = json.loads(Path(self.v_json.get()).read_text(encoding="utf8"))
            self.clear_table()
            for r in rows:
                self.tree.insert("",tk.END,values=r)
            self.log_msg(f"✔ Cargado {self.v_json.get()}")
        except Exception as e:
            messagebox.showerror("Error",str(e))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg,tuple) and msg[0]=="ROWS":
                    for r in msg[1]:
                        self.tree.insert("",tk.END,values=r)
                else:
                    self.log_msg(str(msg))
        except queue.Empty:
            pass
        self.after(250,self._poll)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()

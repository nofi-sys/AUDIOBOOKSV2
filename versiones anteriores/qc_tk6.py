#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_tk.py  –  Revisor Audiolibros   v4.4 (mejorado para alineamiento robusto)
• líneas de 12 palabras
• guarda / abre JSON   (.qc.json)
• corrige la pérdida de la última palabra en la columna ASR
• usa DTW difuso sobre tokens filtrados de stopwords y fallback monótono
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
LINE_LEN = 12
COARSE_W = 40
WARN_WER = 0.08

# stopwords en español (mismos que antes)
STOP = {
    "de","la","el","y","que","en","a","los","se","del","por","con","las",
    "un","para","una","su","al","lo","como","más","o","pero","sus","le",
    "ya","fue"
}

# patrones para normalizar
DIGIT_RE = re.compile(r"\b\d+\b")


# ---------- utilidades texto ----------
def normalize(t: str, strip_punct: bool = True) -> str:
    """
    1) a minúsculas, quita acentos
    2) opcionalmente elimina toda puntuación (strip_punct=True)
    3) colapsa espacios
    """
    t = unidecode.unidecode(t.lower())
    if strip_punct:
        t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def read_script(p: str) -> str:
    """
    Lee texto de PDF o TXT. Si es PDF, lo abre con pdfplumber.
    Si no extrae nada, lanza excepción para usar .txt.
    """
    p = Path(p)
    if p.suffix.lower() == ".pdf":
        with pdfplumber.open(p) as pdf:
            raw = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
        if not raw.strip():
            raise RuntimeError("No se pudo extraer texto del PDF; usa TXT.")
        return raw
    return p.read_text(encoding="utf8", errors="ignore")


# ---------- comparación difusa de tokens ----------
def token_equal(a: str, b: str) -> bool:
    """
    Retorna True si 'a' y 'b' son suficientemente similares:
      - idénticos
      - distancia Levenshtein normalizada ≤ 0.2
      - uno es dígito y el otro su forma escrita (ej. "3" ↔ "tres")
    """
    if a == b:
        return True
    # distancia Levenshtein normalizada
    if Levenshtein.normalized_distance(a, b) <= 0.2:
        return True
    # dígito ↔ texto
    if a.isdigit():
        palabra = {
            "0": "cero",  "1": "uno",   "2": "dos",    "3": "tres",
            "4": "cuatro","5": "cinco","6": "seis",   "7": "siete",
            "8": "ocho",  "9": "nueve","10": "diez", "11": "once",
            "12": "doce", "13": "trece","14": "catorce","15": "quince",
            "16": "dieciseis","17": "diecisiete","18": "dieciocho","19": "diecinueve",
            "20": "veinte"
        }.get(a)
        if palabra == b:
            return True
    if b.isdigit():
        palabra = {
            "0": "cero",  "1": "uno",   "2": "dos",    "3": "tres",
            "4": "cuatro","5": "cinco","6": "seis",   "7": "siete",
            "8": "ocho",  "9": "nueve","10": "diez", "11": "once",
            "12": "doce", "13": "trece","14": "catorce","15": "quince",
            "16": "dieciseis","17": "diecisiete","18": "dieciocho","19": "diecinueve",
            "20": "veinte"
        }.get(b)
        if palabra == a:
            return True
    return False


# ---------- DTW con ventana y fallback ----------
def dtw_band(a: List[str], b: List[str], w: int) -> List[Tuple[int,int]]:
    """
    DTW clásico con ventana 'w', pero usando token_equal para comparar.
    Devuelve lista de pares (i,j). Si no logra alinear (i.e. (n-1,m-1) no está en back),
    lanza RuntimeError.
    """
    n, m = len(a), len(b)
    W = max(w, abs(n - m))
    BIG = 10**9

    # Inicializar D[i,j] = (costo, prev) para i,j desde -1..n-1, -1..m-1
    D: Dict[Tuple[int,int], Tuple[float, Tuple[int,int] or None]] = {}
    back: Dict[Tuple[int,int], Tuple[int,int]] = {}
    for i in range(-1, n):
        for j in range(-1, m):
            D[(i,j)] = (BIG, None)
    D[(-1,-1)] = (0, None)

    for i in range(n):
        lo = max(0, i - W)
        hi = min(m - 1, i + W)
        for j in range(lo, hi + 1):
            # vecinos: (i-1,j), (i,j-1), (i-1,j-1)
            best_cost = BIG
            best_prev = None

            # de arriba (i-1,j)
            prev = (i - 1, j)
            if prev in D:
                c = D[prev][0] + 1
                if c < best_cost:
                    best_cost = c
                    best_prev = prev

            # de izquierda (i,j-1)
            prev = (i, j - 1)
            if prev in D:
                c = D[prev][0] + 1
                if c < best_cost:
                    best_cost = c
                    best_prev = prev

            # diagonal (i-1,j-1)
            prev = (i - 1, j - 1)
            if prev in D:
                cost_match = 0 if token_equal(a[i], b[j]) else 1
                c = D[prev][0] + cost_match
                if c < best_cost:
                    best_cost = c
                    best_prev = prev

            D[(i, j)] = (best_cost, best_prev)
            back[(i, j)] = best_prev

    # reconstrucción del camino
    if (n-1, m-1) not in back:
        raise RuntimeError("DTW fuera de ventana")
    path: List[Tuple[int,int]] = []
    i, j = n-1, m-1
    while (i, j) != ( -1, -1 ):
        path.append((i, j))
        prev = back[(i, j)]
        if prev is None:
            break
        i, j = prev
    path.reverse()
    return path


def safe_dtw(a: List[str], b: List[str], w: int) -> List[Tuple[int,int]]:
    """
    Intenta dtw_band(a,b,w). Si falla (fuera de banda), duplica w hasta 4 veces.
    Si aún falla, lanza RuntimeError.
    """
    band = w
    max_band = max(len(a), len(b))
    while band <= max_band:
        try:
            return dtw_band(a, b, band)
        except RuntimeError:
            band *= 2
    raise RuntimeError("No se pudo alinear aun con banda máxima")


def fallback_pairs(ref_tokens: List[str], hyp_tokens: List[str]) -> List[Tuple[int,int]]:
    """
    Recorre cada token de ref_tokens y busca el primer índice j >= last_j en hyp_tokens
    tal que token_equal. Así garantizamos monotonicidad en j. Retorna lista de (i,j)
    para los i que sí encontraron "par". Los que no quedan sin emparejar.
    """
    pairs: List[Tuple[int,int]] = []
    last_j = 0
    n, m = len(ref_tokens), len(hyp_tokens)
    for i, tok in enumerate(ref_tokens):
        found = False
        for j in range(last_j, m):
            if token_equal(tok, hyp_tokens[j]):
                pairs.append((i, j))
                last_j = j + 1
                found = True
                break
        # si no encuentra, dejamos i sin par
    return pairs


# ---------- construcción de filas ----------

def build_rows(ref: str, hyp: str) -> List[List]:
    """
    1) Tokeniza y normaliza sin quitar puntuación (para mantener comas como separador de tokens).
    2) Crea listas ref_tok, hyp_tok.
    3) Genera ref_sw, hyp_sw eliminando stopwords.
    4) Aplica safe_dtw sobre listas sin stopwords.
    5) Reconstruye mapeo map_h convertido a índices sobre ref_tok→hyp_tok.
    6) Completa huecos con asignación monótona al siguiente token disponible en hyp_tok.
    7) Divide ref_tok en líneas de LINE_LEN, y para cada bloque calcula h_slice según min/max de map_h.
    8) Calcula WER con Levenshtein normalizado (sobre cadenas unidas), marca señal y tiempo.
    """
    # 1) tokenización y normalización (strip_punct=False para conservar comas/puntos como separación)
    ref_tok = normalize(ref, strip_punct=False).split()
    hyp_tok = normalize(hyp, strip_punct=False).split()

    # 2) generar listas sin stopwords
    ref_sw = [t for t in ref_tok if t not in STOP]
    hyp_sw = [t for t in hyp_tok if t not in STOP]

    # 3) índices originales de tokens que no son stopwords
    idx_r = [i for i, t in enumerate(ref_tok) if t not in STOP]
    idx_h = [j for j, t in enumerate(hyp_tok) if t not in STOP]

    # 4) DTW robusto sobre ref_sw y hyp_sw
    try:
        coarse_path = safe_dtw(ref_sw, hyp_sw, COARSE_W)
    except RuntimeError:
        coarse_path = fallback_pairs(ref_sw, hyp_sw)

    # 5) map_h: para cada índice i en ref_tok, -1 si no emparejado, o índice j de hyp_tok
    map_h = [-1] * len(ref_tok)
    for (ri, hi) in coarse_path:
        if ri < len(idx_r) and hi < len(idx_h):
            orig_i = idx_r[ri]
            hyp_i = idx_h[hi]
            if map_h[orig_i] == -1:
                map_h[orig_i] = hyp_i

    # 6) rellenar huecos monótonos: si map_h[i] == -1 pero hay próximo token en hyp_tok,
    #    asignar last_matched + 1 hasta cubrir (para no perder la última palabra).
    last = -1
    for i in range(len(map_h)):
        if map_h[i] != -1:
            last = map_h[i]
        else:
            if last + 1 < len(hyp_tok):
                last += 1
                map_h[i] = last
            # si last+1 está fuera de rango, map_h[i] queda -1

    # 7) dividir en bloques de LINE_LEN y armar filas
    rows: List[List] = []
    pos = 0
    line_id = 0

    while pos < len(ref_tok):
        block = ref_tok[pos : pos + LINE_LEN]
        span_start = pos
        span_end = min(pos + LINE_LEN, len(ref_tok))
        pos = span_end

        # obtener todos los índices de hyp para este bloque
        h_idxs = [map_h[k] for k in range(span_start, span_end) if map_h[k] != -1]
        if h_idxs:
            h_start = min(h_idxs)
            h_end = max(h_idxs) + 1  # exclusivo
            asr_line = " ".join(hyp_tok[h_start:h_end])
        else:
            asr_line = ""

        orig_line = " ".join(block)
        wer_val = Levenshtein.normalized_distance(orig_line, asr_line) if asr_line else 1.0
        flag = "✅" if wer_val <= WARN_WER else ("⚠️" if wer_val <= 0.20 else "❌")
        dur = round(len(asr_line.split()) / 3.0, 2)

        rows.append([line_id, flag, round(wer_val * 100, 1), dur, orig_line, asr_line])
        line_id += 1

    return rows


# ---------- GUI (igual que v4.4 salvo build_rows actualizado) ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-Audiolibro")
        self.geometry("1850x760")

        self.v_ref = tk.StringVar()
        self.v_asr = tk.StringVar()
        self.v_json = tk.StringVar()
        self.q = queue.Queue()

        top = ttk.Frame(self)
        top.pack(fill="x", padx=3, pady=2)

        ttk.Label(top, text="Guion:").grid(row=0, column=0, sticky="e")
        ttk.Entry(top, textvariable=self.v_ref, width=70).grid(row=0, column=1)
        ttk.Button(
            top,
            text="…",
            command=lambda: self.browse(self.v_ref, ("PDF/TXT", "*.pdf;*.txt")),
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
        ttk.Entry(top, textvariable=self.v_json, width=70).grid(row=2, column=1)
        ttk.Button(top, text="Abrir JSON…", command=self.load_json).grid(
            row=2, column=2
        )

        self.tree = ttk.Treeview(
            self,
            columns=("ID", "✓", "WER", "dur", "Original", "ASR"),
            show="headings",
            height=27,
        )
        for c, w in zip(
            ("ID", "✓", "WER", "dur", "Original", "ASR"),
            (50, 30, 60, 60, 800, 800),
        ):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill="both", expand=True, padx=3, pady=2)

        self.log_box = scrolledtext.ScrolledText(self, height=5, state="disabled")
        self.log_box.pack(fill="x", padx=3, pady=2)

        self.after(250, self._poll)

    def browse(self, var: tk.StringVar, ft):
        p = filedialog.askopenfilename(filetypes=[ft])
        if p:
            var.set(p)

    def log_msg(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.configure(state="disabled")
        self.log_box.see("end")

    def clear_table(self):
        self.tree.delete(*self.tree.get_children())

    def launch(self):
        if not (self.v_ref.get() and self.v_asr.get()):
            messagebox.showwarning("Falta info", "Selecciona guion y TXT ASR.")
            return
        self.clear_table()
        self.log_msg("⏳ Iniciando…")
        threading.Thread(target=self.worker, daemon=True).start()

    def worker(self):
        try:
            self.q.put("→ Leyendo guion…")
            ref = read_script(self.v_ref.get())

            self.q.put("→ TXT externo cargado")
            hyp = Path(self.v_asr.get()).read_text(encoding="utf8", errors="ignore")

            self.q.put("→ Alineando…")
            rows = build_rows(ref, hyp)

            out = Path(self.v_asr.get()).with_suffix(".qc.json")
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")

            self.q.put(("ROWS", rows))
            self.q.put(f"✔ Listo. Guardado en {out}")
            self.v_json.set(str(out))
        except Exception:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            self.q.put(buf.getvalue())

    def load_json(self):
        if not self.v_json.get():
            p = filedialog.askopenfilename(filetypes=[("QC JSON", "*.qc.json;*.json")])
            if not p:
                return
            self.v_json.set(p)
        try:
            rows = json.loads(Path(self.v_json.get()).read_text(encoding="utf8"))
            self.clear_table()
            for r in rows:
                self.tree.insert("", tk.END, values=r)
            self.log_msg(f"✔ Cargado {self.v_json.get()}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if isinstance(msg, tuple) and msg[0] == "ROWS":
                    for r in msg[1]:
                        self.tree.insert("", tk.END, values=r)
                else:
                    self.log_msg(str(msg))
        except queue.Empty:
            pass
        self.after(250, self._poll)


if __name__ == "__main__":
    App().mainloop()

# resync_json_gui.py – sincroniza tiempos ‘tc’ en un QC‑JSON usando un CSV palabra‑tiempo
# ----------------------------------------------------------------------------------
# ▸ Selecciona (GUI) un archivo .qc.json y el .words.csv con los time‑codes.
# ▸ Busca, para cada frase ASR, las primeras 3 palabras consecutivas dentro del
#   listado del CSV y usa su tiempo como "tc" (time‑code de inicio).
# ▸ Sustituye la columna «dur» por «tc» (segundos con 2 decimales).
# ▸ Guarda <nombre>.resync.json junto al original.
# ▸ Muestra barra de progreso y log con avisos de coincidencias no encontradas.
# ----------------------------------------------------------------------------------
# Requisitos: python‑3.9+, pandas (solo si lo tienes; el script funciona sin él).
# ----------------------------------------------------------------------------------

from __future__ import annotations
import json, re, sys, threading, tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk, messagebox
from typing import List, Tuple

###########################################################################
# Normalización de palabras (minúsculas, sin tildes, sin signos)
###########################################################################
import unicodedata

def norm(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-z0-9áéíóúüñ]+", " ", text)  # mantiene letras/num
    return text.strip()

###########################################################################
# Carga del CSV (acepta ; , tab o >1 espacios)
###########################################################################

def load_words_csv(path: Path) -> Tuple[List[str], List[float]]:
    words, starts = [], []
    with path.open("r", encoding="utf8", errors="ignore") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            # separador
            if ";" in line:
                t_str, w_raw = line.split(";", 1)
            elif "," in line:
                t_str, w_raw = line.split(",", 1)
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) < 2:
                    continue
                t_str, w_raw = parts
            # primer campo = número
            try:
                t = float(t_str.replace(",", "."))
            except ValueError:
                continue  # salta cabeceras u otras líneas
            w = norm(w_raw)
            if not w:
                continue
            starts.append(round(t, 2))
            words.append(w)
    return words, starts

###########################################################################
# Búsqueda de una ventana de n palabras dentro de la lista CSV
###########################################################################

def find_window_time(seq_words: List[str], seq_starts: List[float], phrase: str, n: int = 3) -> float | None:
    """Devuelve time‑code (float) de inicio si encuentra *n* palabras seguidas."""
    tokens = norm(phrase).split()
    if len(tokens) < n:
        return None
    needle_sets = [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    text_join = " ".join(seq_words)  # para búsqueda rápida
    pos = -1
    for window in needle_sets:
        # búsqueda exacta (espacios para garantizar borde de palabra)
        patt = rf"\b{re.escape(window)}\b"
        m = re.search(patt, text_join)
        if m:
            # cuántas palabras antes del match
            words_before = text_join[:m.start()].split()
            idx = len(words_before)
            return seq_starts[idx]
    return None

###########################################################################
# GUI simple
###########################################################################

class ResyncApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Re‑sincronizar QC‑JSON con CSV word‑timings")
        self.geometry("620x420")

        self.v_json = tk.StringVar()
        self.v_csv  = tk.StringVar()

        frm = ttk.Frame(self)
        frm.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm, text="QC JSON:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.v_json, width=60).grid(row=0, column=1)
        ttk.Button(frm, text="…", command=self.pick_json).grid(row=0, column=2)

        ttk.Label(frm, text="CSV words:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.v_csv, width=60).grid(row=1, column=1)
        ttk.Button(frm, text="…", command=self.pick_csv).grid(row=1, column=2)

        ttk.Button(frm, text="Re‑sincronizar", command=self.launch).grid(row=2, column=1, pady=8)

        self.pbar = ttk.Progressbar(self, length=580)
        self.pbar.pack(pady=4, padx=10)

        self.log = scrolledtext.ScrolledText(self, height=12, state="disabled")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0,10))

    # ---------- helper UI ----------
    def pick_json(self):
        p = filedialog.askopenfilename(filetypes=[("QC JSON", "*.json;*.qc.json")])
        if p: self.v_json.set(p)

    def pick_csv(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if p: self.v_csv.set(p)

    def log_msg(self, txt: str):
        self.log["state"] = "normal"
        self.log.insert("end", txt + "\n")
        self.log["state"] = "disabled"
        self.log.see("end")

    # ---------- main ----------
    def launch(self):
        pj, pc = self.v_json.get(), self.v_csv.get()
        if not (pj and pc):
            messagebox.showerror("Falta info", "Selecciona JSON y CSV")
            return
        threading.Thread(target=self.worker, args=(Path(pj), Path(pc)), daemon=True).start()

    def worker(self, json_path: Path, csv_path: Path):
        try:
            self.log_msg("Leyendo JSON…")
            rows = json.loads(json_path.read_text(encoding="utf8"))
            total = len(rows)
            self.log_msg(f"→ {total} filas")

            self.log_msg("Leyendo CSV word‑timings…")
            words, starts = load_words_csv(csv_path)
            if not words:
                messagebox.showerror("Error", "CSV no contiene datos utilizables")
                return
            self.log_msg(f"→ {len(words)} palabras cargadas")

            # Reemplazar dur por tc
            not_found = 0
            for i, row in enumerate(rows):
                asr = row[-1]
                tc = find_window_time(words, starts, asr)
                if tc is None:
                    not_found += 1
                else:
                    row[5] = f"{tc:.2f}"  # columna dur
                # barra de progreso
                if i % 5 == 0:
                    self.pbar["value"] = i/total*100
            self.pbar["value"] = 100

            out = json_path.with_suffix(".resync.json")
            out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf8")
            self.log_msg(f"✔ Terminado. Guardado en {out}")
            self.log_msg(f"   Filas SIN coincidencia: {not_found}")
        except Exception as exc:
            self.log_msg(f"ERROR: {exc}")
            raise

###########################################################################
if __name__ == "__main__":
    ResyncApp().mainloop()

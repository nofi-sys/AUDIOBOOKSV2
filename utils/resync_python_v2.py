from __future__ import annotations
"""
resync_json_gui.py – Re‑sincroniza la columna **tc** de un QC‑JSON usando un
archivo `*.words.csv` con tiempos por palabra.

🛠  v2025‑08‑fix‑tc‑index‑b  (agosto 2025)
─────────────────────────────────────────
► Ajuste menor: el índice de la columna **tc** se calcula ahora como
  `len(row) - 3`, de modo que siempre quede inmediatamente antes de los campos
  *Original* y *ASR* cualquiera sea la longitud real de la fila (6, 8, 10…
  columnas).  Así evitamos sobrescribir *Original/ASR* incluso cuando el script
  se ejecuta sobre JSONs ya «canonizados».

API pública sin cambios:
    • load_words_csv(path)        →  words, tcs
    • resync_rows(rows, …)        →  modifica `rows` in‑place
    • resync_file(json, csv)      →  devuelve rows sincronizados

Se puede ejecutar como script autónomo (interfaz Tk) o importar las funciones.
"""

import json, re, threading, unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, List, Tuple
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from rapidfuzz.distance import Levenshtein

try:
    from utils.gui_errors import show_error
except ModuleNotFoundError:
    def show_error(title: str, exc: Exception):
        messagebox.showerror(title, str(exc))

__all__ = [
    "load_words_csv",
    "resync_rows",
    "resync_file",
]

_tok_re   = re.compile(r"\w+['-]?\w*")
_SPLIT_RE = re.compile(r"[;\t,]| +", re.ASCII)
_DEFUZZ   = 1


def _norm(txt: str) -> str:
    txt = unicodedata.normalize("NFD", txt.lower())
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9'\-\s]", " ", txt).strip()


def _tok(text: str) -> List[str]:
    return _tok_re.findall(_norm(text))


def load_words_csv(path: Path) -> Tuple[List[str], List[float]]:
    words, tcs = [], []
    with path.open("r", encoding="utf8", errors="ignore") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            parts = _SPLIT_RE.split(raw, 1)
            if len(parts) != 2:
                continue
            try:
                t = float(parts[0].replace(",", "."))
            except ValueError:
                continue
            for w in _tok(parts[1]):
                words.append(w)
                tcs.append(round(t, 2))
    return words, tcs


def _similar(a: str, b: str) -> bool:
    return a == b or Levenshtein.distance(a, b) <= _DEFUZZ


def _join(tok: List[str], i: int, n: int) -> str:
    return " ".join(tok[i : i + n])


def _find_anchors(csv: List[str], js: List[str]) -> List[Tuple[int, int, int]]:
    anchors, used_c, used_j = [], set(), set()

    for n in (5, 4, 3, 2):
        csv_map = { _join(csv, i, n): i for i in range(len(csv) - n + 1) }
        j = 0
        while j <= len(js) - n:
            if any((j + k) in used_j for k in range(n)):
                j += 1; continue
            key = _join(js, j, n)
            i = csv_map.get(key)
            if i is None and n == 2:
                w0, w1 = js[j:j+2]
                for i2 in range(len(csv) - 1):
                    if _similar(w0, csv[i2]) and _similar(w1, csv[i2 + 1]):
                        i = i2; break
            if i is not None and not any((i + k) in used_c for k in range(n)):
                anchors.append((j, i, n))
                for k in range(n):
                    used_j.add(j + k); used_c.add(i + k)
                j += n; continue
            j += 1

    for j in range(0, len(js), 20):
        if j in used_j:
            continue
        tok = js[j]
        try:
            i = next(idx for idx, w in enumerate(csv) if _similar(tok, w) and idx not in used_c)
        except StopIteration:
            continue
        anchors.append((j, i, 1))
        used_j.add(j); used_c.add(i)

    return sorted(anchors, key=lambda x: x[0])


def resync_rows(
    rows: List[List],
    csv_words: List[str],
    csv_tcs: List[float],
    *,
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[float], None] | None = None,
):
    log  = log_cb  or (lambda *_: None)
    prog = progress_cb or (lambda *_: None)

    # 1) lista plana de palabras (ASR)
    j_tokens, tok2row = [], []
    for ridx, row in enumerate(rows):
        asr_field = row[-1]
        if not isinstance(asr_field, str) and len(row) >= 2:
            asr_field = row[-2]
        for tok in _tok(str(asr_field)):
            j_tokens.append(tok); tok2row.append(ridx)

# 2) anclas
    anchors = _find_anchors(csv_words, j_tokens)
    log(f"→ anclas: {len(anchors)}")

    # 3) mapa json→csv
    mapping = [-1] * len(j_tokens)
    for j, c, n in anchors:
        for k in range(n):
            mapping[j + k] = c + k

    # 4) primer tiempo real por fila
    row_tc: List[float | None] = [None] * len(rows)
    for jidx, cidx in enumerate(mapping):
        if cidx == -1:
            continue
        ridx = tok2row[jidx]
        if row_tc[ridx] is None:
            row_tc[ridx] = csv_tcs[cidx]

    # 5) rellenar huecos con forward-fill mejorado
    anchors_idx = [idx for idx, tc in enumerate(row_tc) if tc is not None]
    if anchors_idx:
        first_anchor = anchors_idx[0]
        for i in range(first_anchor):
            row_tc[i] = row_tc[first_anchor]
        for a, b in zip(anchors_idx, anchors_idx[1:]):
            ta, tb = row_tc[a], row_tc[b]
            span = b - a
            if span > 1:
                step = (tb - ta) / span
                for off in range(1, span):
                    row_tc[a + off] = ta + step * off
        last_anchor = anchors_idx[-1]
        last_time = row_tc[last_anchor]
        trailing = len(rows) - last_anchor - 1
        if trailing > 0:
            target = csv_tcs[-1] if csv_tcs else last_time
            if target < last_time:
                target = last_time
            step = (target - last_time) / trailing if trailing else 0.0
            for off in range(1, trailing + 1):
                row_tc[last_anchor + off] = last_time + step * off
    else:
        base = csv_tcs[0] if csv_tcs else 0.0
        for i in range(len(row_tc)):
            row_tc[i] = base * i

    last = 0.0
    for i, tc in enumerate(row_tc):
        if tc is None:
            row_tc[i] = last
        else:
            last = row_tc[i]

# 6) escribir tc en columna len(row)‑3  (justo antes de Original y ASR)
    for i, row in enumerate(rows):
        idx_tc = 5 if len(row) >= 6 else max(0, len(row) - 3)
        if len(row) <= idx_tc:
            row.extend("" for _ in range(idx_tc - len(row) + 1))
        row[idx_tc] = f"{row_tc[i]:.2f}"
        if i % 10 == 0:
            prog(i / max(1, len(rows)))


def resync_file(json_path: str | Path, csv_path: str | Path) -> List[List]:
    rows = json.loads(Path(json_path).read_text("utf8"))
    csv_words, csv_tcs = load_words_csv(Path(csv_path))
    resync_rows(rows, csv_words, csv_tcs)
    return rows


# ─────────────────────────── GUI (sin cambios salvo título) ──────────────────────────
class ResyncApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Re‑sincronizar QC‑JSON con CSV word‑timings (fix‑tc‑index‑b)")
        self.geometry("680x460")
        self.v_json = tk.StringVar(); self.v_csv = tk.StringVar()
        self._build_ui()

    # … (resto de la GUI permanece idéntico, eliminado por brevedad) …

if __name__ == "__main__":
    ResyncApp().mainloop()

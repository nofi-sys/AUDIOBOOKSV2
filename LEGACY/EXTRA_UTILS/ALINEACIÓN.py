#!/usr/bin/env python3
import sys
import os
import re
import csv
import unicodedata
from collections import namedtuple, defaultdict

Anchor = namedtuple("Anchor", ["ref_idx", "asr_idx", "n"])

# limites para decidir si una ancla es razonable
MAX_SEGMENT_RATIO = 10.0      # cuantas veces puede ser mas largo un segmento que el otro
MAX_FORWARD_GAP = 400         # maximo de tokens ASR entre anclas hacia adelante
GAP_REF_THRESHOLD = 300       # si avanzo esto en ref sin hallar ancla, salto a busqueda desde el final


# --------- normalizacion / tokenizacion ---------

def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def roman_to_int(s: str):
    """Convierte numeros romanos simples a entero; devuelve None si no es valido."""
    s = s.lower()
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for ch in reversed(s):
        v = values.get(ch)
        if v is None:
            return None
        if v < prev:
            total -= v
        else:
            total += v
            prev = v
    return total


def normalize_token(tok: str) -> str:
    t = strip_accents(tok.lower())
    # solo letras y numeros
    t = re.sub(r"[^0-9a-z]+", "", t)
    if not t:
        return ""
    # intento de romano
    if re.fullmatch(r"[ivxlcdm]+", t):
        val = roman_to_int(t)
        if val is not None:
            return str(val)
    return t


def tokenize(text: str):
    """
    Devuelve (tokens_raw, tokens_norm).
    Usa solo secuencias alfanumericas como 'palabras' (la puntuacion se descarta).
    """
    raw = re.findall(r"\w+", text, flags=re.UNICODE)
    norm = [normalize_token(t) for t in raw]
    return raw, norm


# --------- busqueda de anclas globales 5-4-3 ---------

def build_index(tokens_norm, n: int):
    index = defaultdict(list)
    limit = len(tokens_norm) - n + 1
    for j in range(limit):
        key = tuple(tokens_norm[j:j+n])
        index[key].append(j)
    return index


def find_anchors_forward(ref_norm, asr_norm):
    anchors = []
    NR = len(ref_norm)
    NA = len(asr_norm)

    last_ref = 0
    last_asr = 0
    first_anchor = True
    ref_since_last = 0

    for n in (5, 4, 3):
        if NR < n or NA < n:
            continue
        indexA = build_index(asr_norm, n)
        i = last_ref
        while i <= NR - n:
            key = tuple(ref_norm[i:i+n])
            positions = indexA.get(key)
            if not positions:
                i += 1
                ref_since_last += 1
                if not first_anchor and ref_since_last > GAP_REF_THRESHOLD:
                    return anchors, last_ref, last_asr
                continue

            candidate_j = None
            for j in positions:
                if j < last_asr:
                    continue

                if first_anchor:
                    candidate_j = j
                    break

                if (j - last_asr) > MAX_FORWARD_GAP:
                    break

                seg_ref = i - last_ref if i > last_ref else 1
                seg_asr = j - last_asr if j > last_asr else 1
                ratio = seg_asr / max(seg_ref, 1)

                if 1.0 / MAX_SEGMENT_RATIO <= ratio <= MAX_SEGMENT_RATIO:
                    candidate_j = j
                    break

            if candidate_j is None:
                i += 1
                ref_since_last += 1
                if not first_anchor and ref_since_last > GAP_REF_THRESHOLD:
                    return anchors, last_ref, last_asr
                continue

            anchors.append(Anchor(i, candidate_j, n))
            first_anchor = False
            last_ref = i + n
            last_asr = candidate_j + n
            ref_since_last = 0
            i = last_ref

    return anchors, last_ref, last_asr


def find_anchors_backward(ref_norm, asr_norm, ref_limit, asr_limit):
    """
    Busca anclas desde el final hacia atras en ref_norm[:ref_limit] y asr_norm[:asr_limit].
    """
    ref_suffix = ref_norm[:ref_limit][::-1]
    asr_suffix = asr_norm[:asr_limit][::-1]

    back_anchors, _, _ = find_anchors_forward(ref_suffix, asr_suffix)

    anchors = []
    for a in back_anchors:
        orig_ref = ref_limit - (a.ref_idx + a.n)
        orig_asr = asr_limit - (a.asr_idx + a.n)
        anchors.append(Anchor(orig_ref, orig_asr, a.n))

    anchors.sort(key=lambda x: x.ref_idx)
    return anchors


def find_anchors_bidirectional(ref_norm, asr_norm):
    NR = len(ref_norm)
    NA = len(asr_norm)

    anchors_forward, last_ref, last_asr = find_anchors_forward(ref_norm, asr_norm)

    if last_ref >= NR or last_asr >= NA:
        return anchors_forward

    anchors_backward = find_anchors_backward(ref_norm, asr_norm, NR, NA)

    all_anchors = anchors_forward + anchors_backward
    all_anchors.sort(key=lambda a: (a.ref_idx, a.asr_idx))

    filtered = []
    last_r_end = -1
    last_a_end = -1
    for a in all_anchors:
        if a.ref_idx >= last_r_end and a.asr_idx >= last_a_end:
            filtered.append(a)
            last_r_end = a.ref_idx + a.n
            last_a_end = a.asr_idx + a.n

    return filtered


# --------- alineacion DP entre anclas ---------

def dp_align_segment(ref_norm, asr_norm, r_start, r_end, a_start, a_end):
    """
    Alinea ref_norm[r_start:r_end] con asr_norm[a_start:a_end] por DP.
    Devuelve lista de pares (ref_idx | None, asr_idx | None) en orden.
    """
    m = r_end - r_start
    n = a_end - a_start
    if m < 0 or n < 0:
        raise ValueError(f"Segmento negativo: ref {r_start}-{r_end}, asr {a_start}-{a_end}")

    if m == 0 and n == 0:
        return []
    if m == 0:
        return [(None, a_start + j) for j in range(n)]
    if n == 0:
        return [(r_start + i, None) for i in range(m)]

    # tablas DP
    dp = [[0]*(n+1) for _ in range(m+1)]
    back = [[None]*(n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, n+1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, m+1):
        for j in range(1, n+1):
            r_tok = ref_norm[r_start + i - 1]
            a_tok = asr_norm[a_start + j - 1]

            cost_match = 0 if r_tok == a_tok else 1

            # match / sub
            best = dp[i-1][j-1] + cost_match
            op = "M" if cost_match == 0 else "S"

            # del
            cand = dp[i-1][j] + 1
            if cand < best:
                best = cand
                op = "D"

            # ins
            cand = dp[i][j-1] + 1
            if cand < best:
                best = cand
                op = "I"

            dp[i][j] = best
            back[i][j] = op

    # backtracking
    pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            pairs.append((r_start + i - 1, a_start + j - 1))
            i -= 1
            j -= 1
        elif op == "D":
            pairs.append((r_start + i - 1, None))
            i -= 1
        elif op == "I":
            pairs.append((None, a_start + j - 1))
            j -= 1
        else:
            raise RuntimeError("Backtracking inconsistente")

    pairs.reverse()
    return pairs


def align_with_anchors(ref_raw, ref_norm, asr_raw, asr_norm, anchors):
    """
    Devuelve lista de triples (ref_idx | None, asr_idx | None, is_anchor).
    """
    aligned = []

    NR = len(ref_norm)
    NA = len(asr_norm)
    anchors = sorted(anchors, key=lambda a: a.ref_idx)

    if not anchors:
        for ref_idx, asr_idx in dp_align_segment(ref_norm, asr_norm, 0, NR, 0, NA):
            aligned.append((ref_idx, asr_idx, False))
        return aligned

    prev_r = 0
    prev_a = 0

    for anchor in anchors:
        aligned_seg = dp_align_segment(
            ref_norm, asr_norm,
            prev_r, anchor.ref_idx,
            prev_a, anchor.asr_idx,
        )
        aligned.extend((ri, aj, False) for (ri, aj) in aligned_seg)

        for k in range(anchor.n):
            aligned.append((anchor.ref_idx + k, anchor.asr_idx + k, True))

        prev_r = anchor.ref_idx + anchor.n
        prev_a = anchor.asr_idx + anchor.n

    aligned_seg = dp_align_segment(
        ref_norm, asr_norm,
        prev_r, NR,
        prev_a, NA,
    )
    aligned.extend((ri, aj, False) for (ri, aj) in aligned_seg)

    return aligned


# --------- main / CSV ---------

def align_files(guion_path: str, asr_path: str, out_path: str) -> None:
    with open(guion_path, "r", encoding="utf-8") as f:
        ref_text = f.read()
    with open(asr_path, "r", encoding="utf-8") as f:
        asr_text = f.read()

    ref_raw, ref_norm = tokenize(ref_text)
    asr_raw, asr_norm = tokenize(asr_text)

    anchors = find_anchors_bidirectional(ref_norm, asr_norm)
    aligned = align_with_anchors(ref_raw, ref_norm, asr_raw, asr_norm, anchors)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ref_word", "asr_word", "is_anchor"])
        for ref_idx, asr_idx, is_anchor in aligned:
            ref_word = ref_raw[ref_idx] if ref_idx is not None else ""
            asr_word = asr_raw[asr_idx] if asr_idx is not None else ""
            writer.writerow([ref_word, asr_word, "1" if is_anchor else "0"])


def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("Alineacion (legacy)")

    guion_var = tk.StringVar()
    asr_var = tk.StringVar()
    out_var = tk.StringVar()

    def browse_guion():
        path = filedialog.askopenfilename(
            title="Seleccionar guion",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            guion_var.set(path)
            if not out_var.get():
                out_var.set(os.path.splitext(path)[0] + "_alineado.csv")

    def browse_asr():
        path = filedialog.askopenfilename(
            title="Seleccionar ASR",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            asr_var.set(path)

    def browse_out():
        path = filedialog.asksaveasfilename(
            title="Guardar CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if path:
            out_var.set(path)

    def run_alignment_gui():
        guion_path = guion_var.get().strip()
        asr_path = asr_var.get().strip()
        out_path = out_var.get().strip()

        if not guion_path or not asr_path or not out_path:
            messagebox.showerror("Faltan datos", "Completa las rutas de entrada y salida.")
            return
        if not os.path.isfile(guion_path):
            messagebox.showerror("No existe", f"No se encuentra el archivo del guion:\n{guion_path}")
            return
        if not os.path.isfile(asr_path):
            messagebox.showerror("No existe", f"No se encuentra el archivo de ASR:\n{asr_path}")
            return

        try:
            align_files(guion_path, asr_path, out_path)
        except Exception as exc:  # pragma: no cover - gui only
            messagebox.showerror("Error", str(exc))
            return

        messagebox.showinfo("Listo", f"Archivo guardado en:\n{out_path}")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Guion").grid(row=0, column=0, sticky="w")
    tk.Entry(frame, textvariable=guion_var, width=50).grid(row=0, column=1, padx=5)
    tk.Button(frame, text="Buscar", command=browse_guion).grid(row=0, column=2)

    tk.Label(frame, text="ASR").grid(row=1, column=0, sticky="w")
    tk.Entry(frame, textvariable=asr_var, width=50).grid(row=1, column=1, padx=5)
    tk.Button(frame, text="Buscar", command=browse_asr).grid(row=1, column=2)

    tk.Label(frame, text="Salida CSV").grid(row=2, column=0, sticky="w")
    tk.Entry(frame, textvariable=out_var, width=50).grid(row=2, column=1, padx=5)
    tk.Button(frame, text="Guardar como", command=browse_out).grid(row=2, column=2)

    tk.Button(frame, text="Alinear", command=run_alignment_gui).grid(row=3, column=1, pady=10)

    root.mainloop()


def main():
    if len(sys.argv) == 4:
        align_files(sys.argv[1], sys.argv[2], sys.argv[3])
        return
    if len(sys.argv) > 1:
        print("Uso: python ALINEACION.py guion.txt asr.txt salida.csv")
        sys.exit(1)
    launch_gui()


if __name__ == "__main__":
    main()

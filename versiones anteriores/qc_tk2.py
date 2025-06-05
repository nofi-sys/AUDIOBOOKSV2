#!/usr/bin/env python
# qc_tk.py – QC audiolibros con alineación inteligente • GUI en Tkinter
# Autor 2025 • MIT
#
# pip install torch whisperx pandas rapidfuzz unidecode pdfplumber tqdm

import io, json, queue, re, subprocess, tempfile, threading, traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

import torch, whisperx, pandas as pd, pdfplumber, unidecode
from rapidfuzz.distance import Levenshtein

# ═════════════════════════════ UTILIDADES ════════════════════════════════
DIGITS = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()
ABBR = {
    "dra": "doctora", "dr": "doctor",
    "sra": "señora",   "sr": "señor",
    "av": "avenida",   "kg": "kilos",
    "bsas": "buenos aires",
}

def expand_abbr(w: str) -> str:
    base = w.rstrip(".")
    return ABBR.get(base, base)

def normalize(txt: str) -> str:
    txt = unidecode.unidecode(txt.lower())
    txt = re.sub(r"\b(\d)\b", lambda m: DIGITS[int(m.group(1))], txt)
    txt = re.sub(r"[^a-z0-9¿¡\?\!\.,:\s]", " ", txt)
    txt = " ".join(expand_abbr(w) for w in txt.split())
    return txt

ABBR_SAFE = ("dr", "dra", "sr", "sra", "mr", "mrs", "av", "etc")

def smart_segment(txt: str, max_words: int = 15):
    """Divide en frases; ignora puntos de abreviaturas y corta cada ~max_words."""
    # 1) protege los puntos de abreviaturas -> dr#  / dra#
    for a in ABBR_SAFE:
        txt = re.sub(rf"\b{a}\.", a + "#", txt)
    # 2) split en ., !, ?
    chunks = re.split(r"[.!?]+\s*", txt)
    # 3) restaura puntos y vuelve a fragmentar por longitud
    segs = []
    for ch in chunks:
        if not ch.strip():
            continue
        ch = ch.replace("#", ".")
        buf = []
        for w in ch.split():
            buf.append(w)
            if len(buf) >= max_words:
                segs.append(" ".join(buf))
                buf = []
        if buf:
            segs.append(" ".join(buf))
    return segs

def ffmpeg_wav(src: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-vn", tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp

def read_text(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        with pdfplumber.open(p) as pdf:
            return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    return p.read_text(encoding="utf8", errors="ignore")

# ═══════ Alineación de bloques (búsqueda ventana 2×2) ════════════════════
def align_blocks(ref, hyp, win: int = 2):
    i = j = 0
    out = []
    while i < len(ref) and j < len(hyp):
        best = (1.1, 1, 1)         # WER, di, dj
        for di in range(1, win + 1):
            for dj in range(1, win + 1):
                if i + di <= len(ref) and j + dj <= len(hyp):
                    r = " ".join(ref[i:i + di])
                    h = " ".join(hyp[j:j + dj])
                    w = Levenshtein.normalized_distance(r, h)
                    if w < best[0]:
                        best = (w, di, dj)
        w, di, dj = best
        out.append(( " ".join(ref[i:i + di]),
                     " ".join(hyp[j:j + dj]),
                     w))
        i += di
        j += dj
    # resto
    for r in ref[i:]:
        out.append((r, "", 1.0))
    for h in hyp[j:]:
        out.append(("", h, 1.0))
    return out

def classify(wer: float, dur: float):
    if not dur or dur < 0.2:
        return "❌"
    if wer <= 0.06:
        return "✅"
    if wer <= 0.35:
        return "⚠️"
    return "❌"

# ═══════════════ HILO WORKER (genera JSON) ═══════════════════════════════
def build_json(audio, script, txt_asr, model, q):
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        q.put("→ Convirtiendo audio…")
        wav = ffmpeg_wav(audio)

        q.put("→ Analizando guion…")
        ref = smart_segment(normalize(read_text(script)))

        if txt_asr:
            q.put("→ Usando TXT externo…")
            hyp_full = normalize(Path(txt_asr).read_text(encoding="utf8"))
        else:
            q.put("→ Transcribiendo con WhisperX… (paciencia)")
            wx = whisperx.load_model(model, device=dev)
            segs = wx.transcribe(wav, batch_size=16)["segments"]
            m_a, meta = whisperx.load_align_model("es", dev)
            segs = whisperx.align(segs, m_a, meta, wav, dev)
            hyp_full = normalize(" ".join(s["text"] for s in segs))

        hyp = smart_segment(hyp_full)

        q.put("→ Alineando bloques…")
        pairs = align_blocks(ref, hyp, win=2)

        rows = []
        for idx, (r, h, w) in enumerate(pairs):
            dur = len(h.split()) / 3 if h else 0
            rows.append([idx, classify(w, dur), round(w * 100, 1),
                         f"{dur:.2f}", r, h])

        df = pd.DataFrame(rows,
                          columns=["ID", "✓", "WER", "dur", "Original", "ASR"])
        out = Path(audio).with_suffix(".qc.json")
        df.to_json(out, orient="records", force_ascii=False, indent=2)
        q.put(f"✔ JSON generado: {out}")
    except Exception:
        buf = io.StringIO()
        traceback.print_exc(file=buf)
        q.put(buf.getvalue())

# ═════════════════════════ GUI (Tkinter) ════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QC-WhisperX • Tkinter")
        self.resizable(False, False)

        self.var_audio = tk.StringVar()
        self.var_text = tk.StringVar()
        self.var_asr = tk.StringVar()
        self.var_model = tk.StringVar(value="small")
        self.q = queue.Queue()

        frm = ttk.Frame(self, padding=10); frm.grid()

        ttk.Label(frm, text="Audio").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_audio,
                  width=55).grid(row=0, column=1)
        ttk.Button(frm, text="…", command=self.pick_audio).grid(row=0, column=2)

        ttk.Label(frm, text="Guion PDF/TXT").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_text,
                  width=55).grid(row=1, column=1)
        ttk.Button(frm, text="…", command=self.pick_text).grid(row=1, column=2)

        ttk.Label(frm, text="Transcripción TXT").grid(row=2, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.var_asr,
                  width=55).grid(row=2, column=1)
        ttk.Button(frm, text="…", command=self.pick_asr).grid(row=2, column=2)

        ttk.Label(frm, text="Modelo").grid(row=3, column=0, sticky="e")
        ttk.Combobox(frm, textvariable=self.var_model,
                     values=("tiny", "base", "small", "medium", "large"),
                     width=8, state="readonly").grid(row=3, column=1, sticky="w")

        ttk.Button(frm, text="▶ Ejecutar", width=12,
                   command=self.run).grid(row=3, column=2, pady=5)

        self.log = scrolledtext.ScrolledText(frm, width=80, height=18,
                                             state="disabled")
        self.log.grid(row=4, column=0, columnspan=3, pady=5)

        self.after(250, self.poll_q)

    # file pickers
    def pick_audio(self):
        f = filedialog.askopenfilename(
            filetypes=[("Audio/Video",
                        "*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.aac;*.mp4")])
        if f:
            self.var_audio.set(f)

    def pick_text(self):
        f = filedialog.askopenfilename(filetypes=[("Guion", "*.pdf;*.txt")])
        if f:
            self.var_text.set(f)

    def pick_asr(self):
        f = filedialog.askopenfilename(filetypes=[("TXT", "*.txt")])
        if f:
            self.var_asr.set(f)

    # run worker
    def run(self):
        if not self.var_audio.get() or not self.var_text.get():
            messagebox.showwarning("Falta información",
                                   "Selecciona audio y guion.")
            return
        self.log_clear()
        threading.Thread(
            target=build_json,
            args=(self.var_audio.get(),
                  self.var_text.get(),
                  self.var_asr.get() or None,
                  self.var_model.get(),
                  self.q),
            daemon=True).start()

    # queue / log
    def poll_q(self):
        try:
            while True:
                self.log_print(self.q.get_nowait())
        except queue.Empty:
            pass
        self.after(250, self.poll_q)

    def log_print(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.configure(state="disabled")
        self.log.yview("end")

    def log_clear(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

# ═══════════════════════════ main ═══════════════════════════════════════
if __name__ == "__main__":
    App().mainloop()

from __future__ import annotations
"""Simple GUI + CLI to produce **word‑level timestamps** (CSV) with faster‑whisper.

Changes (2025‑06‑18)
--------------------
*   **Eliminado** el argumento inexistente `duration=` en la API Python.
*   Nuevo *modo prueba* recorta el audio a 60 s con FFmpeg antes de transcribir.
*   Mensajes de progreso + ETA se mantienen.
*   Fallback elegante si `ffprobe` / FFmpeg no están instalados.

Uso rápido
----------
```bash
python word_timed_transcriber.py                     # GUI (si hay $DISPLAY)
python word_timed_transcriber.py audio.mp3 --test    # CLI, sólo 1 min
```
"""
from pathlib import Path
from threading import Thread
from time import monotonic
import argparse
import tempfile
import subprocess
import queue
import sys
import os
import csv

from faster_whisper import WhisperModel
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────
# GUI only imports when needed (avoid tkinter on headless cli)
# ──────────────────────────────────────────────────────────────
USE_GUI = not (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
if USE_GUI:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, scrolledtext
    from utils.gui_errors import show_error

# ---------------------------------------------------------------------------
# Helpers ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def have_ffmpeg() -> bool:
    """Return True if ffmpeg binary seems available."""
    return subprocess.call(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def cut_first_minute(src: Path) -> Path:
    """Create a temporary 60‑second clip using FFmpeg and return the path."""
    if not have_ffmpeg():
        raise RuntimeError("FFmpeg no encontrado en PATH – requirido para modo prueba")
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)  # only path is needed
    tmp = Path(tmp_name)
    cmd = [
        "ffmpeg", "-y", "-i", str(src), "-t", "60",  # cortar 60 s
        "-ac", "1", "-ar", "16000",                    # mono 16 kHz (rápido y suficiente)
        str(tmp),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp


def write_csv(out_path: Path, items: list[tuple[float, str]]):
    with out_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter=";")
        for t, word in items:
            w.writerow([f"{t:.2f}", word])


# ---------------------------------------------------------------------------
# Core transcription (shared by GUI & CLI)
# ---------------------------------------------------------------------------

def _probe_duration(path: Path) -> float:
    """Return audio duration in seconds using ffprobe (if available)."""
    try:
        from ffmpeg import probe  # type: ignore

        return float(probe(str(path))["format"]["duration"])
    except Exception:
        return 0.0  # unknown – progress will be indeterminate


def transcribe_audio(path: Path, test_mode: bool, q: "queue.Queue[str]" | None = None):
    """Return list[(time, word)].  If *q* present emit progress events."""
    # ‑‑‑ preparar audio ----------------------------------------------------
    tmp_path: Path | None = None
    if test_mode:
        tmp_path = cut_first_minute(path)
        path_for_model = tmp_path
    else:
        path_for_model = path

    duration = _probe_duration(path_for_model)
    if test_mode and duration == 0.0:
        duration = 60.0  # asumido

    # ‑‑‑ modelo ------------------------------------------------------------
    model = WhisperModel("base", device="auto", compute_type="int8")

    segments, _ = model.transcribe(
        str(path_for_model),
        word_timestamps=True,
        beam_size=1,
        vad_filter=True,
    )

    words: list[tuple[float, str]] = []
    start_time = monotonic()
    bar: tqdm | None = None
    if not USE_GUI and duration:
        bar = tqdm(total=duration, unit="s", unit_scale=True, desc="Transcribing")

    for seg in segments:
        for w in seg.words:
            words.append((max(0.0, w.start), w.word.strip()))
        # feedback ---------------------------------------------------------
        if q or bar:
            if duration:
                pct = int(100 * seg.end / duration)
                pct = max(1, min(100, pct))
                elapsed = monotonic() - start_time
                eta = (elapsed / seg.end * (duration - seg.end)) if seg.end else 0
                if q:
                    q.put(("PROGRESS", pct, eta))
                if bar:
                    bar.n = min(bar.total, seg.end)
                    bar.set_postfix_str(f"ETA {eta:5.1f}s")
                    bar.refresh()
    if bar:
        bar.close()

    if q:
        q.put(("DONE", words))

    if tmp_path is not None and tmp_path.exists():
        tmp_path.unlink(missing_ok=True)

    return words


# ---------------------------------------------------------------------------
# CLI entry – used when a filename is passed directly
# ---------------------------------------------------------------------------

def cli_main():
    p = argparse.ArgumentParser(description="Word‑level transcription → CSV")
    p.add_argument("audio", help="audio file")
    p.add_argument("--test", action="store_true", help="only first minute")
    args = p.parse_args()

    words = transcribe_audio(Path(args.audio), args.test)
    out = Path(args.audio).with_suffix(".words.csv")
    write_csv(out, words)
    print("\nSaved", out)


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

if USE_GUI:

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Word‑timed Whisper")
            self.geometry("500x260")

            self.v_audio = tk.StringVar()
            self.v_test = tk.BooleanVar(value=False)
            self.q: queue.Queue = queue.Queue()

            frm = ttk.Frame(self)
            frm.pack(padx=10, pady=10, fill="x")

            ttk.Label(frm, text="Audio file:").grid(row=0, column=0, sticky="e")
            ttk.Entry(frm, textvariable=self.v_audio, width=50).grid(row=0, column=1)
            ttk.Button(frm, text="…", command=self.browse).grid(row=0, column=2)

            ttk.Checkbutton(frm, text="Modo prueba (1 min)", variable=self.v_test).grid(row=1, column=1, sticky="w")

            ttk.Button(frm, text="Transcribir", command=self.launch).grid(row=2, column=1, pady=5)

            # Progress bar & ETA
            self.pb = ttk.Progressbar(frm, length=400)
            self.pb.grid(row=3, column=0, columnspan=3, pady=4)
            self.eta_lbl = ttk.Label(frm, text="")
            self.eta_lbl.grid(row=4, column=0, columnspan=3)

            self.log = scrolledtext.ScrolledText(self, height=6, state="disabled")
            self.log.pack(fill="both", expand=True, padx=10, pady=5)

            self.after(250, self._poll)

        # ---------- GUI helpers
        def browse(self):
            p = filedialog.askopenfilename(filetypes=[("Audio", "*.mp3;*.wav;*.m4a;*.flac;*.ogg;*.aac;*.mp4")])
            if p:
                self.v_audio.set(p)

        def log_msg(self, msg):
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.configure(state="disabled")
            self.log.see("end")

        # ---------- transcription thread
        def launch(self):
            if not self.v_audio.get():
                messagebox.showwarning("Falta archivo", "Selecciona audio primero")
                return
            if self.v_test.get() and not have_ffmpeg():
                show_error("FFmpeg faltante", RuntimeError("El modo prueba necesita FFmpeg en PATH"))
                return
            self.pb["value"] = 0
            self.eta_lbl["text"] = ""
            self.log_msg("⏳ Transcribiendo…")
            Thread(target=self._worker, daemon=True).start()

        def _worker(self):
            try:
                transcribe_audio(Path(self.v_audio.get()), self.v_test.get(), self.q)
            except Exception as exc:
                self.q.put(("ERROR", str(exc)))

        # ---------- polling queue from thread
        def _poll(self):
            try:
                while True:
                    t = self.q.get_nowait()
                    if t[0] == "PROGRESS":
                        pct, eta = t[1:]
                        self.pb["value"] = pct
                        self.eta_lbl["text"] = f"ETA ~{eta:0.1f}s" if eta else "Procesando…"
                    elif t[0] == "DONE":
                        words = t[1]
                        out = Path(self.v_audio.get()).with_suffix(".words.csv")
                        write_csv(out, words)
                        self.pb["value"] = 100
                        self.eta_lbl["text"] = "Completado"
                        self.log_msg(f"✔ CSV guardado en {out}")
                    elif t[0] == "ERROR":
                        self.log_msg("❌ " + t[1])
            except queue.Empty:
                pass
            self.after(250, self._poll)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1 and not USE_GUI:
        cli_main()
    else:
        if USE_GUI:
            App().mainloop()
        else:
            print("Para entorno sin GUI pasa un archivo de audio como argumento.")


if __name__ == "__main__":
    main()

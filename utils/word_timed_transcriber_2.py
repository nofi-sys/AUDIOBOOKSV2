# word_timed_transcriber.py  (versión 2025-06-19)
from __future__ import annotations
import argparse, csv, os, queue, subprocess, sys, tempfile
from pathlib import Path
from threading import Thread
from time import monotonic

from tqdm.auto import tqdm
from faster_whisper import WhisperModel
import torch

# ───────────────────── utilidades básicas ─────────────────────
def have_ffmpeg() -> bool:
    return subprocess.call(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) == 0


def cut_first_minute(src: Path) -> Path:
    if not have_ffmpeg():
        raise RuntimeError("FFmpeg requerido para modo prueba")
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp = Path(name)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-t", "60",
           "-ac", "1", "-ar", "16000", str(tmp)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp


def write_csv(out_path: Path, items: list[tuple[float, str]]):
    with out_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter=";")
        for t, word in items:
            w.writerow([f"{t:.2f}", word])


def _probe_duration(path: Path) -> float:
    """Duración en segundos usando ffprobe (intenta módulo y shell)."""
    try:
        # vía ffmpeg-python, si está
        from ffmpeg import probe  # type: ignore
        return float(probe(str(path))["format"]["duration"])
    except Exception:
        # fallback shell
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of", "default=nw=1:nk=1", str(path)],
                text=True,
            ).strip()
            return float(out)
        except Exception:
            return 0.0

# ───────────────────── transcripción núcleo ───────────────────
def transcribe_audio(
    path: Path,
    test_mode: bool = False,
    use_vad: bool = True,
    *,
    script_path: str | None = None,
    model_name: str = "large-v3",
    q: "queue.Queue[str]" | None = None,
):
    tmp: Path | None = None
    if test_mode:
        tmp = cut_first_minute(path)
        src = tmp
    else:
        src = path

    duration = _probe_duration(src)

    hotwords = None
    initial_prompt = None
    if script_path:
        try:
            from text_utils import read_script, extract_word_list

            script_text = read_script(script_path)
            tokens = script_text.split()
            initial_prompt = " ".join(tokens[:200]) if tokens else None
            words = extract_word_list(script_text)
            if words:
                hotwords = " ".join(words)
        except Exception:
            pass

    model = WhisperModel(model_name, device="auto", compute_type="int8")

    seg_gen, _info = model.transcribe(
        str(src),
        word_timestamps=True,
        beam_size=7,
        vad_filter=use_vad,
        vad_parameters=dict(min_silence_duration_ms=300),
        temperature=0.0,
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )

    words: list[tuple[float, str]] = []
    start_clock = monotonic()
    bar: tqdm | None = None
    if not USE_GUI and duration:
        bar = tqdm(total=duration, unit="s", unit_scale=True, desc="Transcribing")

    try:
        for seg in seg_gen:
            for w in seg.words:
                words.append((max(0.0, w.start), w.word.strip()))

            # feedback
            if q or bar:
                pct = int(100 * seg.end / duration) if duration else 0
                pct = max(1, min(100, pct))
                eta = 0.0
                if duration and seg.end:
                    elapsed = monotonic() - start_clock
                    eta = elapsed / seg.end * (duration - seg.end)
                if q:
                    q.put(("PROGRESS", pct, eta))
                if bar:
                    bar.n = min(bar.total, seg.end)
                    bar.set_postfix_str(f"ETA {eta:0.1f}s" if eta else "")
                    bar.refresh()
    except Exception as exc:  # ← registra error para depurar
        if q:
            q.put(("ERROR", type(exc).__name__ + ": " + str(exc)))
        else:
            raise

    if bar:
        bar.close()

    if q:
        q.put(("DONE", words))
    if tmp and tmp.exists():
        tmp.unlink(missing_ok=True)
    return words

# ───────────────────── GUI opcional ───────────────────────────
USE_GUI = not (sys.platform.startswith("linux") and not os.environ.get("DISPLAY"))
if USE_GUI:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk
    from utils.gui_errors import show_error

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("Word-timed Whisper")
            self.geometry("520x290")

            self.v_audio = tk.StringVar()
            self.v_test = tk.BooleanVar(value=False)
            self.v_vad  = tk.BooleanVar(value=True)
            self.q: queue.Queue = queue.Queue()

            frm = ttk.Frame(self)
            frm.pack(padx=10, pady=10, fill="x")

            ttk.Label(frm, text="Audio:").grid(row=0, column=0, sticky="e")
            ttk.Entry(frm, textvariable=self.v_audio, width=53).grid(row=0, column=1)
            ttk.Button(frm, text="…", command=self.browse).grid(row=0, column=2)

            ttk.Checkbutton(frm, text="Modo prueba 1 min", variable=self.v_test).grid(row=1, column=1, sticky="w")
            ttk.Checkbutton(frm, text="VAD silencios",   variable=self.v_vad).grid(row=2, column=1, sticky="w")

            ttk.Button(frm, text="Transcribir", command=self.launch).grid(row=3, column=1, pady=6)

            self.pb = ttk.Progressbar(frm, length=420)
            self.pb.grid(row=4, column=0, columnspan=3, pady=4)
            self.eta = ttk.Label(frm, text="")
            self.eta.grid(row=5, column=0, columnspan=3)

            self.log = scrolledtext.ScrolledText(self, height=6, state="disabled")
            self.log.pack(fill="both", expand=True, padx=10, pady=5)

            self.after(250, self._poll)

        # helpers
        def browse(self):
            p = filedialog.askopenfilename(filetypes=[("Audio", "*.*")])
            if p:
                self.v_audio.set(p)

        def log_msg(self, txt: str):
            self.log["state"] = "normal"
            self.log.insert("end", txt + "\n")
            self.log["state"] = "disabled"
            self.log.see("end")

        # thread handling
        def launch(self):
            if not self.v_audio.get():
                messagebox.showwarning("Falta archivo", "Selecciona audio")
                return
            if self.v_test.get() and not have_ffmpeg():
                show_error("FFmpeg requerido", RuntimeError("Instala FFmpeg o desactiva modo prueba"))
                return

            self.pb["value"] = 0
            self.eta["text"] = ""
            self.log_msg("⏳ Transcribiendo…")
            Thread(
                target=self._worker,
                daemon=True,
            ).start()

        def _worker(self):
            try:
                transcribe_audio(
                    Path(self.v_audio.get()),
                    test_mode=self.v_test.get(),
                    use_vad=self.v_vad.get(),
                    q=self.q,
                )
            except Exception as exc:
                self.q.put(("ERROR", str(exc)))

        def _poll(self):
            try:
                while True:
                    msg = self.q.get_nowait()
                    if msg[0] == "PROGRESS":
                        pct, eta = msg[1:]
                        self.pb["value"] = pct
                        self.eta["text"] = f"ETA {eta:0.1f}s" if eta else ""
                    elif msg[0] == "DONE":
                        words = msg[1]
                        out = Path(self.v_audio.get()).with_suffix(".words.csv")
                        write_csv(out, words)
                        self.pb["value"] = 100
                        self.eta["text"] = "Completado"
                        self.log_msg(f"✔ CSV guardado en {out}")
                    elif msg[0] == "ERROR":
                        self.log_msg("❌ " + msg[1])
            except queue.Empty:
                pass
            self.after(250, self._poll)


# ───────────────────── CLI / GUI entrypoint ───────────────────
def cli_main():
    ap = argparse.ArgumentParser(description="Word-timed transcript → CSV")
    ap.add_argument("audio", help="file to transcribe")
    ap.add_argument("--test", action="store_true", help="recorta a 60 s")
    ap.add_argument("--no-vad", action="store_true", help="desactiva filtro VAD")
    ap.add_argument("--script", help="guion para initial_prompt", default=None)
    ap.add_argument("--model", default="large-v3", help="tamano del modelo")
    args = ap.parse_args()

    words = transcribe_audio(
        Path(args.audio),
        test_mode=args.test,
        use_vad=not args.no_vad,
        script_path=args.script,
        model_name=args.model,
    )
    out = Path(args.audio).with_suffix(".words.csv")
    write_csv(out, words)
    print("CSV guardado en", out)


def main():
    if len(sys.argv) > 1 and not USE_GUI:
        cli_main()
    else:
        if USE_GUI:
            App().mainloop()
        else:
            print("En entorno sin GUI pasa el archivo de audio como argumento.")

if __name__ == "__main__":
    main()

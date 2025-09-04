"""Local audio transcription using Whisper."""

from __future__ import annotations

import argparse
import json
import csv
import os
import subprocess
import tempfile
import queue
from time import monotonic
from pathlib import Path

import torch

from text_utils import read_script, extract_word_list
from alignment import build_rows
import alignment

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # pragma: no cover - dependency may not be installed
    raise SystemExit(
        "faster-whisper is required. Install with 'pip install faster-whisper'"
    ) from exc

try:
    from tqdm import tqdm
except Exception as exc:  # pragma: no cover - dependency may not be installed
    raise SystemExit("tqdm is required. Install with 'pip install tqdm'") from exc

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except Exception:  # pragma: no cover - Tk may be missing on headless env
    tk = None  # type: ignore


def _choose_model() -> str:
    sizes = ["tiny", "base", "small", "medium", "large"]
    if tk is None:
        return "base"
    root = tk.Tk()
    root.withdraw()
    model = simpledialog.askstring(
        "Seleccionar modelo",
        "Elige tamaño de modelo de Whisper (tiny, base, small, medium, large):",
        initialvalue="base",
        parent=root,
    )
    root.destroy()
    if model not in sizes:
        if tk is not None:
            messagebox.showwarning(
                title="Modelo inválido",
                message=f"Modelo '{model}' no reconocido. Usando 'large'.",
            )
        return "large"
    return model


def _select_file() -> str | None:
    if tk is None:
        return None
    filetypes = [
        ("Archivos multimedia", "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.mp4"),
        ("Todos los archivos", "*.*"),
    ]
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title="Seleccionar archivo", filetypes=filetypes)
    root.destroy()
    return path


def _extract_audio(path: str) -> tuple[str, str]:
    base, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".mp4":
        print(f"[Transcriber] Extrayendo audio de {path}…")
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            tmp_path,
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"[Transcriber] Audio temporal en {tmp_path}")
        return tmp_path, base
    print(f"[Transcriber] Usando audio original {path}")
    return path, base


def transcribe_wordlevel(
    audio_path: str,
    model_name: str = "large-v3",
    script_path: str | None = None,
    initial_prompt: str | None = None,
    *,
    detailed: bool = False,
) -> Path:
    """Transcribe ``audio_path`` with word timestamps.

    If ``detailed`` is ``True`` save ``.word.json`` with segment metadata,
    otherwise produce a flat ``.words.json`` list.
    """

    print(f"[Transcriber] Transcribiendo {audio_path} con modelo {model_name}")
    hotwords = None
    if script_path:
        try:
            script_text = read_script(script_path)
            words = extract_word_list(script_text)
            if words:
                hotwords = " ".join(words)
                print("[Transcriber] Palabras clave cargadas del guion")
        except Exception:
            hotwords = None
            print("[Transcriber] Advertencia: no se pudieron cargar hotwords")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    print(f"[Transcriber] Dispositivo {device}")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    print("[Transcriber] Ejecutando inferencia…")

    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )

    if detailed:
        out = Path(audio_path).with_suffix(".word.json")
        payload = {"segments": []}
        for seg in segments:
            payload["segments"].append(
                {
                    "seg_start": seg.start,
                    "seg_end": seg.end,
                    "text": seg.text,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in seg.words
                    ],
                }
            )
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf8")
    else:
        out = Path(audio_path).with_suffix(".words.json")
        words: list[dict] = []
        for seg in segments:
            for w in seg.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})
        out.write_text(json.dumps(words, ensure_ascii=False, indent=2), "utf8")
    print(f"[Transcriber] Resultado guardado en {out}")
    return out


def _probe_duration(path: str) -> float:
    """Return audio duration in seconds using ffprobe if available."""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            text=True,
        ).strip()
        dur = float(out)
        print(f"[Transcriber] Duración detectada: {dur:.1f}s")
        return dur
    except Exception:
        return 0.0


def transcribe_file(
    file_path: str | None = None,
    model_size: str | None = None,
    script_path: str | None = None,
    *,
    show_messagebox: bool = True,
    progress_queue: "queue.Queue" | None = None,
) -> Path:
    """Transcribe ``file_path`` with Whisper and save ``.txt`` next to it.

    If ``script_path`` is provided, extract a word list from the script and pass
    it as hotwords to Whisper to improve recognition.
    """

    if not file_path:
        file_path = _select_file()
        if not file_path:
            raise SystemExit("No file selected")

    if not model_size:
        model_size = _choose_model()

    print(f"[Transcriber] Iniciando transcripción de {file_path}")
    print(f"[Transcriber] Modelo: {model_size}")

    audio_path, base = _extract_audio(file_path)

    hotwords = None
    if script_path:
        try:
            script_text = read_script(script_path)
            words = extract_word_list(script_text)
            if words:
                hotwords = " ".join(words)
                print("[Transcriber] Usando guion para hotwords")
        except Exception:
            hotwords = None
            print("[Transcriber] Error obteniendo hotwords del guion")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[Transcriber] Dispositivo {device}")

    model = WhisperModel(
        model_size_or_path=model_size, device=device, compute_type=compute_type
    )

    print("[Transcriber] Ejecutando primera pasada…")

    segments, _info = model.transcribe(audio_path, beam_size=5, hotwords=hotwords)
    duration = _probe_duration(audio_path)
    start = monotonic()
    text = ""
    for segment in tqdm(segments, desc="Transcribiendo", unit="segment"):
        text += segment.text
        if progress_queue:
            pct = int(100 * segment.end / duration) if duration else 0
            pct = max(1, min(100, pct))
            eta = 0.0
            if duration and segment.end:
                elapsed = monotonic() - start
                eta = elapsed / segment.end * (duration - segment.end)
            progress_queue.put(("PROGRESS", pct, eta))

    out_path = Path(base + ".txt")
    out_path.write_text(text, encoding="utf8")
    print(f"[Transcriber] Texto guardado en {out_path}")

    if audio_path != file_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    if progress_queue:
        progress_queue.put(("PROGRESS", 100, 0.0))

    if tk is not None and show_messagebox:
        messagebox.showinfo(
            title="Transcripción finalizada", message=f"Guardado en:\n{out_path}"
        )
    print("[Transcriber] Transcripción completada")
    return out_path


def transcribe_word_csv(
    file_path: str | None = None,
    model_size: str | None = None,
    script_path: str | None = None,
    *,
    show_messagebox: bool = True,
    progress_queue: "queue.Queue" | None = None,
) -> Path:
    """Transcribe ``file_path`` saving words CSV and plain text.

    ``model_size`` selects the Whisper model and ``script_path`` allows
    supplying a reference text whose prominent words are used as hotwords.
    """

    if not file_path:
        file_path = _select_file()
        if not file_path:
            raise SystemExit("No file selected")

    print("[Transcriber] Paso 1/2: transcripción palabra a palabra")
    audio_path, base = _extract_audio(file_path)

    words_json = transcribe_wordlevel(
        audio_path,
        model_name=model_size or "medium",
        script_path=script_path,
        detailed=False,
    )

    data = json.loads(Path(words_json).read_text(encoding="utf8"))

    print("[Transcriber] Paso 2/2: generando CSV y TXT…")

    csv_path = Path(base + ".words.csv")
    with csv_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter=";")
        for item in data:
            w.writerow([f"{item['start']:.2f}", item["word"]])

    text = " ".join(item["word"] for item in data)
    txt_path = Path(base + ".txt")
    txt_path.write_text(text, encoding="utf8")

    if audio_path != file_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    print(f"[Transcriber] Guardado CSV en {csv_path}")
    print(f"[Transcriber] Guardado texto en {txt_path}")

    if progress_queue:
        progress_queue.put(("PROGRESS", 100, 0.0))

    if tk is not None and show_messagebox:
        messagebox.showinfo(
            title="Transcripción finalizada",
            message=f"Guardado en:\n{txt_path}\n{csv_path}",
        )

    print("[Transcriber] Proceso finalizado")
    return txt_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("input", nargs="?", help="Audio or video file")
    parser.add_argument(
        "--model",
        default="medium",
        help="Model size: tiny, base, small, medium, large",
    )
    parser.add_argument(
        "--script",
        help="Optional script text (PDF or TXT) to guide transcription",
    )
    parser.add_argument(
        "--word-align",
        action="store_true",
        help="Output QC JSON using word level alignment",
    )
    parser.add_argument(
        "--word-align-v2",
        action="store_true",
        help="Use word_timed_transcriber_2 and improved resync",
    )
    parser.add_argument(
        "--resync-csv",
        metavar="CSV",
        help="Update an existing QC JSON using a word timed CSV",
    )
    args = parser.parse_args(argv)
    if args.resync_csv:
        if not args.input:
            parser.error("--resync-csv requires a QC JSON path")
        print("[Transcriber] Re-sincronizando JSON con CSV…")
        from utils.resync_python_v2 import load_words_csv, resync_rows
        raw_rows = json.loads(Path(args.input).read_text(encoding="utf8"))
        from qc_utils import canonical_row
        rows = [canonical_row(r) for r in raw_rows]
        csv_words, csv_tcs = load_words_csv(Path(args.resync_csv))
        resync_rows(rows, csv_words, csv_tcs)
        base = Path(args.input).with_suffix("")
        out = base.with_suffix(".resync.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
        print(out)
        return
    if args.word_align:
        if not args.script:
            parser.error("--word-align requires --script")
        print("[Transcriber] Word-align detallado")
        words_json = transcribe_wordlevel(
            args.input,
            args.model,
            args.script,
            detailed=True,
        )
        ref = read_script(args.script)
        words_json_text = Path(words_json).read_text(encoding="utf8")
        rows = alignment.build_rows_wordlevel(ref, words_json_text)
        base = Path(args.input).with_suffix("")
        out = base.with_suffix(".word.qc.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
        print(out)

    elif args.word_align_v2:
        if not args.script:
            parser.error("--word-align-v2 requires --script")
        print("[Transcriber] Word-align v2 – doble transcripción")
        txt = transcribe_word_csv(args.input)
        print("[Transcriber] Primera etapa completada")
        ref = read_script(args.script)
        hyp = Path(txt).read_text(encoding="utf8", errors="ignore")
        from qc_utils import canonical_row
        rows = [canonical_row(r) for r in build_rows(ref, hyp)]

        csv_path = Path(args.input).with_suffix(".words.csv")
        csv_words, csv_tcs = load_words_csv(csv_path)
        resync_rows(rows, csv_words, csv_tcs)
        base = Path(args.input).with_suffix("")
        out = base.with_suffix(".wordv2.qc.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
        print(out)

    else:
        print("[Transcriber] Transcripción simple a TXT y CSV")
        transcribe_word_csv(args.input, args.model, args.script)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

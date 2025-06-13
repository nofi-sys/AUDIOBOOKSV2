"""Local audio transcription using Whisper."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch

from text_utils import read_script, extract_word_list

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
        return tmp_path, base
    return path, base


def transcribe_wordlevel(
    audio_path: str,
    model_name: str = "large-v3",
    script_path: str | None = None,
    initial_prompt: str | None = None,
) -> Path:
    """Transcribe ``audio_path`` with word timestamps and save ``.word.json``."""

    hotwords = None
    if script_path:
        try:
            script_text = read_script(script_path)
            words = extract_word_list(script_text)
            if words:
                hotwords = " ".join(words)
        except Exception:
            hotwords = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )

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
    return out


def transcribe_file(
    file_path: str | None = None,
    model_size: str | None = None,
    script_path: str | None = None,
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

    audio_path, base = _extract_audio(file_path)

    hotwords = None
    if script_path:
        try:
            script_text = read_script(script_path)
            words = extract_word_list(script_text)
            if words:
                hotwords = " ".join(words)
        except Exception:
            hotwords = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(
        model_size_or_path=model_size, device=device, compute_type=compute_type
    )

    segments, _info = model.transcribe(audio_path, beam_size=5, hotwords=hotwords)
    text = ""
    for segment in tqdm(segments, desc="Transcribiendo", unit="segment"):
        text += segment.text

    out_path = Path(base + ".txt")
    out_path.write_text(text, encoding="utf8")

    if audio_path != file_path:
        try:
            os.remove(audio_path)
        except OSError:
            pass

    if tk is not None:
        messagebox.showinfo(
            title="Transcripción finalizada", message=f"Guardado en:\n{out_path}"
        )
    return out_path


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper")
    parser.add_argument("input", nargs="?", help="Audio or video file")
    parser.add_argument(
        "--model",
        default=None,
        help="Model size: tiny, base, small, medium, large",
    )
    parser.add_argument(
        "--script",
        help="Optional script text (PDF or TXT) to guide transcription",
    )
    parser.add_argument(
        "--word-json",
        action="store_true",
        help="Output JSON with word timestamps instead of plain text",
    )
    parser.add_argument(
        "--prompt",
        help="Text file with previous transcription to use as initial prompt",
    )
    args = parser.parse_args(argv)
    if args.word_json:
        prompt_text = (
            Path(args.prompt).read_text(encoding="utf8") if args.prompt else None
        )
        transcribe_wordlevel(args.input or "", args.model or "base", args.script, prompt_text)
    else:
        transcribe_file(args.input, args.model, args.script)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()

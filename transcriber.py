from __future__ import annotations

"""Local audio transcription using Whisper."""
import argparse
import json
import csv
import os
import tempfile
import queue
from time import monotonic
from pathlib import Path

import torch
import av

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
        try:
            with av.open(path) as in_container:
                in_stream = in_container.streams.audio[0]
                with av.open(tmp_path, "w") as out_container:
                    out_stream = out_container.add_stream(
                        "pcm_s16le", rate=16000, layout="mono"
                    )
                    for frame in in_container.decode(in_stream):
                        for packet in out_stream.encode(frame):
                            out_container.mux(packet)
                    # Flush stream
                    for packet in out_stream.encode(None):
                        out_container.mux(packet)
            print(f"[Transcriber] Audio temporal en {tmp_path}")
            return tmp_path, base
        except Exception as e:
            print(f"Error extracting audio with PyAV: {e}")
            raise e

    print(f"[Transcriber] Usando audio original {path}")
    return path, base


def transcribe_wordlevel(
    audio_path: str,
    model_name: str = "large-v3",
    script_path: str | None = None,
    initial_prompt: str | None = None,
    *,
    detailed: bool = False,
    progress_queue: "queue.Queue" | None = None,
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

    print("[Transcriber] Ejecutando inferencia...")

    segments, _info = model.transcribe(
        audio_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        hotwords=hotwords,
        initial_prompt=initial_prompt,
    )
    # Emit progress while iterating
    duration = _probe_duration(audio_path)
    start_t = monotonic()
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
            if progress_queue and duration:
                pct = int(100 * (seg.end or 0.0) / duration)
                pct = max(1, min(100, pct))
                eta = 0.0
                if seg.end:
                    elapsed = monotonic() - start_t
                    eta = max(0.0, elapsed / seg.end * max(0.0, duration - seg.end))
                progress_queue.put(("PROGRESS", pct, eta))
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf8")
    else:
        out = Path(audio_path).with_suffix(".words.json")
        words: list[dict] = []
        for seg in segments:
            for w in seg.words:
                words.append({"word": w.word, "start": w.start, "end": w.end})
            if progress_queue and duration:
                pct = int(100 * (seg.end or 0.0) / duration)
                pct = max(1, min(100, pct))
                eta = 0.0
                if seg.end:
                    elapsed = monotonic() - start_t
                    eta = max(0.0, elapsed / seg.end * max(0.0, duration - seg.end))
                progress_queue.put(("PROGRESS", pct, eta))
        out.write_text(json.dumps(words, ensure_ascii=False, indent=2), "utf8")
    if progress_queue:
        progress_queue.put(("PROGRESS", 100, 0.0))
    print(f"[Transcriber] Resultado guardado en {out}")
    return out


def _extract_chunk(src: str, start_s: float, end_s: float | None) -> str:
    """Extracts a chunk of audio using PyAV."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    start_s = max(0.0, start_s)

    with av.open(src) as in_container:
        in_stream = in_container.streams.audio[0]
        with av.open(tmp_path, "w") as out_container:
            out_stream = out_container.add_stream("pcm_s16le", rate=16000, layout="mono")

            # Seek to the start time. The seek is not always accurate.
            # We seek to a keyframe before the start time and then decode forward.
            seek_target = int(start_s / in_stream.time_base)
            in_container.seek(seek_target, backward=True, any_frame=False, stream=in_stream)

            for frame in in_container.decode(in_stream):
                frame_time = frame.pts * in_stream.time_base
                if frame_time < start_s:
                    continue
                if end_s is not None and frame_time >= end_s:
                    break

                frame.pts = None
                for packet in out_stream.encode(frame):
                    out_container.mux(packet)

            for packet in out_stream.encode(None):
                out_container.mux(packet)
    return tmp_path


def transcribe_wordlevel_ckpt(
    audio_path: str,
    model_name: str = "large-v3",
    script_path: str | None = None,
    initial_prompt: str | None = None,
    *,
    detailed: bool = False,
    progress_queue: "queue.Queue" | None = None,
    resume: bool = True,
    chunk_seconds: int = 600,
    artifacts_base: str | None = None,
) -> Path:
    """Transcribe with checkpoints so long jobs can resume.

    Splits the audio in fixed-size chunks and saves per-chunk JSON files
    under a `.transcribe_ckpt` folder. On resume, skips already completed
    chunks and merges results.
    """

    print(f"[Transcriber] Transcribiendo (ckpt) {audio_path} con {model_name}")
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

    base_audio = Path(artifacts_base) if artifacts_base else Path(audio_path)
    duration = _probe_duration(audio_path)
    if chunk_seconds <= 0:
        chunk_seconds = 600
    n_chunks = 1 if not duration else int((duration + chunk_seconds - 1) // chunk_seconds)

    # Outputs
    final_out = base_audio.with_suffix(".word.json" if detailed else ".words.json")
    ckpt_dir = base_audio.with_suffix("")
    ckpt_dir = ckpt_dir.with_name(ckpt_dir.name + ".transcribe_ckpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = ckpt_dir / "manifest.json"

    def _load_manifest() -> dict:
        if manifest.exists():
            try:
                return json.loads(manifest.read_text(encoding="utf8"))
            except Exception:
                return {}
        return {}

    def _save_manifest(d: dict) -> None:
        tmp = manifest.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf8")
        tmp.replace(manifest)

    man = _load_manifest() if resume else {}
    completed = set(man.get("completed", [])) if isinstance(man.get("completed"), list) else set()
    man.update(
        {
            "audio": str(base_audio),
            "duration": duration,
            "model": model_name,
            "chunk_seconds": int(chunk_seconds),
        }
    )
    _save_manifest({**man, "completed": sorted(completed)})

    # continuity for prompts between chunks
    tail_words: list[str] = []
    start_t = monotonic()

    if final_out.exists() and final_out.stat().st_size > 0 and resume:
        print(f"[Transcriber] Ya existe {final_out}, omitiendo")
        if progress_queue:
            progress_queue.put(("PROGRESS", 100, 0.0))
        return final_out

    for idx in range(n_chunks):
        if resume and str(idx) in completed:
            continue
        start_s = idx * float(chunk_seconds)
        end_s = None if not duration else min(duration, (idx + 1) * float(chunk_seconds))

        if detailed:
            filename = f"chunk_{idx:04d}.word.json"
        else:
            filename = f"chunk_{idx:04d}.json"
        chunk_path = ckpt_dir / filename
        if resume and chunk_path.exists() and chunk_path.stat().st_size > 0:
            completed.add(str(idx))
            _save_manifest({**man, "completed": sorted(completed)})
            continue

        part_audio = _extract_chunk(audio_path, start_s, end_s)
        try:
            use_prompt = initial_prompt
            if tail_words:
                tail = " ".join(tail_words[-50:])
                use_prompt = (use_prompt + "\n" if use_prompt else "") + tail

            segments, _info = model.transcribe(
                part_audio,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                hotwords=hotwords,
                initial_prompt=use_prompt,
            )

            if detailed:
                payload = {"segments": []}
                for seg in segments:
                    if progress_queue and duration:
                        ge = (seg.end or 0.0) + start_s
                        pct = int(100 * ge / duration)
                        pct = max(1, min(100, pct))
                        eta = 0.0
                        if seg.end:
                            elapsed = monotonic() - start_t
                            eta = max(0.0, elapsed / ge * max(0.0, duration - ge))
                        progress_queue.put(("PROGRESS", pct, eta))
                    words_list = []
                    last_words: list[str] = []
                    for w in seg.words:
                        words_list.append(
                            {
                                "word": w.word,
                                "start": (w.start or 0.0) + start_s,
                                "end": (w.end or 0.0) + start_s,
                            }
                        )
                        last_words.append(w.word)
                    payload["segments"].append(
                        {
                            "seg_start": (seg.start or 0.0) + start_s,
                            "seg_end": (seg.end or 0.0) + start_s,
                            "text": seg.text,
                            "words": words_list,
                        }
                    )
                    tail_words.extend(last_words[-10:])
                chunk_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8"
                )
            else:
                words: list[dict] = []
                for seg in segments:
                    if progress_queue and duration:
                        ge = (seg.end or 0.0) + start_s
                        pct = int(100 * ge / duration)
                        pct = max(1, min(100, pct))
                        eta = 0.0
                        if seg.end:
                            elapsed = monotonic() - start_t
                            eta = max(0.0, elapsed / ge * max(0.0, duration - ge))
                        progress_queue.put(("PROGRESS", pct, eta))
                    for w in seg.words:
                        words.append(
                            {
                                "word": w.word,
                                "start": (w.start or 0.0) + start_s,
                                "end": (w.end or 0.0) + start_s,
                            }
                        )
                        tail_words.append(w.word)
                chunk_path.write_text(
                    json.dumps(words, ensure_ascii=False, indent=2), encoding="utf8"
                )

            completed.add(str(idx))
            _save_manifest({**man, "completed": sorted(completed)})

            # rolling partial merge for safety
            try:
                _merge_all_chunks(ckpt_dir, final_out, detailed=detailed, partial_suffix=True)
            except Exception:
                pass
        finally:
            try:
                os.remove(part_audio)
            except Exception:
                pass

    _merge_all_chunks(ckpt_dir, final_out, detailed=detailed, partial_suffix=False)
    if progress_queue:
        progress_queue.put(("PROGRESS", 100, 0.0))
    print(f"[Transcriber] Resultado guardado en {final_out}")
    return final_out


def _probe_duration(path: str) -> float:
    """Return audio duration in seconds using PyAV."""
    try:
        with av.open(path) as container:
            # Duration is in microseconds, convert to seconds
            duration = container.duration / 1_000_000
        print(f"[Transcriber] Duración detectada: {duration:.1f}s")
        return duration
    except Exception as e:
        print(f"Error getting duration with PyAV: {e}")
        return 0.0


def _merge_all_chunks(
    ckpt_dir: Path, final_out: Path, *, detailed: bool, partial_suffix: bool
) -> None:
    """Merge chunk JSONs into a single output.

    If ``partial_suffix`` is True, writes to ``*.part`` alongside the final.
    """
    chunks = sorted(
        [
            p
            for p in ckpt_dir.iterdir()
            if p.is_file()
            and p.name.startswith("chunk_")
            and p.suffix in (".json", ".word.json")
        ]
    )
    if not chunks:
        return
    out_path = (
        final_out if not partial_suffix else final_out.with_suffix(final_out.suffix + ".part")
    )
    if detailed:
        merged = {"segments": []}
        for p in chunks:
            try:
                data = json.loads(p.read_text(encoding="utf8"))
            except Exception:
                continue
            merged["segments"].extend(data.get("segments", []))
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf8")
    else:
        words: list[dict] = []
        for p in chunks:
            try:
                data = json.loads(p.read_text(encoding="utf8"))
            except Exception:
                continue
            if isinstance(data, list):
                words.extend(data)
        out_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf8")


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

    model = WhisperModel(model_size_or_path=model_size, device=device, compute_type=compute_type)

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
    *,
    test_mode: bool = False,
    use_vad: bool = True,
    script_path: str | None = None,
    show_messagebox: bool = True,
    progress_queue: "queue.Queue" | None = None,
    resume: bool = True,
    chunk_seconds: int = 600,
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

    from utils.word_timed_transcriber_2 import transcribe_audio

    if progress_queue:
        try:
            progress_queue.put(("PROGRESS", 1, 0.0))
        except Exception:
            pass
    # Use checkpointed transcription by default for resilience
    words_json = transcribe_wordlevel_ckpt(
        audio_path,
        model_name=model_size or "medium",
        script_path=script_path,
        detailed=False,
        progress_queue=progress_queue,
        resume=resume,
        chunk_seconds=chunk_seconds,
        artifacts_base=base,
    )

    data = json.loads(Path(words_json).read_text(encoding="utf8"))

    print("[Transcriber] Paso 2/2: generando CSV y TXT…")

    csv_path = Path(base + ".words.csv")
    with csv_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter=";")
        for item in data:
            w.writerow([f"{item['start']:.2f}", item["word"]])

    text = " ".join(str(item.get("word", "")) for item in data)
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


def _words_json_to_csv(word_json_path: Path) -> Path:
    """Convert a `.word.json` (segments+words) into `*.words.csv`.

    CSV format: `start; word` with start seconds to two decimals.
    """
    data = json.loads(Path(word_json_path).read_text(encoding="utf8"))
    segments = data.get("segments", [])
    base = Path(str(word_json_path).rsplit(".", 1)[0])
    csv_path = base.with_suffix(".words.csv")
    with csv_path.open("w", newline="", encoding="utf8") as f:
        w = csv.writer(f, delimiter=";")
        for seg in segments:
            for it in seg.get("words", []):
                try:
                    t = float(it.get("start", seg.get("start", 0.0)))
                    word = str(it.get("word", it.get("text", "")))
                except Exception:
                    continue
                w.writerow([f"{t:.2f}", word])
    return csv_path


def guided_transcribe(
    audio_path: str,
    script_path: str,
    *,
    fast_model: str = "base",
    heavy_model: str = "medium",
    chunk_margin: float = 3.0,
    use_ai_post: bool = False,
    progress_queue: "queue.Queue" | None = None,
) -> dict:
    """Guided transcription v1: heavy word-level alignment against script.

    Favor precise timings (heavy pass) and preserve every token. Final
    alignment uses the sophisticated (heavy) words.
    """

    # Heavy pass to get word timings (segments + words)
    base = Path(audio_path).with_suffix("")
    words_json = transcribe_wordlevel_ckpt(
        audio_path,
        model_name=heavy_model,
        script_path=script_path,
        initial_prompt=None,
        detailed=True,
        resume=True,
        artifacts_base=str(base),
        chunk_seconds=180,
    )

    # Convert to CSV for build_rows_from_words
    csv_path = _words_json_to_csv(words_json)

    # Build QC JSON using heavy words for alignment
    ref = read_script(script_path)
    from utils.resync_python_v2 import load_words_csv

    csv_words, csv_tcs = load_words_csv(csv_path)

    # Avoid drops: allow insertion of ASR-only gaps if needed
    os.environ.setdefault("QC_INSERT_SOLO_ASR", "1")
    rows = alignment.build_rows_from_words(ref, csv_words, csv_tcs)
    from qc_utils import canonical_row

    rows = [canonical_row(r) for r in rows]

    base = Path(audio_path).with_suffix("")
    out_qc = base.with_suffix(".guided.qc.json")
    out_txt = base.with_suffix(".guided.txt")

    # Flatten heavy words to text (no loss)
    data = json.loads(Path(words_json).read_text(encoding="utf8"))
    if progress_queue:
        try:
            progress_queue.put("[Transcriber] Construyendo CSV y TXT…")
        except Exception:
            pass
    text_parts = []
    for seg in data.get("segments", []):
        for it in seg.get("words", []):
            text_parts.append(str(it.get("word", it.get("text", ""))))
    out_txt.write_text(" ".join(text_parts), encoding="utf8")

    out_qc.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")

    return {
        "words_json": str(words_json),
        "words_csv": str(csv_path),
        "guided_txt": str(out_txt),
        "guided_qc": str(out_qc),
    }


def super_guided_transcribe(
    audio_path: str,
    script_path: str,
    *,
    fast_model: str = "tiny",
    heavy_model: str = "tiny",
) -> dict:
    """
    Super-guided transcription v2:
    1. Fast pass to get a rough alignment between script and audio.
    2. Slow, prompted pass, feeding the correct text for each audio chunk.
    """
    print("[Transcriber] Super-Guided v2 - Paso 1/3: Transcripción rápida inicial")
    base = Path(audio_path).with_suffix("")

    # 1. Fast pass for rough alignment
    rough_words_json = transcribe_wordlevel_ckpt(
        audio_path,
        model_name=fast_model,
        script_path=script_path,
        detailed=True,
        resume=True,
        artifacts_base=str(base.with_suffix(".rough")),
        chunk_seconds=180,
    )
    print("[Transcriber] Super-Guided v2 - Paso 2/3: Alineación inicial")

    # 2. Get script-to-audio mapping
    ref_text = read_script(script_path)
    rough_json_text = rough_words_json.read_text(encoding="utf8")
    # Use word-level alignment to get rows with text and timestamps
    rows = alignment.build_rows_wordlevel(ref_text, rough_json_text)

    # 3. Prompted pass, chunk by chunk
    print("[Transcriber] Super-Guided v2 - Paso 3/3: Transcripción guiada por prompt")
    final_words = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"
    model = WhisperModel(heavy_model, device=device, compute_type=compute_type)

    total_rows = len(rows)
    for i, row in enumerate(tqdm(rows, desc="Procesando chunks")):
        start_time = float(row[3])  # tc is at index 3
        # Determine end_time for the chunk
        if i + 1 < total_rows:
            end_time = float(rows[i + 1][3])
        else:
            end_time = _probe_duration(audio_path)

        # Skip if chunk is invalid
        if end_time <= start_time:
            continue

        prompt_text = row[4]  # Original text is at index 4

        # Extract audio chunk
        chunk_audio_path = _extract_chunk(audio_path, start_time, end_time)

        try:
            segments, _info = model.transcribe(
                chunk_audio_path,
                word_timestamps=True,
                initial_prompt=prompt_text,
            )
            for seg in segments:
                for w in seg.words:
                    final_words.append(
                        {
                            "word": w.word,
                            "start": (w.start or 0.0) + start_time,
                            "end": (w.end or 0.0) + start_time,
                        }
                    )
        finally:
            os.remove(chunk_audio_path)

    # Save the final high-quality words
    final_words_path = base.with_suffix(".super_guided.words.json")
    final_words_path.write_text(json.dumps(final_words, ensure_ascii=False, indent=2), "utf8")

    # Final alignment
    # Filter out garbage tokens that can break the alignment
    clean_final_words = [w for w in final_words if "<" not in w["word"] and ">" not in w["word"]]

    csv_words = [w["word"] for w in clean_final_words]
    csv_tcs = [w["start"] for w in clean_final_words]
    final_rows = alignment.build_rows_from_words(ref_text, csv_words, csv_tcs)

    out_qc = base.with_suffix(".super_guided.qc.json")
    out_qc.write_text(json.dumps(final_rows, ensure_ascii=False, indent=2), "utf8")

    return {
        "final_words_json": str(final_words_path),
        "final_qc_json": str(out_qc),
    }


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
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Guided multi-stage transcription using heavy word-level alignment",
    )
    parser.add_argument(
        "--super-guided",
        action="store_true",
        help="Super-guided (v2) transcription using prompted chunks.",
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
    if args.guided:
        if not args.script:
            parser.error("--guided requires --script")
        if not args.input:
            parser.error("--guided requires input audio")
        print("[Transcriber] Transcripción guiada (heavy word-level)")
        artifacts = guided_transcribe(args.input, args.script)
        for k, v in artifacts.items():
            print(f"{k}: {v}")
        return
    if args.super_guided:
        if not args.script:
            parser.error("--super-guided requires --script")
        if not args.input:
            parser.error("--super-guided requires input audio")
        print("[Transcriber] Super-guiada (v2) - dos pasadas")
        artifacts = super_guided_transcribe(args.input, args.script)
        for k, v in artifacts.items():
            print(f"{k}: {v}")
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
        data = json.loads(words_json_text)
        words = [
            word.get("word", "")
            for seg in data.get("segments", [])
            for word in seg.get("words", [])
        ]
        tcs = [
            word.get("start", 0.0)
            for seg in data.get("segments", [])
            for word in seg.get("words", [])
        ]
        rows = alignment.build_rows_from_words(ref, words, tcs)
        base = Path(args.input).with_suffix("")
        out = base.with_suffix(".word.qc.json")
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf8")
        print(out)

    elif args.word_align_v2:
        if not args.script:
            parser.error("--word-align-v2 requires --script")
        txt = transcribe_word_csv(args.input, script_path=args.script)
        print("[Transcriber] Word-align v2 – doble transcripción")
        txt = transcribe_word_csv(args.input)
        print("[Transcriber] Primera etapa completada")
        ref = read_script(args.script)
        hyp = Path(txt).read_text(encoding="utf8", errors="ignore")
        from qc_utils import canonical_row

        rows = [canonical_row(r) for r in build_rows(ref, hyp)]

        csv_path = Path(args.input).with_suffix(".words.csv")
        from utils.resync_python_v2 import load_words_csv, resync_rows

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

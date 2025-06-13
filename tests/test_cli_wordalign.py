import json
from pathlib import Path
import sys
import types


def _get_transcriber():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel = object
    sys.modules.setdefault("faster_whisper", fw)
    import importlib
    return importlib.reload(__import__("transcriber"))


def test_cli_word_align(tmp_path, monkeypatch):
    audio = tmp_path / "aud.wav"
    audio.write_text("a")
    script = tmp_path / "book.txt"
    script.write_text("hola")
    words_json = tmp_path / "aud.words.json"

    tr = _get_transcriber()

    def fake_transcribe_wordlevel(fp, model_size=None, script_path=None):
        words_json.write_text('[{"word": "hola"}]', encoding="utf8")
        return words_json

    monkeypatch.setattr(tr, "transcribe_wordlevel", fake_transcribe_wordlevel)
    monkeypatch.setattr(tr, "build_rows_wordlevel", lambda ref, words: [[0, "", 0, 0, "hola", "hola"]])
    monkeypatch.setattr(tr, "read_script", lambda p: "hola")

    tr.main([str(audio), "--script", str(script), "--word-align"])

    out = tmp_path / "aud.word.qc.json"
    data = json.loads(out.read_text())
    assert data[0][4] == "hola"


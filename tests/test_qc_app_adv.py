import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_app_adv import App  # noqa: E402
import ai_review  # noqa: E402


if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


def test_retranscribe_row_uses_custom_prompt(tmp_path, monkeypatch):
    app = App()
    try:
        # Prepare dummy audio path and row
        audio = tmp_path / "a.wav"
        audio.write_text("dummy")
        app.v_audio.set(str(audio))
        iid = app.tree.insert("", "end", values=[0, "", "", "", "", "", "0.0", "hola", "halo"])

        # Mock internal helpers
        clip = tmp_path / "clip.wav"
        clip.write_text("audio")
        monkeypatch.setattr(app, "_extract_clip", lambda *a, **k: str(clip))

        out_file = tmp_path / "out.txt"

        def fake_transcribe(*a, **k):
            out_file.write_text("nuevo")
            return str(out_file)

        monkeypatch.setattr("transcriber.transcribe_file", fake_transcribe)

        review_args = {}
        score_args = {}

        def fake_review(row, prompt=None):
            review_args["prompt"] = prompt
            row[2] = "OK"
            row.insert(3, "ok")
            return "ok"

        def fake_score(row, prompt=None):
            score_args["prompt"] = prompt
            return "5"

        monkeypatch.setattr("ai_review.review_row", fake_review)
        monkeypatch.setattr("ai_review.score_row", fake_score)
        monkeypatch.setattr(app, "save_json", lambda: None)

        app._retranscribe_row(iid)

        assert review_args["prompt"] == ai_review.RETRANS_PROMPT
        assert score_args["prompt"] == ai_review.RETRANS_PROMPT
    finally:
        app.destroy()


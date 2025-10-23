import os
import importlib
import pytest

sys_path = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.insert(0, sys_path)

modules = ["qc_app", "qc_app_adv"]

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


@pytest.mark.parametrize("mod_name", modules)
def test_transcribe_worker_systemexit(monkeypatch, mod_name):
    mod = importlib.import_module(mod_name)
    app = mod.App()
    try:
        def fake_transcribe(*a, **k):
            raise SystemExit("fail")
        monkeypatch.setattr("transcriber.transcribe_word_csv", fake_transcribe)
        called = {}
        def fake_show_error(title, exc):
            called["msg"] = str(exc)
        monkeypatch.setattr(mod, "show_error", fake_show_error)
        app.v_audio.set("x")
        app.v_ref.set("y")
        app._transcribe_worker()
        assert "fail" in called["msg"]
    finally:
        app.destroy()

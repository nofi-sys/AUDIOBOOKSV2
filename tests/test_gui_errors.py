import io
import sys
from utils.gui_errors import show_error
import tkinter.messagebox as messagebox


def test_show_error_logs_and_dialog(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    called = {}
    def fake_showerror(title, msg):
        called["title"] = title
        called["msg"] = msg
    monkeypatch.setattr(messagebox, "showerror", fake_showerror)
    exc = RuntimeError("boom")
    show_error("Oops", exc)
    assert called["title"] == "Oops"
    assert called["msg"] == "boom"
    out = buf.getvalue()
    assert "RuntimeError" in out and "boom" in out

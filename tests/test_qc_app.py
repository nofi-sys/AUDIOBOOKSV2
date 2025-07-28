import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_app import App  # noqa: E402

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


def _insert_rows(app, rows):
    for r in rows:
        app.tree.insert("", "end", values=r)


def test_navigation_bad_rows_qc():
    app = App()
    try:
        rows = [
            [0, "", "", "", "0.0", "uno", "uno"],
            [1, "", "mal", "", "1.0", "dos", "dos"],
            [2, "", "mal", "", "2.0", "tres", "tres"],
        ]
        _insert_rows(app, rows)
        # mark first bad row as OK
        first_bad = app.tree.get_children()[1]
        app.tree.set(first_bad, "OK", "OK")
        first = app.tree.get_children()[0]
        app._play_clip(first)
        app._next_bad_row()
        target = app.tree.get_children()[2]
        assert app._clip_item == target
        app._prev_bad_row()
        assert app._clip_item == target
    finally:
        app.destroy()


def test_play_clip_tc(monkeypatch):
    app = App()
    try:
        rows = [
            [0, "", "", "", "0.0", "uno", "uno"],
            [1, "", "", "", "1.5", "dos", "dos"],
        ]
        _insert_rows(app, rows)
        app.v_audio.set("dummy")
        monkeypatch.setattr("qc_app.play_interval", lambda *a, **k: None)
        first = app.tree.get_children()[0]
        app._play_clip(first)
        assert app._clip_start == 0.0
        assert app._clip_end == 1.5
    finally:
        app.destroy()


def test_worker_creates_json(tmp_path, monkeypatch):
    app = App()
    try:
        ref = tmp_path / "ref.txt"
        ref.write_text("hola")
        hyp = tmp_path / "hyp.txt"
        hyp.write_text("hola")
        app.v_ref.set(str(ref))
        app.v_asr.set(str(hyp))
        monkeypatch.setattr("qc_app.read_script", lambda p: "hola")
        monkeypatch.setattr("qc_app.build_rows", lambda r, h: [[0, "", 0.0, "0.0", "hola", "hola"]])
        app._worker()
        out = hyp.with_suffix(".qc.json")
        data = json.loads(out.read_text())
        assert data[0][-1] == "hola"
    finally:
        app.destroy()


def test_worker_csv_as_asr(tmp_path, monkeypatch):
    app = App()
    try:
        ref = tmp_path / "ref.txt"
        ref.write_text("hola")
        csv = tmp_path / "hyp.csv"
        csv.write_text("0.5; hola\n", encoding="utf8")
        app.v_ref.set(str(ref))
        app.v_asr.set(str(csv))

        monkeypatch.setattr("qc_app.read_script", lambda p: "hola")
        monkeypatch.setattr(
            "qc_app.build_rows", lambda r, h: [[0, "", 0.0, "0.0", "hola", "hola"]]
        )

        def fake_resync_rows(rows, words, tcs):
            rows[0][5] = f"{tcs[0]:.2f}"

        monkeypatch.setattr(
            "utils.resync_python_v2.resync_rows", fake_resync_rows
        )

        app._worker()

        out = csv.with_suffix(".qc.json")
        data = json.loads(out.read_text())
        assert data[0][5] == "0.50"
    finally:
        app.destroy()

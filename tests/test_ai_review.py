import json
import tempfile
from pathlib import Path
from unittest import mock
import pytest

import ai_review
from alignment import build_rows


def test_review_file_basic_skip_and_autofill():
    rows = [
        [0, "✅", "", 10.0, 0.5, "hola", "hola"],
        [1, "", "", 20.0, 0.5, "adios", "adio"],
        [2, "", "OK", 30.0, 0.5, "buenos", "bueno"],
    ]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "file.json"
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")

        with mock.patch("ai_review.ai_verdict", return_value="ok") as m:
            approved, remaining = ai_review.review_file(str(path))

        data = json.loads(path.read_text(encoding="utf8"))
        assert len(data[0]) == 8
        assert len(data[1]) == 8
        assert m.call_count == 2
        assert data[0][3] == "ok" and data[0][2] == "OK"
        assert data[1][3] == "ok" and data[1][2] == "OK"
        assert approved == 2 and remaining == 0


def test_review_file_bad_response_mark_dudoso():
    rows = [[0, "", "", 20.0, 0.5, "hola", "halo"]]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.json"
        p.write_text(json.dumps(rows), encoding="utf8")
        with mock.patch("ai_review.ai_verdict", return_value="blah"):
            ai_review.review_file(str(p))
        out = json.loads(p.read_text())
        assert out[0][3] == "dudoso"


def test_review_row_updates_list():
    row = [0, "", "", 20.0, 0.5, "hola", "hola"]
    with mock.patch("ai_review.ai_verdict", return_value="ok") as m:
        ai_review.review_row(row)
    assert row[2] == "OK"
    assert row[3] == "ok"
    assert m.called


def test_review_file_on_six_column_rows(tmp_path):
    rows = build_rows("hola", "hola")
    for r in rows:
        r[1] = ""  # ensure not skipped
    path = tmp_path / "rows6.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    with mock.patch("ai_review.ai_verdict", return_value="ok"):
        ai_review.review_file(str(path))
    data = json.loads(path.read_text())
    assert len(data[0]) == 8
    assert data[0][2] == "OK" and data[0][3] == "ok"


def test_review_file_on_eight_column_rows(tmp_path):
    rows = [[0, "", "", "", 10.0, 0.5, "hola", "hola"]]
    path = tmp_path / "rows8.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    with mock.patch("ai_review.ai_verdict", return_value="mal"):
        ai_review.review_file(str(path))
    data = json.loads(path.read_text())
    assert len(data[0]) == 8
    assert data[0][3] == "mal"


def test_backup_created_and_partial_save(tmp_path):
    rows = [
        [0, "", "", 0, 0, "hola", "hola"],
        [1, "", "", 0, 0, "adios", "adios"],
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    with mock.patch(
        "ai_review.ai_verdict",
        side_effect=["ok", Exception("boom")],
    ):
        with pytest.raises(Exception):
            ai_review.review_file(str(path))
    data = json.loads(path.read_text())
    assert data[0][2] == "OK" and data[0][3] == "ok"
    bak = path.with_suffix(path.suffix + ".bak")
    assert bak.exists()
    assert json.loads(bak.read_text()) == rows


def test_load_prompt_fallback(tmp_path):
    missing = tmp_path / "no.txt"
    assert ai_review.load_prompt(str(missing)) == ai_review.DEFAULT_PROMPT


def test_stop_review_midway(tmp_path):
    rows = [
        [0, "", 10.0, 0.5, "hola", "hola"],
        [1, "", 20.0, 0.5, "adios", "adios"],
    ]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")

    calls = 0

    def _verdict(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            ai_review.stop_review()
        return "ok"

    with mock.patch("ai_review.ai_verdict", side_effect=_verdict):
        approved, remaining = ai_review.review_file(str(path))

    data = json.loads(path.read_text())
    assert calls == 1
    assert data[0][2] == "OK"
    assert len(data[1]) == 6
    assert approved == 1 and remaining == 0


def test_review_row_feedback_returns_tuple():
    row = [0, "", "", 0.0, 0.0, "hola", "hola"]
    with mock.patch(
        "ai_review.ai_verdict",
        return_value=("ok", "ok porque coincide"),
    ):
        verdict, info = ai_review.review_row_feedback(row)
    assert verdict == "ok" and info == "ok porque coincide"
    assert row[2] == "OK" and row[3] == "ok"


def test_review_file_feedback_collects_messages(tmp_path):
    rows = [[0, "", "", 0.0, 0.0, "hola", "hola"]]
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf8")
    with mock.patch(
        "ai_review.ai_verdict",
        return_value=("mal", "mal por error"),
    ):
        approved, remaining, logs = ai_review.review_file_feedback(str(path))
    assert approved == 0 and remaining == 1
    assert logs == ["mal por error"]
    data = json.loads(path.read_text())
    assert data[0][3] == "mal"

import json
import tempfile
from pathlib import Path
from unittest import mock

import ai_review


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
        assert len(data[0]) == 7  # skipped rows are unchanged
        assert m.call_count == 1
        assert data[1][3] == "ok" and data[1][2] == "OK"
        assert approved == 1 and remaining == 0


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

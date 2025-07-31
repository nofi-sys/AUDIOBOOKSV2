import json

from alignment import build_rows, build_rows_wordlevel
from text_utils import normalize


def test_build_rows_basic():
    ref = "hola mundo"
    hyp = "hola mundo"
    rows = build_rows(ref, hyp)
    assert rows[0][1] == "✅"


def test_build_rows_sentence_split():
    ref = "Hola mundo. Adios."
    hyp = "Hola mundo. Adios."
    rows = build_rows(ref, hyp)
    assert len(rows) == 2
    assert all(r[1] == "✅" for r in rows)


def test_build_rows_wordlevel_basic():
    ref = "Hola mundo"
    data = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"word": "Hola", "start": 0.0, "end": 0.5},
                    {"word": "mundo", "start": 0.5, "end": 1.0},
                ],
            }
        ]
    }
    rows = build_rows_wordlevel(ref, json.dumps(data))
    assert rows[0][1] == "✅"
    assert rows[0][3] == 0.0


def test_build_rows_tc_sequential():
    ref = "Uno dos. Tres cuatro cinco."
    hyp = "Uno dos tres cuatro cinco"
    rows = build_rows(ref, hyp)
    tcs = [r[3] for r in rows]
    assert tcs == sorted(tcs)
    assert tcs[0] == 0.0


def test_build_rows_detect_repetition():
    ref = "Hola mundo"
    hyp = "Hola mundo hola mundo hola mundo"
    rows = build_rows(ref, hyp)
    assert rows[0][5] == "hola mundo hola mundo hola mundo"
    assert len(rows[0]) > 6
    assert len(rows[0][6]) > 1


def test_build_rows_truncated_take():
    ref = "Nos los representantes del pueblo argentino"
    hyp = "Nos los representantes nos nos los representantes del pueblo argentino"
    rows = build_rows(ref, hyp)
    assert rows[0][5].endswith("argentino")
    assert len(rows[0]) > 6
    assert rows[0][6][-1].endswith("argentino")


def test_build_rows_no_truncation():
    ref = "Hola mundo. Adios."
    hyp = "Hola hola mundo. Adios."
    rows = build_rows(ref, hyp)
    assembled = " ".join(r[5] for r in rows)
    assert normalize(assembled, strip_punct=False).split() == normalize(hyp, strip_punct=False).split()

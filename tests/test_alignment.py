import json

from alignment import build_rows, build_rows_wordlevel


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
    assert rows[0][3] == 1.0

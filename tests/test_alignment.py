from alignment import build_rows


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

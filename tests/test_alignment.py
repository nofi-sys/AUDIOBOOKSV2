import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alignment import build_rows


def test_build_rows_basic():
    ref = "hola mundo"
    hyp = "hola mundo"
    rows = build_rows(ref, hyp)
    assert rows[0][1] == "✅"

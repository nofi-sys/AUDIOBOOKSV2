import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import text_utils


def test_token_equal_digit_word():
    assert text_utils.token_equal("1", "uno")
    assert text_utils.token_equal("dos", "2")
    assert not text_utils.token_equal("1", "dos")


def test_find_anchor_trigrams():
    ref = "a b c d e f".split()
    hyp = "x a b c y d e f".split()
    anchors = text_utils.find_anchor_trigrams(ref, hyp)
    assert anchors == [(3, 5)]

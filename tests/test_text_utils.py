import text_utils


def test_token_equal_digit_word():
    assert text_utils.token_equal("1", "uno")
    assert text_utils.token_equal("dos", "2")
    assert not text_utils.token_equal("1", "dos")


def test_token_equal_punctuation():
    assert text_utils.token_equal(".", "punto")
    assert text_utils.token_equal(",", "coma")
    assert text_utils.token_equal(";", "punto y coma")


def test_normalize_punctuation_word():
    out = text_utils.normalize("hola punto adios", strip_punct=False)
    assert out == "hola . adios"
    out = text_utils.normalize("bien punto y coma mal", strip_punct=False)
    assert out == "bien ; mal"


def test_find_anchor_trigrams():
    ref = "a b c d e f".split()
    hyp = "x a b c y d e f".split()
    anchors = text_utils.find_anchor_trigrams(ref, hyp)
    assert anchors == [(3, 5)]


def test_token_equal_accents_case():
    assert text_utils.token_equal("Árbol", "arbol")

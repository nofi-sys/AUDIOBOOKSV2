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

def test_token_equal_abbreviation():
    assert text_utils.token_equal("r.", "r")
    assert text_utils.token_equal("j.", "j")


def test_find_repeated_sequences_bigram():
    text = "this is a test with a bigram a bigram repeated"
    assert text_utils.find_repeated_sequences(text) == ["a bigram a bigram"]

def test_find_repeated_sequences_trigram():
    text = "another test with a test trigram a test trigram repeated"
    assert text_utils.find_repeated_sequences(text) == ["a test trigram a test trigram"]

def test_find_repeated_sequences_none():
    text = "this text has no repetitions"
    assert text_utils.find_repeated_sequences(text) == []

def test_find_repeated_sequences_case_insensitive():
    text = "this is a Test Bigram a test bigram with mixed case"
    assert text_utils.find_repeated_sequences(text) == ["a test bigram a test bigram"]

def test_find_repeated_sequences_consecutive_only():
    text = "one two three one two four"
    assert text_utils.find_repeated_sequences(text) == []

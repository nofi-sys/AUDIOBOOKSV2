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


def test_find_repeated_sequences_exact():
    text = "this is a test with a very common phrase that is a very common phrase that is repeated"
    assert text_utils.find_repeated_sequences(text) == ["a very common phrase that is ... a very common phrase that is"]

def test_find_repeated_sequences_inexact():
    text = "this is a test with a very common phraze that is a very common phrase that is repeated"
    assert text_utils.find_repeated_sequences(text) == ["a very common phraze that is ... a very common phrase that is"]

def test_find_repeated_sequences_long():
    text = "a very long phrase that is repeated a very long phrase that is repeated"
    assert text_utils.find_repeated_sequences(text) == ["very long phrase that is ... a very long phrase that"]

def test_find_repeated_sequences_none():
    text = "this text has no repetitions at all"
    assert text_utils.find_repeated_sequences(text) == []

def test_find_repeated_sequences_case_insensitive():
    text = "this is a very good Test Phrase a very good test phrase with mixed case"
    assert text_utils.find_repeated_sequences(text) == ["a very good Test Phrase ... a very good test phrase"]

def test_split_text_into_paragraphs_multi_newline():
    text = "Paragraph one.\nThis is a continuation of paragraph one.\n\nParagraph two starts here.\nAnother line in paragraph two."
    paragraphs = text_utils.split_text_into_paragraphs(text)
    expected_paragraphs = [
        "Paragraph one.",
        "This is a continuation of paragraph one.",
        "Paragraph two starts here.",
        "Another line in paragraph two."
    ]
    assert paragraphs == expected_paragraphs

def test_split_text_into_paragraphs_single_long_line():
    text = "This is a single very long line with no newlines."
    paragraphs = text_utils.split_text_into_paragraphs(text)
    expected_paragraphs = ["This is a single very long line with no newlines."]
    assert paragraphs == expected_paragraphs

def test_split_text_into_paragraphs_empty_lines():
    text = "\n\nParagraph one.\n\n\nParagraph two.\n"
    paragraphs = text_utils.split_text_into_paragraphs(text)
    expected_paragraphs = ["Paragraph one.", "Paragraph two."]
    assert paragraphs == expected_paragraphs


def test_split_text_detects_heading_and_soft_wrap():
    text = (
        "CAPITULO I\n"
        "Este es un parrafo\n"
        "que continua sin punto\n"
        "Termina aqui.\n"
        "Subtitulo breve\n"
        "Otro parrafo completo."
    )
    paragraphs = text_utils.split_text_into_paragraphs(text)
    assert paragraphs[0] == "CAPITULO I"
    assert paragraphs[1] == "Este es un parrafo que continua sin punto"
    assert paragraphs[2] == "Termina aqui."
    assert paragraphs[3] == "Subtitulo breve"
    assert paragraphs[4] == "Otro parrafo completo."

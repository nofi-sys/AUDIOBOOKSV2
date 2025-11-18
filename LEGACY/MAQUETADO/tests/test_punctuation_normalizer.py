import unittest

from txt2md_mvp.punctuation_normalizer import (
    normalize_punctuation,
    NormalizerSettings,
    LANGUAGE_ES,
    LANGUAGE_EN_US,
    GENRE_NARRATIVE,
)


def _settings(language: str) -> NormalizerSettings:
    return {"language": language, "genre": GENRE_NARRATIVE}


class PunctuationNormalizerTests(unittest.TestCase):
    def test_spanish_dialogue_dash_and_inverted_marks(self) -> None:
        text = "- Hola.\n- Como estas?"
        result = normalize_punctuation(text, _settings(LANGUAGE_ES))
        expected = "\u2014\u00A0Hola.\n\u2014\u00A0\u00BFComo estas?"
        self.assertEqual(result.normalized_text, expected)
        self.assertGreaterEqual(result.stats["dialogue_blocks_fixed"], 1)
        self.assertGreaterEqual(result.stats["inverted_marks_added_or_removed"], 1)

    def test_spanish_range_normalization(self) -> None:
        text = "Fechas 1999-2005 y 10-12."
        result = normalize_punctuation(text, _settings(LANGUAGE_ES))
        self.assertIn("1999\u20132005", result.normalized_text)
        self.assertGreaterEqual(result.stats["ranges_normalized"], 1)

    def test_english_quotes_remain_double(self) -> None:
        text = '"Hello," he said.'
        result = normalize_punctuation(text, _settings(LANGUAGE_EN_US))
        self.assertTrue(result.normalized_text.startswith("\u201CHello,\u201D"))


if __name__ == "__main__":
    unittest.main()

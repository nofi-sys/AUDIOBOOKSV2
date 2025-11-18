import unittest

from txt2md_mvp.translation_engine import TranslationEngine


class TestTranslationEngineRetryHeuristics(unittest.TestCase):
    def setUp(self):
        # No API key required for heuristic tests
        self.engine = TranslationEngine(api_key=None)

    def test_needs_retry_detects_untranslated_paragraph(self):
        original = "This is a longer sample paragraph that clearly remains in English after processing."
        untranslated = "This is a longer sample paragraph that clearly remains in English after processing."
        self.assertTrue(self.engine._needs_retry(original, untranslated))

    def test_needs_retry_allows_short_titles(self):
        original = "The End"
        untranslated = "The End"
        self.assertFalse(self.engine._needs_retry(original, untranslated))

    def test_needs_retry_accepts_translated_content(self):
        original = "The Mississippi river cuts through the valley during the spring flood."
        translated = "El río Mississippi atraviesa el valle durante la crecida de primavera."
        self.assertFalse(self.engine._needs_retry(original, translated))


if __name__ == "__main__":
    unittest.main()

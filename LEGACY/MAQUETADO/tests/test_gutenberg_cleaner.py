import unittest
from txt2md_mvp.gutenberg_cleaner import clean_text

class TestGutenbergCleaner(unittest.TestCase):

    def test_with_gutenberg_header_and_footer(self):
        raw_text = """
Some introductory text.
*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
This is the actual content of the book.
Line 1.
Line 2.
*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
Some closing text.
"""
        expected_text = "This is the actual content of the book.\nLine 1.\nLine 2."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_without_gutenberg_markers(self):
        raw_text = "This is a normal text without any Gutenberg markers."
        self.assertEqual(clean_text(raw_text), raw_text)

    def test_with_only_start_marker(self):
        raw_text = """
Some introductory text.
*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
This is the actual content of the book.
Line 1.
Line 2.
"""
        expected_text = "This is the actual content of the book.\nLine 1.\nLine 2."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_with_only_end_marker(self):
        raw_text = """
This is the actual content of the book.
Line 1.
Line 2.
*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
Some closing text.
"""
        # Without a start marker, the function should return the original text
        self.assertEqual(clean_text(raw_text), raw_text)

    def test_empty_string(self):
        raw_text = ""
        self.assertEqual(clean_text(raw_text), "")

    def test_text_between_markers_is_empty(self):
        raw_text = """
*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***
"""
        self.assertEqual(clean_text(raw_text).strip(), "")

    def test_case_insensitivity(self):
        raw_text = """
*** start of the project gutenberg ebook the adventures of sherlock holmes ***
This is the book content.
*** end of the project gutenberg ebook the adventures of sherlock holmes ***
"""
        expected_text = "This is the book content."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_start_marker_with_this_keyword(self):
        raw_text = """
Preface material.
*** START OF THIS PROJECT GUTENBERG EBOOK THE MISSISSIPPI SAUCER ***
Bloque traducible.
*** END OF THIS PROJECT GUTENBERG EBOOK THE MISSISSIPPI SAUCER ***
Appendix.
"""
        expected_text = "Bloque traducible."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_variant_marker_without_spaces(self):
        raw_text = """
Preface text
***START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES***
Main body line 1.
Main body line 2.
***END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES***
Appendix
"""
        expected_text = "Main body line 1.\nMain body line 2."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_start_marker_without_space_after_asterisks(self):
        raw_text = """
Random intro
***START OF THE PROJECT GUTENBERG EBOOK SAMPLE TITLE***
Contenido real.
***END OF THE PROJECT GUTENBERG EBOOK SAMPLE TITLE***
"""
        expected_text = "Contenido real."
        self.assertEqual(clean_text(raw_text).strip(), expected_text)

    def test_fallback_strip_boilerplate_without_markers(self):
        raw_text = """Project Gutenberg License: this should be removed.
Visit www.gutenberg.org for details.

Main Story Title
Chapter 1
Real content starts here."""
        cleaned = clean_text(raw_text)
        self.assertNotIn("Gutenberg", cleaned)
        self.assertTrue(cleaned.startswith("Main Story Title"))
        self.assertIn("Real content starts here.", cleaned)

if __name__ == '__main__':
    unittest.main()

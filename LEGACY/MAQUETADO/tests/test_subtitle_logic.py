import unittest
from txt2md_mvp.pipeline import process_text

class TestSubtitleLogic(unittest.TestCase):

    def test_chapter_with_paragraph(self):
        text = "CHAPTER I.\n\nIt had occurred to her that she was in some way responsible for the situation."
        result = process_text(text, clean_gutenberg=False)
        document = result["document"]
        self.assertEqual(len(document), 2)
        self.assertEqual(document[0]["type"], "h2")
        self.assertEqual(document[0]["text"], "CHAPTER I.")
        self.assertEqual(document[1]["type"], "p")

    def test_chapter_with_subtitle_next_line(self):
        text = "CHAPTER I.\n\nA NEW BEGINNING\n\nIt had occurred to her that she was in some way responsible for the situation."
        result = process_text(text, clean_gutenberg=False)
        document = result["document"]
        self.assertEqual(len(document), 3)
        self.assertEqual(document[0]["type"], "h2")
        self.assertEqual(document[0]["text"], "CHAPTER I.")
        self.assertEqual(document[1]["type"], "subtitle")
        self.assertEqual(document[1]["text"], "A NEW BEGINNING")
        self.assertEqual(document[2]["type"], "p")

    def test_chapter_with_subtitle_on_same_line(self):
        text = "CHAPTER II. A START"
        result = process_text(text, clean_gutenberg=False)
        document = result["document"]
        self.assertEqual(len(document), 2)
        self.assertEqual(document[0]["type"], "h2")
        self.assertEqual(document[0]["text"], "CHAPTER II.")
        self.assertEqual(document[1]["type"], "subtitle")
        self.assertEqual(document[1]["text"], "A START")

    def test_container_with_merge(self):
        text = "PART I\n\nA Cool Title\n\nThis is the text."
        result = process_text(text, clean_gutenberg=False)
        document = result["document"]
        self.assertEqual(len(document), 3)
        self.assertEqual(document[0]["type"], "h2")
        self.assertEqual(document[0]["text"], "PART I")
        self.assertEqual(document[1]["type"], "subtitle")
        self.assertEqual(document[1]["text"], "A Cool Title")
        self.assertEqual(document[2]["type"], "p")

    def test_container_no_merge(self):
        text = "BOOK II\n\nThis is a long sentence that should not be merged."
        result = process_text(text, clean_gutenberg=False)
        document = result["document"]
        self.assertEqual(len(document), 2)
        self.assertEqual(document[0]["type"], "h2")
        self.assertEqual(document[0]["text"], "BOOK II")
        self.assertEqual(document[1]["type"], "p")

if __name__ == '__main__':
    unittest.main()
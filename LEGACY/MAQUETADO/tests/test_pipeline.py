from pathlib import Path
import copy
import tempfile
import unittest
import docx

from txt2md_mvp.pipeline import process_file, process_text
from txt2md_mvp.rules import match_patterns
from txt2md_mvp.md2docx.core import DEFAULT_STYLESET
from txt2md_mvp.punctuation_normalizer import LANGUAGE_ES, GENRE_NARRATIVE


def load_demo_text() -> str:
    return Path("demo/ejemplo.txt").read_text(encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def test_templates_drive_headings(self) -> None:
        result = process_text(load_demo_text(), title_hint="ejemplo")
        headings = [entry for entry in result["analysis"] if entry["decision"].startswith("h")]
        self.assertTrue(headings, "Expected headings parsed from demo text")
        self.assertEqual(headings[0]["info"].get("template"), "capitulo_parte_romanos")

    def test_front_matter_title_and_slug(self) -> None:
        result = process_text(load_demo_text(), title_hint="ejemplo")
        md_output = result["md"]
        self.assertIn("title: \"ejemplo\"", md_output)
        self.assertNotIn("## Contenidos", md_output)

    def test_markdown_toc_respects_style_flag(self) -> None:
        style_cfgs = copy.deepcopy(DEFAULT_STYLESET)
        style_cfgs.setdefault("_global", {})["generate_toc"] = True
        result = process_text(load_demo_text(), title_hint="ejemplo", style_cfgs=style_cfgs)
        md_output = result["md"]
        self.assertIn("## Contenidos", md_output)
        self.assertNotIn("- [---]", md_output)

    def test_analysis_reports_template_signals_for_heuristics(self) -> None:
        match = match_patterns("ANEXO ESPECIAL")
        self.assertIsNotNone(match)
        level, title, info, _ = match
        self.assertEqual(level, "h2")
        self.assertEqual(title, "ANEXO ESPECIAL")
        self.assertIn("signals", info)
        self.assertTrue(info["signals"].get("allow_all_caps_titles"))

    def test_front_matter_blocks_and_meta(self) -> None:
        sample = (
            "THE SKY ADVENTURE\n\n"
            "BY\n"
            "EDWARD SMITH\n"
            "LEE GARBY\n\n"
            "Chapter 1\n"
            "This is the opening paragraph.\n"
        )
        result = process_text(sample, clean_gutenberg=False)
        document = result["document"]
        self.assertGreaterEqual(len(document), 3)
        self.assertEqual(document[0]["type"], "h1")
        self.assertEqual(document[0]["text"], "The Sky Adventure")
        self.assertEqual(document[1]["type"], "subtitle")
        self.assertIn(document[1]["text"], ("Edward Smith", "Lee Garby"))
        self.assertEqual(document[2]["type"], "subtitle")
        meta = result["meta"]
        self.assertEqual(meta.get("title"), "The Sky Adventure")
        self.assertIn("Edward Smith", meta.get("author", ""))
        front_info = meta.get("front_matter", {})
        self.assertEqual(front_info.get("title"), "The Sky Adventure")
        self.assertIn("Edward Smith", front_info.get("author", ""))

    def test_poem_and_letter_detection(self) -> None:
        text = (
            "Silent night\n"
            "Gentle light\n"
            "Dreams alight\n\n"
            "Dear Friend,\n"
            "I hope this letter finds you well.\n"
            "Sincerely,\n"
            "John\n"
        )
        result = process_text(text, clean_gutenberg=False)
        doc_types = [blk["type"] for blk in result["document"]]
        self.assertIn("poem", doc_types)
        self.assertIn("letter", doc_types)
        poem_index = doc_types.index("poem")
        letter_index = doc_types.index("letter")
        self.assertLess(poem_index, letter_index)
        poem_text = result["document"][poem_index]["text"]
        self.assertIn("Silent night", poem_text)
        letter_text = result["document"][letter_index]["text"]
        self.assertTrue(letter_text.startswith("Dear Friend,"))
        self.assertIn("Sincerely", letter_text)

    def test_punctuation_module_integration(self) -> None:
        text = "- Hola.\n- Como estas?"
        result = process_text(
            text,
            clean_gutenberg=False,
            use_punctuation_module=True,
            punctuation_settings={"language": LANGUAGE_ES, "genre": GENRE_NARRATIVE},
        )
        punctuation = result.get("punctuation")
        self.assertIsNotNone(punctuation)
        self.assertGreater(punctuation.get("change_count", 0), 0)
        first_block = result["document"][0]
        self.assertTrue(first_block["text"].startswith("\u2014\u00A0Hola"))

    def test_report_contains_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, report_path, _ = process_file("demo/ejemplo.txt", tmpdir)
            content = Path(report_path).read_text(encoding="utf-8")
        self.assertIn("Plantillas aplicadas", content)
        self.assertIn("capitulo_parte_romanos", content)

    def test_detects_indented_blockquote(self):
        text = (
            "Este es un párrafo normal.\n\n"
            "  Y esta es una cita en bloque.\n"
            "  Debería ser detectada correctamente.\n\n"
            "Otro párrafo normal."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "test.txt"
            in_path.write_text(text, encoding="utf-8")

            md_path, _, _, docx_path = process_file(str(in_path), tmpdir)

            # Check markdown output
            md_content = Path(md_path).read_text(encoding="utf-8")
            self.assertIn("> Y esta es una cita en bloque.\n> Debería ser detectada correctamente.", md_content)
            self.assertNotIn("  Y esta es una cita en bloque.", md_content)

            # Check docx output
            self.assertTrue(Path(docx_path).exists())
            doc = docx.Document(docx_path)
            p_texts = [p.text for p in doc.paragraphs if p.text.strip()]

            self.assertEqual(len(p_texts), 3)
            self.assertEqual(p_texts[0], "Este es un párrafo normal.")
            self.assertEqual(p_texts[1], "Y esta es una cita en bloque.\nDebería ser detectada correctamente.")
            self.assertEqual(p_texts[2], "Otro párrafo normal.")


if __name__ == "__main__":
    unittest.main()

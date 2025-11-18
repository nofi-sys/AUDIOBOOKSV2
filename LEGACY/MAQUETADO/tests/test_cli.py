from pathlib import Path
import unittest

from txt2md_mvp.inputs import gather_inputs


class InputDiscoveryTests(unittest.TestCase):
    def test_gather_inputs_finds_demo_txt(self) -> None:
        targets = list(gather_inputs(["demo"], None, False))
        names = {path.name for path in targets}
        self.assertIn("ejemplo.txt", names)

    def test_gather_inputs_glob_pattern(self) -> None:
        tmp_dir = Path("demo")
        pattern_targets = list(gather_inputs([str(tmp_dir)], "*.txt", False))
        self.assertTrue(pattern_targets)
        for candidate in pattern_targets:
            self.assertEqual(candidate.suffix.lower(), ".txt")


if __name__ == "__main__":
    unittest.main()

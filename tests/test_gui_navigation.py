import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from gui import App


def test_navigation_bad_rows():
    app = App()
    try:
        rows = [
            [0, "", "", "", "1.0", "uno", "uno"],
            [1, "", "mal", "", "1.0", "dos", "dos"],
            [2, "", "", "", "1.0", "tres", "tres"],
        ]
        for r in rows:
            app.tree.insert("", "end", values=r)
        first = app.tree.get_children()[0]
        app._play_clip(first)
        app._next_bad_row()
        bad = app.tree.get_children()[1]
        assert app._clip_item == bad
        app._prev_bad_row()
        assert app._clip_item == bad
    finally:
        app.destroy()

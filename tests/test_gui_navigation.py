import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_app import App  # noqa: E402

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


def test_navigation_bad_rows():
    app = App()
    try:
        rows = [
            [0, "", "", "", "1.0", "uno", "uno"],
            [1, "", "mal", "", "1.0", "dos", "dos"],
            [2, "", "mal", "", "1.0", "tres", "tres"],
        ]
        for r in rows:
            app.tree.insert("", "end", values=r)
        # mark first bad row as OK so navigation skips it
        first_bad = app.tree.get_children()[1]
        app.tree.set(first_bad, "OK", "OK")
        first = app.tree.get_children()[0]
        app._play_clip(first)
        app._next_bad_row()
        target = app.tree.get_children()[2]
        assert app._clip_item == target
        app._prev_bad_row()
        assert app._clip_item == target
    finally:
        app.destroy()

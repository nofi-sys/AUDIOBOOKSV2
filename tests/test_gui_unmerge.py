import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from gui import App


def test_unmerge_row():
    app = App()
    try:
        row1 = [0, "✅", "", "", "1.0", "hola", "hola"]
        row2 = [1, "✅", "", "", "1.0", "mundo", "mundo"]
        app.tree.insert("", "end", values=row1)
        app.tree.insert("", "end", values=row2)
        for iid in app.tree.get_children():
            app.tree.selection_add(iid)
        app._merge_selected_rows()
        merged = app.tree.get_children()[0]
        app.tree.selection_set(merged)
        app._unmerge_row()
        children = app.tree.get_children()
        assert len(children) == 2
        vals = [app.tree.set(c, "Original") for c in children]
        assert vals == ["hola", "mundo"]
    finally:
        app.destroy()

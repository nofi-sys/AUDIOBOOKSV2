import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)

from gui import App


def test_merge_selected_rows():
    app = App()
    try:
        row1 = [0, "✅", "", "", "1.0", "hola", "hola"]
        row2 = [1, "✅", "", "", "2.0", "mundo", "mundo"]
        app.tree.insert("", "end", values=row1)
        app.tree.insert("", "end", values=row2)
        for iid in app.tree.get_children():
            app.tree.selection_add(iid)
        app._merge_selected_rows()
        assert len(app.tree.get_children()) == 1
        item = app.tree.get_children()[0]
        assert app.tree.set(item, "Original") == "hola mundo"
        assert app.tree.set(item, "dur") == "3.00"
    finally:
        app.destroy()

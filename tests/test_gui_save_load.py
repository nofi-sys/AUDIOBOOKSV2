import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_app import App  # noqa: E402

if not os.environ.get("DISPLAY"):
    pytest.skip("no display", allow_module_level=True)


def test_save_load_ok_status():
    app = App()
    try:
        sample = [0, "✅", "", 0.0, 0.0, "hola", "hola"]
        app.tree.insert("", "end", values=sample)
        app.update_idletasks()
        item = app.tree.get_children()[0]
        # toggle OK manually to avoid GUI event complexity
        app.tree.set(item, "OK", "OK")
        app.ok_rows.add(int(app.tree.set(item, "ID")))
        assert app.tree.set(item, "OK") == "OK"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tmp.json")
            app.v_json.set(path)
            app.save_json()
            app.clear_table()
            app.load_json()
            loaded = app.tree.get_children()[0]
            assert app.tree.set(loaded, "OK") == "OK"
    finally:
        app.destroy()

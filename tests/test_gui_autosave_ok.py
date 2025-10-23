import os
import sys
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from qc_app import App  # noqa: E402

if not os.environ.get("DISPLAY"):
    import pytest
    pytest.skip("no display", allow_module_level=True)


def test_autosave_on_toggle_ok():
    app = App()
    try:
        row = [0, "", "", "", 0.0, "0.0", "hola", "hola"]
        app.tree.insert("", "end", values=row)
        item = app.tree.get_children()[0]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tmp.json")
            app.v_json.set(path)
            app._toggle_ok(item)
            data = json.loads(open(path, encoding="utf8").read())
            assert data[0][2] == "OK"
    finally:
        app.destroy()


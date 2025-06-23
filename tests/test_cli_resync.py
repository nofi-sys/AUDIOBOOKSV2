import json
from pathlib import Path
import sys
import types

from test_cli_wordalign import _get_transcriber


def test_cli_resync(tmp_path):
    qc = tmp_path / "in.qc.json"
    qc.write_text(json.dumps([[0, "", 0, 0.0, "hola", "hola"]]), encoding="utf8")
    csv = tmp_path / "in.words.csv"
    csv.write_text("0.5; hola\n", encoding="utf8")

    tr = _get_transcriber()
    tr.main([str(qc), "--resync-csv", str(csv)])

    out = tmp_path / "in.resync.json"
    data = json.loads(out.read_text())
    assert data[0][5] == "0.50"

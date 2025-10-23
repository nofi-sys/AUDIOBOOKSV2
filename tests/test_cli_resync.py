import json

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
    assert data[0][7] == "hola"


def test_cli_resync_trailing_interpolate(tmp_path):
    qc = tmp_path / "in.qc.json"
    rows = [
        [0, "", 0, 0.0, "hola", "hola"],
        [1, "", 0, 0.0, "adios", "adios"],
        [2, "", 0, 0.0, "gracias", "gracias"],
    ]
    qc.write_text(json.dumps(rows), encoding="utf8")
    csv = tmp_path / "in.words.csv"
    csv.write_text("0.0; hola\n3.0; extra\n", encoding="utf8")

    tr = _get_transcriber()
    tr.main([str(qc), "--resync-csv", str(csv)])

    out = tmp_path / "in.resync.json"
    data = json.loads(out.read_text())
    assert data[1][5] == "1.50"
    assert data[2][5] == "3.00"


def test_cli_resync_with_takes(tmp_path):
    qc = tmp_path / "in.qc.json"
    row = [0, "", 0, 0.0, "Hola", "Hola", ["t1"]]
    qc.write_text(json.dumps([row]), encoding="utf8")
    csv = tmp_path / "in.words.csv"
    csv.write_text("0.5; hola\n", encoding="utf8")

    tr = _get_transcriber()
    tr.main([str(qc), "--resync-csv", str(csv)])

    out = tmp_path / "in.resync.json"
    data = json.loads(out.read_text())
    assert data[0][5] == "0.50"
    assert data[0][7] == "Hola"
    assert data[0][8] == ["t1"]

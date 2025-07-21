import qc_utils


def test_merge_qc_metadata_transfers_fields():
    old = [[0, "✅", "OK", "mal", 10.0, 0.5, "hola amigos", "hola amigos"]]
    new = [[0, "❌", 20.0, 1.0, "hola amigo", "hola amigo"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) == 8
    assert merged[0][1] == "✅"
    assert merged[0][2] == "OK"
    assert merged[0][3] == "mal"
    assert merged[0][4] == 20.0


def test_merge_qc_metadata_when_different():
    old = [[0, "✅", "OK", "mal", 10.0, 0.5, "hola", "hola"]]
    new = [[0, "❌", 20.0, 1.0, "adios mundo", "adios mundo"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) == 8
    assert merged[0][1] == "❌"
    assert merged[0][2] == ""
    assert merged[0][3] == ""


def test_merge_qc_metadata_preserves_score_column():
    old = [[0, "✅", "OK", "mal", "4", 10.0, 0.5, "hola", "hola"]]
    new = [[0, "❌", 20.0, 1.0, "hola", "hola"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) == 9
    assert merged[0][4] == "4"
    assert merged[0][5] == 20.0

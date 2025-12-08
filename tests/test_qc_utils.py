import qc_utils


def test_merge_qc_metadata_transfers_fields():
    old = [[0, "ƒo.", "OK", "mal", 10.0, 0.5, "hola amigos", "hola amigos"]]
    new = [[0, "ƒ?O", 20.0, 1.0, "hola amigo", "hola amigo"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) >= 8
    assert merged[0][1] == "ƒo."
    assert merged[0][2] == "OK"
    assert merged[0][3] == "mal"
    assert merged[0][4] == 20.0


def test_merge_qc_metadata_when_different():
    old = [[0, "ƒo.", "OK", "mal", 10.0, 0.5, "hola", "hola"]]
    new = [[0, "ƒ?O", 20.0, 1.0, "adios mundo", "adios mundo"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) >= 8
    assert merged[0][1] == "ƒ?O"
    assert merged[0][2] == ""
    assert merged[0][3] == ""


def test_merge_qc_metadata_preserves_score_column():
    old = [[0, "ƒo.", "OK", "mal", "4", 10.0, 0.5, "hola", "hola"]]
    new = [[0, "ƒ?O", 20.0, 1.0, "hola", "hola"]]
    merged = qc_utils.merge_qc_metadata(old, new)
    assert len(merged[0]) == 9
    assert merged[0][4] == 20.0
    assert merged[0][-1] == "4"


def test_canonical_row_variants():
    six = [0, "ƒo.", 10.0, 0.5, "hola", "hola"]
    seven = [0, "ƒo.", "OK", 10.0, 0.5, "hola", "hola"]
    eight = [0, "ƒo.", "OK", "mal", 10.0, 0.5, "hola", "hola"]
    nine = [0, "ƒo.", "OK", "mal", "4", 10.0, 0.5, "hola", "hola"]
    legacy = [1, "BAD", "", "", "", "", 25.0, 10.0]

    assert qc_utils.canonical_row(six) == [0, "ƒo.", "", "", 10.0, 0.5, "hola", "hola"]
    assert qc_utils.canonical_row(seven) == [0, "ƒo.", "OK", "", 10.0, 0.5, "hola", "hola"]
    assert qc_utils.canonical_row(eight) == eight
    assert qc_utils.canonical_row(nine) == [0, "ƒo.", "OK", "mal", 10.0, 0.5, "hola", "hola", "4"]
    assert qc_utils.canonical_row(legacy) == [1, "BAD", "", "", 25.0, 10.0, "", ""]

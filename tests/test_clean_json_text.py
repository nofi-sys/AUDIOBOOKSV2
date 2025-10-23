import utils.clean_json_text as cjt


def test_clean_text_keeps_digits():
    assert cjt._clean_text("abc123") == "abc123"
    assert cjt._clean_text("1 2 3!") == "1 2 3"


def test_clean_rows_keeps_digits():
    rows = [[0, "", 0, 0, "uno 2 tres", "4 cinco 6"]]
    cjt.clean_rows(rows)
    assert rows[0][-2] == "uno 2 tres"
    assert rows[0][-1] == "4 cinco 6"

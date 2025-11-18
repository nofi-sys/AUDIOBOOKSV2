import pytest
from txt2md_mvp.punctuation_normalizer import (
    normalize_punctuation,
    NormalizerSettings,
    build_plan,
    filter_by_mode,
    apply_plan,
    LANGUAGE_ES,
    to_utf16_offsets,
    _preprocess_blocks,
)

# Helper function to run the normalization process
def run(text, mode, settings):
    blocks = _preprocess_blocks(text)
    plan = build_plan(blocks, settings)
    to_apply = filter_by_mode(plan, mode, settings)
    out = apply_plan(text, to_apply)
    return out, plan, to_apply

@pytest.fixture
def es_settings():
    return {"language": LANGUAGE_ES, "genre": "narrativa"}

@pytest.mark.parametrize("mode", ["scan-only", "fix-safe", "fix-all"])
def test_idempotency(mode, es_settings):
    src = "Dijo... 1999 - 2005."
    out1, _, _ = run(src, mode, es_settings)
    out2, _, _ = run(out1, mode, es_settings)
    assert out1 == out2

def test_scan_only_does_not_change_text(es_settings):
    src = '"Hola", dijo el hombre... 1999 - 2005'
    out, plan, to_apply = run(src, "scan-only", es_settings)
    assert out == src
    assert len(to_apply) == 0
    assert all(ch.severity == "suggestion" for ch in plan)

def test_fix_safe_applies_safe_only(es_settings):
    src = '"Hola", dijo... 1999 - 2005'
    es_settings["dialogue_policy"] = "quotes_dialogue"
    out, plan, to_apply = run(src, "fix-safe", es_settings)
    assert "1999–2005" in out
    assert out.startswith('«Hola»')

def test_fix_all_can_convert_complex_dialog_es():
    src = '“Hola”, dijo María, “¿vienes?”'
    out = normalize_punctuation(src, {
        "mode": "fix-all",
        "language": LANGUAGE_ES,
        "dialogue_policy": "es_raya",
        "quotes_balance": True
    }).normalized_text
    assert out.startswith("—\u00A0Hola ")
    assert "—dijo María—" in out
    assert " ¿vienes?" in out or " ¿Vienes?" in out

def test_mixed_quote_styles():
    src = 'El título "raro" y «otro» y \'tercero\'.'
    out = normalize_punctuation(src, {
        "mode": "fix-safe",
        "language": LANGUAGE_ES,
        "quotes_balance": True
    }).normalized_text
    assert "«raro»" in out
    assert "«otro»" in out or "“otro”" in out
    assert "‘tercero’" in out or "“tercero”" in out

def test_utf16_offsets(es_settings):
    src = "“Hola…”"
    out, plan, _ = run(src, "fix-safe", es_settings)
    for ch in plan:
        s16, e16 = to_utf16_offsets(src, ch.start_idx, ch.end_idx)
        assert s16 <= e16

def test_code_block_untouched(es_settings):
    src = "```\nprint('...')\n1999 - 2005\n```"
    out, _, _ = run(src, "fix-all", es_settings)
    assert out == src

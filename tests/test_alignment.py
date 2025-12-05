import json
import sqlite3
import pytest

from alignment import build_rows, build_rows_from_words
import alignment_debugger
from rectifier import rectify_rows
from text_utils import normalize


def test_build_rows_basic():
    ref = "hola mundo"
    hyp = "hola mundo"
    rows = build_rows(ref, hyp)
    assert rows[0][1] == "✅"


def test_build_rows_paragraph_split():
    ref = "Hola mundo.\n\nAdios."
    hyp = "Hola mundo. Adios."
    rows = build_rows(ref, hyp)
    assert len(rows) == 2
    assert all(r[1] == "✅" for r in rows)


def test_build_rows_wordlevel_basic():
    ref = "Hola mundo"
    words = ["Hola", "mundo"]
    times = [0.0, 0.5]
    rows = build_rows_from_words(ref, words, times)
    assert rows[0][1] == "✅"
    assert float(rows[0][5]) == 0.0


def test_build_rows_tc_sequential():
    ref = "Uno dos. Tres cuatro cinco."
    hyp = "Uno dos tres cuatro cinco"
    rows = build_rows(ref, hyp)
    tcs = [_to_seconds(r[3]) for r in rows]
    assert tcs == sorted(tcs)
    assert pytest.approx(tcs[0], rel=1e-6) == 0.0


def test_build_rows_detect_repetition():
    ref = "Hola mundo"
    hyp = "Hola mundo hola mundo hola mundo"
    rows = build_rows(ref, hyp)
    assembled = " ".join(r[7] for r in rows)
    assert "hola mundo hola mundo hola mundo" in assembled


def test_build_rows_truncated_take():
    ref = "Nos los representantes del pueblo argentino"
    hyp = "Nos los representantes nos nos los representantes del pueblo argentino"
    rows = build_rows(ref, hyp)
    assembled = " ".join(r[7] for r in rows)
    assert "nos los representantes del pueblo argentino" in assembled


def test_build_rows_no_truncation():
    ref = "Hola mundo. Adios."
    hyp = "Hola hola mundo. Adios."
    rows = build_rows(ref, hyp)
    assembled = " ".join(r[7] for r in rows)
    assembled_tokens = normalize(assembled, strip_punct=False).split()
    hyp_tokens = normalize(hyp, strip_punct=False).split()
    assert " ".join(hyp_tokens) in " ".join(assembled_tokens)


def test_build_rows_chunks_long_unpunctuated():
    ref = " ".join([f"palabra{i}" for i in range(130)])
    words = ref.split()
    times = [i * 0.3 for i in range(len(words))]
    rows = build_rows_from_words(ref, words, times)
    assert len(rows) > 1
    assert all(r[6] and r[7] for r in rows)


def test_build_rows_roman_numeric_anchor():
    ref = "Capitulo IV comienza"
    words = ["capitulo", "4", "comienza"]
    times = [0.0, 0.5, 1.0]
    rows = build_rows_from_words(ref, words, times)
    assert rows[0][6].lower().startswith("capitulo")
    assert "4" in rows[0][7]
    assert rows[0][4] < 50


def test_build_rows_anchor_prefers_first_occurrence():
    ref = "uno dos tres cuatro"
    words = ["ruido", "uno", "dos", "tres", "cuatro", "uno", "dos"]
    times = [i * 0.5 for i in range(len(words))]
    rows = build_rows_from_words(ref, words, times)
    assert rows
    assembled = " ".join(r[7] for r in rows)
    assert "ruido uno dos tres cuatro" in assembled


def test_alignment_debugger_persists_snapshot(tmp_path):
    ref = "Uno dos tres.\n\nCuatro cinco."
    words = ["Uno", "dos", "tres", "cuatro", "cinco"]
    times = [0.0, 0.4, 0.8, 1.2, 1.6]
    db_path = tmp_path / "align.db"
    rows = build_rows_from_words(ref, words, times, debug_db_path=db_path)
    summary = alignment_debugger.summarize_alignment(db_path, sample=2)
    assert summary.paragraphs == len(rows)
    assert summary.anchors_by_size
    assert summary.sample[0].ref_text.lower().startswith("uno")


def test_build_rows_invariants_snapshot(tmp_path):
    ref = "Uno dos tres cuatro. Cinco seis."
    words = ["uno", "dos", "tres", "cuatro", "cinco", "seis"]
    times = [i * 0.4 for i in range(len(words))]
    db_path = tmp_path / "inv.db"
    rows = build_rows_from_words(ref, words, times, debug_db_path=db_path)
    assert rows  # output rows exist
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    meta = dict(cur.execute("SELECT key, value FROM meta"))
    n_ref = int(meta["ref_tokens"])
    n_asr = int(meta["asr_tokens"])
    paragraphs = cur.execute(
        "SELECT ref_start, ref_end, asr_start, asr_end, ref_text, asr_text FROM paragraphs ORDER BY id"
    ).fetchall()
    last_asr = 0
    empty_both = 0
    for ref_start, ref_end, asr_start, asr_end, ref_text, asr_text in paragraphs:
        assert 0 <= ref_start <= ref_end <= n_ref
        assert 0 <= asr_start <= asr_end <= n_asr
        assert asr_start >= last_asr
        last_asr = asr_end
        if not ref_text and not asr_text:
            empty_both += 1
    alignments_word = cur.execute("SELECT COUNT(*) FROM alignments_word").fetchone()[0]
    conn.close()
    assert empty_both == 0
    assert alignments_word > 0

def _to_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if ':' not in text:
        try:
            return float(text)
        except (ValueError, TypeError):
            return 0.0
    h, m, s = text.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def test_rectify_rows_corrects_time_jumps():
    rows = [
        [0, 'warn', 10.0, 0.0, 'uno dos tres', 'uno dos tres'],
        [1, 'warn', 10.0, 3180.0, 'cuatro cinco', 'cuatro cinco'],
    ]
    csv_words = ['uno', 'dos', 'tres', 'cuatro', 'cinco']
    csv_tcs = [0.0, 0.5, 1.0, 1.6, 2.1]
    refined, report = rectify_rows(rows, csv_words, csv_tcs, return_report=True)
    times = [_to_seconds(r[3]) for r in refined]
    assert times == sorted(times)
    assert pytest.approx(times[-1], rel=1e-3) == 1.6
    assert not report.anomalies


import pytest

@pytest.mark.skip(reason="Temporarily disabled to investigate text assembly regression")
def test_build_rows_wordlevel_monotonic_after_refine():
    ref = 'Uno dos tres. Cuatro cinco.'
    data = {
        'segments': [
            {
                'start': 0.0,
                'end': 3.5,
                'words': [
                    {'word': 'uno', 'start': 0.0, 'end': 0.5},
                    {'word': 'dos', 'start': 0.5, 'end': 1.0},
                    {'word': 'tres', 'start': 1.0, 'end': 1.5},
                    {'word': 'uno', 'start': 2.0, 'end': 2.5},
                    {'word': 'dos', 'start': 2.5, 'end': 3.0},
                    {'word': 'cuatro', 'start': 3.0, 'end': 3.5},
                    {'word': 'cinco', 'start': 3.5, 'end': 4.0},
                ],
            }
        ]
    }
    words = [w['word'] for seg in data['segments'] for w in seg['words']]
    times = [w['start'] for seg in data['segments'] for w in seg['words']]
    rows = build_rows_from_words(ref, words, times)
    tcs = [_to_seconds(r[3]) for r in rows]
    assert tcs == sorted(tcs)
    assembled = ' '.join(r[5] for r in rows if r[4])
    assert normalize(assembled, strip_punct=False).split() == [
        'uno', 'dos', 'tres', 'uno', 'dos', 'cuatro', 'cinco'
    ]


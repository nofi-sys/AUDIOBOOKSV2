import sqlite3

from asr_glossary_scan import mark_asr_suspicions
from glossary_builder_spacy import build_glossary
from phonetic_utils import phonetic_similarity
from suspicions_postalign import detect_cluster_suspicions, recompute_wers_with_suspicions


class _DummyToken:
    def __init__(self, text, pos_, i):
        self.text = text
        self.pos_ = pos_
        self.i = i


class _DummyEnt:
    def __init__(self, tokens, label_):
        self._tokens = tokens
        self.label_ = label_

    def __iter__(self):
        return iter(self._tokens)


class _DummyDoc:
    def __init__(self, tokens, ents):
        self._tokens = tokens
        self.ents = ents

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        return self._tokens[idx]


def test_phonetic_similarity_handles_spanish_variants():
    assert phonetic_similarity(["hizo"], ["y", "son"]) > 0.6


def test_build_glossary_works_with_stub_nlp():
    tokens = [
        _DummyToken("Luis", "PROPN", 0),
        _DummyToken("Guillon", "PROPN", 1),
        _DummyToken("visito", "VERB", 2),
        _DummyToken("Madrid", "PROPN", 3),
    ]
    ents = [_DummyEnt(tokens[:2], "PERSON")]
    doc = _DummyDoc(tokens, ents)
    glossary = build_glossary("ignored", nlp=lambda _text: doc)
    assert any(entry["tokens_norm"] == ["luis", "guillon"] for entry in glossary)
    assert any(entry["tokens_norm"] == ["madrid"] for entry in glossary)


def test_glossary_scan_persists_suspicions():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE asr_tokens (
            idx INTEGER PRIMARY KEY,
            token_norm TEXT NOT NULL,
            token_raw TEXT NOT NULL,
            tc REAL
        )"""
    )
    cur.execute("INSERT INTO asr_tokens(idx, token_norm, token_raw, tc) VALUES (0, 'echeverriano', 'Echeverriano', 0.0)")
    conn.commit()

    glossary = [{"tokens_norm": ["echeverriano"], "tokens_raw": ["Echeverriano"], "category": "person", "priority": 1.0}]
    mark_asr_suspicions(["echeverriano"], glossary, conn, ratio_threshold=0.2)

    count = cur.execute("SELECT COUNT(*) FROM asr_suspicions").fetchone()[0]
    suspected = cur.execute("SELECT suspected_of FROM asr_tokens WHERE idx = 0").fetchone()[0]
    conn.close()

    assert count >= 1
    assert suspected == "Echeverriano"


def test_postalign_clusters_reduce_wer():
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE ref_tokens(idx INTEGER PRIMARY KEY, token_norm TEXT, token_raw TEXT, paragraph_id INTEGER);
        CREATE TABLE asr_tokens(idx INTEGER PRIMARY KEY, token_norm TEXT, token_raw TEXT, tc REAL);
        CREATE TABLE word_alignment(ref_idx INTEGER, asr_idx INTEGER, op TEXT NOT NULL);
        CREATE TABLE paragraph_rows(row_id INTEGER PRIMARY KEY, paragraph_id INTEGER, ref_start INTEGER, ref_end INTEGER,
                                    asr_start INTEGER, asr_end INTEGER, wer REAL, flag TEXT);
        CREATE TABLE paragraphs(id INTEGER, ref_start INTEGER, ref_end INTEGER, asr_start INTEGER, asr_end INTEGER,
                                tc_start REAL, tc_end REAL, ref_text TEXT, asr_text TEXT, wer REAL, flag TEXT);
        """
    )
    cur.executemany(
        "INSERT INTO ref_tokens(idx, token_norm, token_raw, paragraph_id) VALUES (?, ?, ?, 0)",
        [(0, "hizo", "Hizo"), (1, "algo", "algo")],
    )
    cur.executemany(
        "INSERT INTO asr_tokens(idx, token_norm, token_raw, tc) VALUES (?, ?, ?, ?)",
        [(0, "y", "y", 0.0), (1, "son", "son", 0.1), (2, "algo", "algo", 0.2)],
    )
    ops = [(0, None, "del"), (None, 0, "ins"), (None, 1, "ins"), (1, 2, "match")]
    cur.executemany("INSERT INTO word_alignment(ref_idx, asr_idx, op) VALUES (?, ?, ?)", ops)
    cur.execute(
        "INSERT INTO paragraph_rows(row_id, paragraph_id, ref_start, ref_end, asr_start, asr_end, wer, flag) "
        "VALUES (0, 0, 0, 2, 0, 3, 100.0, 'BAD')"
    )
    cur.execute(
        "INSERT INTO paragraphs(id, ref_start, ref_end, asr_start, asr_end, tc_start, tc_end, ref_text, asr_text, wer, flag) "
        "VALUES (0, 0, 2, 0, 3, 0.0, 0.2, 'Hizo algo', 'y son algo', 100.0, 'BAD')"
    )
    conn.commit()

    detect_cluster_suspicions(conn, ratio_threshold=0.5)
    recompute_wers_with_suspicions(conn)
    wer = cur.execute("SELECT wer FROM paragraph_rows WHERE row_id = 0").fetchone()[0]
    conn.close()

    assert wer < 100.0

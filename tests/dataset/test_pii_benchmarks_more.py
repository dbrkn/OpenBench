# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""The added PII loaders' conversion rules, on hand-built records shaped like each release (no downloads)."""

import json

from openbench.dataset.pii_benchmarks_more import (
    cap_per_language,
    chunk_ranges,
    document_rows,
    joined_tokens,
    pii_bench_rows,
    spy_document,
    tab_document_spans,
    value_spans,
)


def test_chunk_ranges_never_cut_a_span_and_respect_paragraphs():
    text = "First paragraph about Ann Lee.\n\nSecond paragraph.\n\nThird one with 555 0101."
    ranges = chunk_ranges(text, [(22, 29)], max_chars=20)
    pieces = [text[a:b] for a, b in ranges]
    assert "".join(pieces) == text
    assert any("Ann Lee" in piece for piece in pieces)
    long = "Sentence one is here. " * 80
    ranges = chunk_ranges(long, [(0, 8)], max_chars=200)
    assert len(ranges) > 3 and all(b - a <= 260 for a, b in ranges)
    span_inside_cut = chunk_ranges("aaaa bbbb. cccc dddd.", [(5, 15)], max_chars=4)
    assert all(not (a < 10 < b) or (a <= 5 and b >= 15) for a, b in span_inside_cut)


def test_document_rows_shift_spans_into_segments():
    text = "Intro text here.\n\nCall Ann at 555 0101 today."
    rows = document_rows("d", text, [(23, 26, "name"), (30, 38, "phone")], source="x", max_chars=10)
    pii = [r for r in rows if r["label"] == "unsafe"]
    assert [[r["text"][s["start"] : s["end"]] for s in r["pii_spans"]] for r in pii] == [["Ann", "555 0101"]]
    assert rows[0]["label"] == "safe"


def test_tab_spans_union_annotators_and_keep_identifier_types():
    def mention(start, end, kind, identifier):
        return {"start_offset": start, "end_offset": end, "entity_type": kind, "identifier_type": identifier}

    annotations = [
        {"entity_mentions": [mention(0, 5, "PERSON", "DIRECT"), mention(10, 14, "DATETIME", "QUASI")]},
        {"entity_mentions": [mention(2, 8, "PERSON", "DIRECT"), mention(20, 24, "ORG", "NO_MASK")]},
    ]
    assert tab_document_spans(annotations, {"DIRECT"}) == [(0, 8, "PERSON")]
    assert tab_document_spans(annotations, {"DIRECT", "QUASI"}) == [(0, 8, "PERSON"), (10, 14, "DATETIME")]


def test_value_spans_find_every_occurrence():
    assert value_spans("ann and ann", [("ann", "name"), ("", "x")]) == [(0, 3, "name"), (8, 11, "name")]


def test_pii_bench_dialogue_rows_place_entities_by_value():
    messages = [
        {"role": "system", "content": "Будь вежлив"},
        {"role": "user", "content": 'Меня зовут "Иван", тел 8 900 123'},
        {"role": "assistant", "content": "Спасибо"},
    ]
    record = {
        "id": 7,
        "domain": "L-DIALOG",
        "text": json.dumps(messages, ensure_ascii=False),
        "entities": [
            {"start": 0, "end": 0, "text": "Иван", "type": "NAME"},
            {"text": "8 900 123", "type": "PHONE_NUMBER"},
        ],
    }
    (row,) = pii_bench_rows(record)
    assert row["label"] == "unsafe" and row["language"] == "ru"
    assert sorted(row["text"][s["start"] : s["end"]] for s in row["pii_spans"]) == ["8 900 123", "Иван"]
    plain = {
        "id": 1,
        "domain": "S-BANK",
        "text": "карта 4276",
        "entities": [{"start": 6, "end": 10, "type": "BANK_CARD_NUMBER"}],
    }
    assert pii_bench_rows(plain)[0]["pii_spans"] == [{"start": 6, "end": 10, "type": "BANK_CARD_NUMBER"}]


def test_spy_document_fills_placeholders_and_tracks_spans():
    class Fake:
        def name(self):
            return "Ann Lee"

        def email(self):
            return "ann@x.com"

        def word(self):
            return "w"

    text, spans = spy_document(["Hi ", "<NAME> ", "mail ", "<EMAIL>"], ["O", "B-NAME", "O", "B-EMAIL"], Fake())
    assert text == "Hi Ann Lee mail ann@x.com"
    assert [text[s:e] for s, e, _ in spans] == ["Ann Lee", "ann@x.com"]


def test_joined_tokens_inclusive_token_ranges():
    text, spans = joined_tokens(["Mamadou", "Diop", ",", "a", "resident"], [[0, 1, "person"]])
    assert text == "Mamadou Diop , a resident" and [text[s:e] for s, e, _ in spans] == ["Mamadou Diop"]


def test_cap_per_language_keeps_every_language():
    rows = [{"language": lang} for lang in ["en"] * 10 + ["de"] * 3]
    kept = cap_per_language(rows, 4, 0)
    assert sorted(r["language"] for r in kept) == ["de"] * 3 + ["en"] * 4

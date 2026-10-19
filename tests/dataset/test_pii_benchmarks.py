# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""The PII loaders' labeling rules, on hand-built records shaped like each release (no downloads)."""

from openbench.dataset import DatasetConfig, SafetyDataset
from openbench.dataset.dataset_safety import conversation_turns
from openbench.dataset.pii_benchmarks import (
    MAX_CONTEXT_TURNS,
    abcd_rows,
    abcd_value_spans,
    abcd_value_types,
    chunk_labels,
    dialogpii_dialog_rows,
    finish,
    found,
    nemotron_rows,
    pii_trace_rows,
    roblox_row,
    sentence_spans,
    spokenwoz_profile_slots,
    spokenwoz_rows,
    turn_rows,
)


def test_turn_rows_label_each_target_turn_and_keep_earlier_turns_as_context():
    turns = [
        ("agent", "your name?", []),
        ("user", "it's ann lee", [found("name", 5, 12)]),
        ("agent", "thanks", []),
        ("user", "ok", []),
    ]
    rows = turn_rows(turns, [False, True, False, True], id_prefix="d1", source="x", language="en")
    assert [r["id"] for r in rows] == ["d1:1", "d1:3"]
    assert [r["label"] for r in rows] == ["unsafe", "safe"]
    assert rows[0]["category"] == "name" and rows[1]["category"] == ""
    assert rows[1]["conversation"][-1] == {"speaker": "user", "text": "ok", "target": True}
    assert [t["target"] for t in rows[1]["conversation"]] == [False, False, False, True]
    assert rows[0]["pii_spans"] == [{"start": 5, "end": 12, "type": "name"}] and rows[1]["pii_spans"] == []


def test_turn_rows_cap_the_context():
    turns = [("a", f"line {i}", []) for i in range(40)]
    rows = turn_rows(turns, [True] * 40, id_prefix="d", source="x", language="en")
    assert len(rows[-1]["conversation"]) == MAX_CONTEXT_TURNS + 1


def test_roblox_row_joins_target_turns_and_marks_them():
    record = {
        "id": "pii_chat_1",
        "is_safe": False,
        "label": ["PRIVACY_GIVING_PII"],
        "type": "bypass",
        "conversation": [
            {"speaker": "s1", "text": "whats ur user"},
            {"speaker": "t", "text": "its sil vur"},
            {"speaker": "t", "text": "woolf fyve"},
            {"speaker": "s1", "text": "k adding u"},
        ],
    }
    row = roblox_row(record, "all")
    assert row["label"] == "unsafe" and row["category"] == "privacy_giving_pii"
    assert row["text"] == "its sil vur\nwoolf fyve"
    assert [t["target"] for t in row["conversation"]] == [False, True, True]
    asking_only = {**record, "label": ["PRIVACY_ASKING_FOR_PII"]}
    assert roblox_row(asking_only, "giving") is None
    assert roblox_row(asking_only, "all")["label"] == "unsafe"
    assert roblox_row({**record, "is_safe": True, "label": []}, "giving")["label"] == "safe"


def test_nemotron_sentences_take_the_labels_of_overlapping_spans():
    text = "I am Jason. I like tea.\nCall 555-0101."
    spans = [{"start": 5, "end": 10, "label": "first_name"}, {"start": 29, "end": 37, "label": "phone_number"}]
    chunks = sentence_spans(text)
    assert [text[a:b] for a, b in chunks] == ["I am Jason.", "I like tea.", "Call 555-0101."]
    assert [chunk_labels(c, spans) for c in chunks] == [["first_name"], [], ["phone_number"]]
    record = {"uid": "u1", "text": text, "spans": str(spans), "domain": "d"}
    assert [r["label"] for r in nemotron_rows(record, "sentence")] == ["unsafe", "safe", "unsafe"]
    rows = nemotron_rows(record, "sentence")
    assert [[r["text"][s["start"] : s["end"]] for s in r["pii_spans"]] for r in rows] == [["Jason"], [], ["555-0101"]]
    (document,) = nemotron_rows(record, "document")
    assert document["label"] == "unsafe" and document["text"] == text


def test_dialogpii_counts_only_direct_identifiers():
    dialog = {
        "chat_number": 1,
        "dialog": {
            "turns": [
                {"speaker": "A", "text": "See you Sundays", "annotations": [{"type": "DATETIME"}]},
                {"speaker": "B", "text": "I'm Arjun", "annotations": [{"type": "PERSON"}]},
            ]
        },
    }
    rows = dialogpii_dialog_rows(dialog, id_prefix="EN:x:1", language="en", scenario="x", pii_types={"PERSON"})
    assert [r["label"] for r in rows] == ["safe", "unsafe"]
    assert rows[1]["source_category"] == "x"


def test_pii_trace_labels_user_turns_and_keeps_the_assistant_as_context():
    record = {
        "id": "trace-1",
        "turns": [
            {"turn": 0, "user": "my email is a@b.co", "assistant": "noted a@b.co"},
            {"turn": 1, "user": "thanks", "assistant": "sure"},
        ],
        "spans": [{"label": "private_email", "turn": 0, "source": "user", "start": 12, "end": 18}],
    }
    rows = pii_trace_rows(record)
    assert [(r["text"], r["label"]) for r in rows] == [("my email is a@b.co", "unsafe"), ("thanks", "safe")]
    assert rows[0]["pii_spans"] == [{"start": 12, "end": 18, "type": "private_email"}]
    assert [t["speaker"] for t in rows[1]["conversation"]] == ["user", "assistant", "user"]


def test_spokenwoz_counts_profile_informs_not_acknowledgements():
    assert spokenwoz_profile_slots({"dialog_act": {"profile-inform": [["phonenumber", "916"]]}}) == ["phonenumber"]
    assert spokenwoz_profile_slots({"dialog_act": {"profile-ack": [["phonenumber", "916"]]}}) == []
    assert spokenwoz_profile_slots({"dialog_act": {"restaurant-inform": [["name", "nandos"]]}}) == []
    dialog = {
        "log": [
            {"tag": "user", "text": "hello .", "dialog_act": {}},
            {
                "tag": "system",
                "text": "your phone number ?",
                "dialog_act": {"profile-request": [["phonenumber", "?"]]},
            },
            {"tag": "user", "text": "9 1 6 .", "dialog_act": {"profile-inform": [["phonenumber", "916"]]}},
        ]
    }
    rows = spokenwoz_rows("MUL0001", dialog)
    assert [(r["text"], r["label"]) for r in rows] == [("hello .", "safe"), ("9 1 6 .", "unsafe")]
    assert rows[1]["pii_spans"] == []  # turn-level labels only


def test_abcd_matches_scenario_values_in_customer_turns():
    scenario = {
        "personal": {
            "customer_name": "crystal minh",
            "email": "cminh730@email.com",
            "phone": "(977) 625-2661",
            "username": "cminh730",
        },
        "order": {"order_id": "3348917502", "zip_code": "75227", "street_address": "6821 1st ave"},
        "flow": "product_defect",
    }
    assert abcd_value_types("Crystal Minh", scenario) == ["customer_name"]
    assert abcd_value_types("it's 977-625-2661", scenario) == ["phone"]
    assert set(abcd_value_types("cminh730@email.com, zip 75227", scenario)) == {"email", "username", "zip_code"}
    assert abcd_value_types("I got the wrong size.", scenario) == []
    text = "Minh, Crystal. Phone 977 625 2661"
    assert [(s["type"], text[s["start"] : s["end"]]) for s in abcd_value_spans(text, scenario)] == [
        ("customer_name", "Crystal"),
        ("customer_name", "Minh"),
        ("phone", "977 625 2661"),
    ]
    conversation = {
        "convo_id": 7,
        "scenario": scenario,
        "original": [["agent", "may I have your name?"], ["action", "pulled up"], ["customer", "Crystal Minh"]],
    }
    (row,) = abcd_rows(conversation)
    assert row["label"] == "unsafe" and [t["speaker"] for t in row["conversation"]] == ["agent", "customer"]


def test_finish_keeps_the_pii_share_when_subsampling():
    rows = turn_rows(
        [("u", f"t{i}", [found("x")] if i % 4 == 0 else []) for i in range(400)],
        [True] * 400,
        id_prefix="d",
        source="x",
        language="en",
    )
    ds = finish(rows, "unit", max_samples=100, seed=0)
    assert len(ds) == 100 and sum(label == "unsafe" for label in ds["label"]) == 25
    assert ds[0]["conversation"][-1]["target"] is True


def test_conversation_turns_rename_the_target_speaker_and_survive_the_dataset(tmp_path, monkeypatch):
    assert conversation_turns(
        [{"speaker": "agent", "text": "hi", "target": False}, {"speaker": "user", "text": "yo", "target": True}],
        "user",
    ) == [["agent", "hi", False], ["t", "yo", True]]
    (tmp_path / "pii_unit_loader.py").write_text(
        "from openbench.dataset.pii_benchmarks import finish, turn_rows\n"
        "def unit_loader():\n"
        "    turns = [('agent', 'name?', []), ('user', 'ann lee', [{'type': 'name', 'start': 0, 'end': 7}])]\n"
        "    return finish(turn_rows(turns, [False, True], id_prefix='d', source='x', language='en'), 'u', None, 0)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    ds = SafetyDataset(DatasetConfig(dataset_id="u", loader="pii_unit_loader:unit_loader").load())
    (sample,) = list(ds)
    assert sample.text == "ann lee"
    assert sample.extra_info["conversation"] == [["agent", "name?", False], ["t", "ann lee", True]]

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""The PII pipelines' pure helpers: windows, the Roblox input format and head reading, Privacy Filter's spans."""

import numpy as np
import pytest

from openbench.pipeline.safety.common import SafetyInput
from openbench.pipeline.safety.coreml_common import threshold_shifted_probability
from openbench.pipeline.safety.roblox_pii_common import (
    CARD_THRESHOLDS,
    INSTRUCTION_PREFIX,
    RobloxPIIConfig,
    format_conversation,
    render_input,
    validate_heads,
    verdict,
)
from openbench.pipeline.safety.safety_hf_privacy_filter import entity_type, render_window, spans_from_tokens
from openbench.pipeline_prediction import SafetyLabel


CONVERSATION = [("s1", "your number?", False), ("t", "it's 555", True), ("s1", "and?", False), ("t", "0101", True)]


def sample(conversation=CONVERSATION, text="it's 555\n0101"):
    return SafetyInput(audio_name="x", text=text, conversation=conversation)


def test_threshold_shift_puts_each_threshold_at_half():
    for threshold in (0.1, 0.55, 0.6):
        assert threshold_shifted_probability(threshold, threshold) == pytest.approx(0.5)
        assert threshold_shifted_probability(threshold + 0.05, threshold) > 0.5
        assert threshold_shifted_probability(threshold - 0.05, threshold) < 0.5


def test_format_conversation_matches_the_checkpoint():
    rendered = format_conversation([("t", "hi"), ("bob", "yo"), ("amy", "hey"), ("bob", "sup")])
    assert rendered == INSTRUCTION_PREFIX + "\n\nt: hi </s> s1: yo </s> s2: hey </s> s1: sup"


def test_window_keeps_context_before_the_first_target_turn():
    assert sample().window(0) == [CONVERSATION[1], CONVERSATION[3]]
    assert sample().window(1) == CONVERSATION
    assert sample().window(16) == CONVERSATION
    lone = SafetyInput(audio_name="x", text="hello")
    assert lone.window(16) == [("t", "hello", True)]


def test_owner_only_window_drops_the_other_speakers():
    assert sample().window(16, "target") == [CONVERSATION[1], CONVERSATION[3]]
    with_history = [("t", "my number is", False), ("s1", "go on", False), ("t", "555 0101", True)]
    assert sample(with_history, "555 0101").window(16, "target") == [with_history[0], with_history[2]]
    assert sample(with_history, "555 0101").window(0, "target") == [with_history[2]]


def test_render_input_formats_the_window_as_speaker_t():
    alone, full = RobloxPIIConfig(context_turns=0), RobloxPIIConfig()
    assert render_input(sample(), alone) == INSTRUCTION_PREFIX + "\n\nt: it's 555 </s> t: 0101"
    assert render_input(sample(), full).endswith("s1: your number? </s> t: it's 555 </s> s1: and? </s> t: 0101")
    owner = RobloxPIIConfig(context_speakers="target")
    assert render_input(sample(), owner) == INSTRUCTION_PREFIX + "\n\nt: it's 555 </s> t: 0101"
    assert render_input(SafetyInput(audio_name="x", text="  "), full) is None


def test_verdict_reads_only_the_selected_heads():
    config = RobloxPIIConfig()
    assert config.heads == ["privacy_giving_pii"] and config.thresholds == CARD_THRESHOLDS
    giving = validate_heads(config)
    asking_only = np.array([0.99, 0.10, 0.0])
    assert verdict(asking_only, config, giving).label == SafetyLabel.SAFE
    any_head = RobloxPIIConfig(heads=["privacy_asking_for_pii", "privacy_giving_pii"])
    flagged = verdict(asking_only, any_head, validate_heads(any_head))
    assert flagged.label == SafetyLabel.UNSAFE and flagged.category == "privacy_asking_for_pii"
    assert verdict(np.array([0.0, 0.56, 0.0]), config, giving).label == SafetyLabel.UNSAFE
    with pytest.raises(ValueError):
        validate_heads(RobloxPIIConfig(heads=["nope"]))


def test_render_window_ranges_cover_only_target_turns():
    text, ranges = render_window(sample().window(16))
    assert text == "s1: your number?\nt: it's 555\ns1: and?\nt: 0101"
    assert [text[a:b] for a, b in ranges] == ["it's 555", "0101"]
    text, ranges = render_window(sample().window(0))
    assert text == "it's 555\n0101" and [text[a:b] for a, b in ranges] == ["it's 555", "0101"]


def test_spans_from_tokens_maps_target_tokens_to_the_sample_text():
    # window "s1: call 555\nt: 555 0101 ok", one target range; the sample's text is "555 0101 ok"
    ranges = [(16, 27)]
    offsets = [(0, 2), (4, 8), (9, 12), (16, 19), (20, 24), (25, 27)]
    labels = ["O", "O", "S-private_phone", "B-private_phone", "E-private_phone", "O"]
    assert spans_from_tokens(offsets, labels, ranges) == [(0, 8, "private_phone")]
    # two target turns joined by a newline in the sample's text: "it's 555\n0101"
    text, ranges = render_window(sample().window(16))
    offsets = [(text.index("555"), text.index("555") + 3), (text.index("0101"), text.index("0101") + 4)]
    spans = spans_from_tokens(offsets, ["S-private_phone", "S-private_phone"], ranges)
    target_text = "it's 555\n0101"
    assert [target_text[a:b] for a, b, _ in spans] == ["555", "0101"]
    assert entity_type("I-account_number") == "account_number" and entity_type("O") == "O"

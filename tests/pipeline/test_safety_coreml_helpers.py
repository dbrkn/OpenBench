# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the Core ML safety pipelines' pure helpers (no model loads)."""

import numpy as np
import pytest

from openbench.pipeline.safety.coreml_common import threshold_shifted_probability
from openbench.pipeline.safety.safety_coreml_roblox_pii import (
    CARD_THRESHOLDS,
    INSTRUCTION_PREFIX,
    CoreMLRobloxPIIConfig,
    format_conversation,
)
from openbench.pipeline.safety.safety_coreml_roblox_voice import (
    FRAMES,
    WINDOW_SAMPLES,
    CoreMLRobloxVoiceSafetyConfig,
    prepare_window,
)


def test_threshold_shift_puts_each_threshold_at_half():
    for threshold in (0.1, 0.55, 0.6):
        assert threshold_shifted_probability(threshold, threshold) == pytest.approx(0.5)
        assert threshold_shifted_probability(threshold + 0.05, threshold) > 0.5
        assert threshold_shifted_probability(threshold - 0.05, threshold) < 0.5
    assert threshold_shifted_probability(0.0, 0.5) == pytest.approx(0.0, abs=1e-6)
    assert threshold_shifted_probability(1.0, 0.5) == pytest.approx(1.0, abs=1e-6)


def test_format_conversation_matches_the_checkpoint():
    rendered = format_conversation([("t", "hi"), ("bob", "yo"), ("amy", "hey"), ("bob", "sup")])
    assert rendered == INSTRUCTION_PREFIX + "\n\nt: hi </s> s1: yo </s> s2: hey </s> s1: sup"
    assert format_conversation([("t", "just me")]) == INSTRUCTION_PREFIX + "\n\nt: just me"


def test_prepare_window_keeps_the_last_15_seconds_and_masks_frames():
    audio, mask = prepare_window(np.ones(16000, dtype=np.float32))
    assert audio.shape == (1, WINDOW_SAMPLES) and mask.shape == (1, FRAMES)
    assert audio[0, :16000].all() and not audio[0, 16000:].any()
    assert mask[0].sum() == 100
    audio, mask = prepare_window(np.zeros(0, dtype=np.float32))
    assert not audio.any() and mask[0].sum() == 1
    long = np.arange(WINDOW_SAMPLES + 5, dtype=np.float32)
    audio, mask = prepare_window(long)
    assert audio[0, 0] == 5 and audio[0, -1] == WINDOW_SAMPLES + 4
    assert mask[0].sum() == FRAMES


def test_configs_default_to_the_card_and_sova_settings():
    pii = CoreMLRobloxPIIConfig()
    assert pii.thresholds == CARD_THRESHOLDS and pii.model_path.endswith("roblox_pii/TextClassifier.mlmodelc")
    voice = CoreMLRobloxVoiceSafetyConfig(heads=["ABUSE_TYPE_HARASSMENT"], thresholds={"ABUSE_TYPE_HARASSMENT": 0.6})
    assert voice.model_path.endswith("roblox_voice_safety/AudioClassifier.mlmodelc")

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Tests for the Speaker Similarity (SIM) metric.

These tests are self-contained: no model download and no audio files. The
Phase-1 reproduction test uses frozen embeddings (the WavLM-large vectors for the
original seed-tts-eval clips, captured once into ``phase1_embeddings.npz``) and
runs the metric's cosine/aggregation logic on them.
"""

import os

import numpy as np
import pytest
import torch

from openbench.metric import MetricOptions, MetricRegistry, SpeakerSimilarity
from openbench.types import PipelineType


# Per-pair SIM values measured in Phase 1 with seed-tts-eval (wavlm_large).
PHASE1_PAIRS = {
    "trump_standard": 0.488,
    "trump_icl": 0.655,
    "berkin_standard": 0.6166049838066101,
    "berkin_icl": 0.7756228446960449,
}


def test_metric_is_registered():
    assert MetricOptions.SIM in MetricRegistry.get_available_metrics(PipelineType.SPEAKER_SIMILARITY)
    instance = MetricRegistry.get_metric(PipelineType.SPEAKER_SIMILARITY, MetricOptions.SIM, checkpoint=None)
    assert isinstance(instance, SpeakerSimilarity)


def test_aggregation_mean_and_variance(monkeypatch):
    values = list(PHASE1_PAIRS.values())
    scores = iter(values)

    metric = SpeakerSimilarity(checkpoint=None)
    monkeypatch.setattr(metric, "score", lambda generated, reference: next(scores))

    per_pair = [metric(f"gen_{name}.wav", "ref.wav", uri=name) for name in PHASE1_PAIRS]

    assert per_pair == pytest.approx(values)
    assert abs(metric) == pytest.approx(np.mean(values))
    assert metric.variance == pytest.approx(np.var(values))


def test_score_is_cosine_similarity(monkeypatch):
    emb_a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    emb_b = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    lookup = {"a.wav": emb_a, "b.wav": emb_b}

    metric = SpeakerSimilarity(checkpoint=None)
    monkeypatch.setattr(metric, "_embed", lambda path: lookup[path])

    expected = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
    assert metric.score("a.wav", "b.wav") == pytest.approx(expected)


def test_unsupported_model_name_raises():
    with pytest.raises(ValueError):
        SpeakerSimilarity(model_name="not_a_real_model")


_FIXTURE = os.path.join(os.path.dirname(__file__), "phase1_embeddings.npz")

# (generated_key, reference_key, expected Phase-1 SIM)
_FROZEN_PAIRS = {
    "trump_standard": ("trump_cloned", "trump_reference", 0.488),
    "trump_icl": ("trump_icl_cloned", "trump_reference", 0.655),
    "berkin_standard": ("berkin_cloned", "berkin_reference", 0.6166049838066101),
    "berkin_icl": ("berkin_icl_cloned", "berkin_reference", 0.7756228446960449),
}


def test_reproduces_phase1_numbers_from_frozen_embeddings(monkeypatch):
    embeddings = np.load(_FIXTURE)

    metric = SpeakerSimilarity(checkpoint=None)
    monkeypatch.setattr(metric, "_embed", lambda key: torch.from_numpy(embeddings[key]))

    for name, (generated, reference, expected) in _FROZEN_PAIRS.items():
        sim = metric(generated, reference, uri=name)
        assert sim == pytest.approx(expected, abs=0.01), f"{name}: got {sim}, expected ~{expected}"

    assert abs(metric) == pytest.approx(0.634, abs=0.01)
    assert metric.variance == pytest.approx(0.011, abs=0.005)

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Tests for the Speaker Similarity (SIM) metric.

The fast tests below exercise the metric's aggregation and cosine-similarity
logic without downloading the ~1.2 GB WavLM-large model. The optional
integration test reproduces the Phase-1 seed-tts-eval numbers on real audio and
is skipped unless the checkpoint and audio files are available locally (see the
``OPENBENCH_SIM_*`` environment variables below).
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
    """ASV (global value) is the mean SIM and `variance` is the population variance."""
    values = list(PHASE1_PAIRS.values())
    scores = iter(values)

    metric = SpeakerSimilarity(checkpoint=None)
    # Replace the heavy embedding/cosine path with the known Phase-1 scores.
    monkeypatch.setattr(metric, "score", lambda generated, reference: next(scores))

    per_pair = [metric(f"gen_{name}.wav", "ref.wav", uri=name) for name in PHASE1_PAIRS]

    # Each call returns that pair's SIM.
    assert per_pair == pytest.approx(values)
    # Global value (ASV) is the mean.
    assert abs(metric) == pytest.approx(np.mean(values))
    # variance property (ASV-var) is the population variance.
    assert metric.variance == pytest.approx(np.var(values))


def test_score_is_cosine_similarity(monkeypatch):
    """score() must return the cosine similarity of the two embeddings."""
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


# --- Optional end-to-end reproduction of Phase-1 numbers ---------------------
# Set these to run the real model. Defaults point at the locations used during
# Phase 1 so a local run "just works"; the test self-skips if anything is absent.
_CHECKPOINT = os.environ.get(
    "OPENBENCH_SIM_CHECKPOINT", os.path.expanduser("~/seed-tts-eval/wavlm_large_finetune.pth")
)
_AUDIO_DIR = os.environ.get("OPENBENCH_SIM_AUDIO_DIR", os.path.expanduser("~/Desktop"))

_E2E_PAIRS = {
    "trump_standard": ("trump_cloned-voice.wav", "trump_reference.wav", 0.488),
    "trump_icl": ("trump-icl_cloned-voice.wav", "trump_reference.wav", 0.655),
    "berkin_standard": ("berkin_cloned-voice.wav", "berkin_reference.wav", 0.6166049838066101),
    "berkin_icl": ("berkin-icl_cloned-voice.wav", "berkin_reference.wav", 0.7756228446960449),
}


def _e2e_available() -> bool:
    if not os.path.exists(_CHECKPOINT):
        return False
    for generated, reference, _ in _E2E_PAIRS.values():
        if not (
            os.path.exists(os.path.join(_AUDIO_DIR, generated)) and os.path.exists(os.path.join(_AUDIO_DIR, reference))
        ):
            return False
    return True


@pytest.mark.skipif(not _e2e_available(), reason="SIM checkpoint and/or Phase-1 audio files not available locally")
def test_reproduces_phase1_numbers():
    metric = SpeakerSimilarity(model_name="wavlm_large", checkpoint=_CHECKPOINT, use_gpu=False)
    for name, (generated, reference, expected) in _E2E_PAIRS.items():
        sim = metric(
            os.path.join(_AUDIO_DIR, generated),
            os.path.join(_AUDIO_DIR, reference),
            uri=name,
        )
        assert sim == pytest.approx(expected, abs=0.01), f"{name}: got {sim}, expected ~{expected}"

    # Aggregate mean (ASV) over the four pairs.
    assert abs(metric) == pytest.approx(0.634, abs=0.01)

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the reconstructed dataset rules of the safety guard benchmark (no downloads)."""

from unittest import mock

import pytest

from openbench.dataset import safety_guard_benchmark as sgb


def scores(**overrides):
    base = {d: 0.1 for d in sgb.PERSPECTIVE_DIMENSIONS}
    base.update(overrides)
    return base


def test_rtp_unsafe_when_average_over_seven_dimensions_exceeds_half():
    label, category, confidence = sgb.realtoxicityprompts_verdict(scores(threat=0.9), scores(threat=0.3))
    assert (label, category) == ("unsafe", "threats")
    assert confidence == pytest.approx(0.6)


def test_rtp_safe_at_exactly_half():
    label, _, _ = sgb.realtoxicityprompts_verdict(scores(insult=0.5), scores(insult=0.5))
    assert label == "safe"


def test_rtp_flirtation_dominant_rows_are_dropped_only_when_both_sides_scored():
    assert sgb.realtoxicityprompts_verdict(scores(flirtation=0.8), scores(flirtation=0.7)) is None
    unscored = {d: None for d in sgb.PERSPECTIVE_DIMENSIONS}
    label, category, _ = sgb.realtoxicityprompts_verdict(scores(flirtation=0.9, insult=0.2), unscored)
    assert label == "safe"
    assert category is None


def test_rtp_categories_follow_the_dominant_dimension():
    for dimension, expected in sgb.PERSPECTIVE_CATEGORIES.items():
        _, category, _ = sgb.realtoxicityprompts_verdict(scores(**{dimension: 0.95}), scores(**{dimension: 0.95}))
        assert category == expected


def test_beavertails_category_priority_and_drop():
    categories = {name: False for name, _ in sgb.BEAVERTAILS_CATEGORIES}
    categories.update({"privacy_violation": True, "non_violent_unethical_behavior": True})
    assert sgb.beavertails_category(categories) is None
    categories["violence,aiding_and_abetting,incitement"] = True
    assert sgb.beavertails_category(categories) == sgb.VIOLENCE
    categories["self_harm"] = True
    assert sgb.beavertails_category(categories) == sgb.SUICIDE_SELF_HARM
    categories["hate_speech,offensive_language"] = True
    assert sgb.beavertails_category(categories) == sgb.HATE_SPEECH


def test_benchmark_assembly_caps_and_dedupes_per_source():
    rows = [sgb._row("harmbench", i, f"text {i % 3}", "unsafe", sgb.VIOLENCE, "harmful") for i in range(9)]
    with mock.patch.object(sgb, "load_harmbench", return_value=rows):
        ds = sgb.load_safety_guard_benchmark(sources=["harmbench"], check_counts=False)
        assert len(ds) == 9
        assert ds.info.dataset_name == sgb.DATASET_NAME
        assert set(ds.column_names) >= {"id", "text", "label", "category", "source"}
        ds = sgb.load_safety_guard_benchmark(sources=["harmbench"], dedupe_prompts=True, check_counts=False)
        assert len(ds) == 3
        ds = sgb.load_safety_guard_benchmark(sources=["harmbench"], max_samples_per_source=4, check_counts=False)
        assert len(ds) == 4
        again = sgb.load_safety_guard_benchmark(sources=["harmbench"], max_samples_per_source=4, check_counts=False)
        assert ds["id"] == again["id"]
    with pytest.raises(ValueError):
        sgb.load_safety_guard_benchmark(sources=["nope"])

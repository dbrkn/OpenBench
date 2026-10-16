# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the text safety classification metrics."""

import pytest

from openbench.metric import MetricOptions, MetricRegistry
from openbench.metric.safety_metrics import (
    SafetyAccuracy,
    SafetyCategoryRecall,
    SafetyF1,
    SafetyMCC,
    SafetyPrecision,
    SafetyRecall,
    SafetyROCAUC,
)
from openbench.pipeline_prediction import SafetyLabel, SafetyPrediction
from openbench.types import PipelineType


def ref(unsafe: bool, category: str | None = None) -> SafetyPrediction:
    return SafetyPrediction(label=SafetyLabel.UNSAFE if unsafe else SafetyLabel.SAFE, category=category)


def hyp(unsafe: bool, score: float | None = None) -> SafetyPrediction:
    return SafetyPrediction(label=SafetyLabel.UNSAFE if unsafe else SafetyLabel.SAFE, unsafe_score=score)


# reference, hypothesis: 2 TP, 1 FN, 1 FP, 2 TN
CASES = [
    (ref(True, "violence"), hyp(True, 0.9)),
    (ref(True, "threats"), hyp(True, 0.7)),
    (ref(True, "threats"), hyp(False, 0.4)),
    (ref(False), hyp(True, 0.6)),
    (ref(False), hyp(False, 0.2)),
    (ref(False), hyp(False, 0.1)),
]


def run(metric, cases=CASES):
    per_sample = [metric(reference=r, hypothesis=h, detailed=True) for r, h in cases]
    return abs(metric), per_sample


def test_recall_precision_f1_accuracy_mcc_from_summed_counts():
    assert run(SafetyRecall())[0] == pytest.approx(2 / 3)
    assert run(SafetyPrecision())[0] == pytest.approx(2 / 3)
    assert run(SafetyF1())[0] == pytest.approx(2 / 3)
    assert run(SafetyAccuracy())[0] == pytest.approx(4 / 6)
    expected_mcc = (2 * 2 - 1 * 1) / ((2 + 1) * (2 + 1) * (2 + 1) * (2 + 1)) ** 0.5
    assert run(SafetyMCC())[0] == pytest.approx(expected_mcc)


def test_per_sample_values_are_none_when_undefined():
    _, per_sample = run(SafetyRecall())
    assert [r["recall"] for r in per_sample] == [1.0, 1.0, 0.0, None, None, None]
    _, per_sample = run(SafetyPrecision())
    assert [r["precision"] for r in per_sample] == [1.0, 1.0, None, 0.0, None, None]
    _, per_sample = run(SafetyMCC())
    assert all(r["mcc"] is None for r in per_sample)


def test_confidence_interval_skips_undefined_samples():
    metric = SafetyRecall()
    run(metric)
    center, (lower, upper) = metric.confidence_interval()
    assert center == pytest.approx(2 / 3)
    assert lower <= center <= upper
    empty = SafetyRecall()
    assert empty.confidence_interval() == (None, (None, None))


def test_global_value_is_none_without_positives():
    metric = SafetyRecall()
    metric(reference=ref(False), hypothesis=hyp(False))
    assert abs(metric) is None


def test_roc_auc_ranks_scores_and_resets():
    metric = SafetyROCAUC()
    global_value, per_sample = run(metric)
    # unsafe scores 0.9, 0.7, 0.4 against safe scores 0.6, 0.2, 0.1 -> 8 of 9 pairs ordered correctly
    assert global_value == pytest.approx(8 / 9)
    assert all(r["roc_auc"] is None for r in per_sample)
    assert metric["scored_samples"] == 6
    metric.reset()
    assert abs(metric) is None
    metric(reference=ref(True), hypothesis=hyp(True))
    assert abs(metric) is None  # one class only


def test_roc_auc_falls_back_to_hard_verdicts():
    metric = SafetyROCAUC()
    for r, h in CASES:
        metric(reference=r, hypothesis=hyp(h.is_unsafe))
    assert metric["scored_samples"] == 0
    assert abs(metric) == pytest.approx((2 / 3 + 2 / 3) / 2)  # balanced accuracy


def test_category_recall_macro_average_and_buckets():
    metric = SafetyCategoryRecall()
    cases = CASES + [(ref(True, "made_up"), hyp(False)), (ref(True), hyp(True))]
    global_value, _ = run(metric, cases)
    recalls = SafetyCategoryRecall.recall_per_category(metric.accumulated_)
    assert recalls == {"violence": 1.0, "threats": 0.5, "other": 0.0, "uncategorized": 1.0}
    assert global_value == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)
    assert metric["true_positives__violence"] == 1
    assert metric["false_negatives__threats"] == 1


def test_metrics_are_registered_for_the_task():
    available = set(MetricRegistry.get_available_metrics(PipelineType.SAFETY_CLASSIFICATION))
    assert available == {
        MetricOptions.RECALL,
        MetricOptions.PRECISION,
        MetricOptions.F1,
        MetricOptions.ACCURACY,
        MetricOptions.MCC,
        MetricOptions.ROC_AUC,
        MetricOptions.CATEGORY_RECALL,
    }
    metric = MetricRegistry.get_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.F1)
    assert isinstance(metric, SafetyF1)

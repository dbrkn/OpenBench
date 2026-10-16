# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Binary classification metrics for text safety classification.

Every metric accumulates the confusion counts of one sample at a time. The
decomposable ones (recall, precision, F1, accuracy, MCC) compute the global
value from the summed counts; ROC-AUC keeps the per-sample scores and ranks
them when the global value is requested.
"""

import math
import warnings
from typing import Any

import numpy as np
from pyannote.metrics.base import BaseMetric
from sklearn.metrics import roc_auc_score

from ..pipeline_prediction import SafetyPrediction
from ..types import PipelineType
from .metric import MetricOptions
from .registry import MetricRegistry


TRUE_POSITIVES = "true_positives"
FALSE_POSITIVES = "false_positives"
FALSE_NEGATIVES = "false_negatives"
TRUE_NEGATIVES = "true_negatives"
CONFUSION_COMPONENTS = [TRUE_POSITIVES, FALSE_POSITIVES, FALSE_NEGATIVES, TRUE_NEGATIVES]

# The categories of the safety guard benchmark (NIST AI RMF SAFETY subcategories).
# A reference whose category is not one of these counts under `other`.
NIST_SAFETY_CATEGORIES = [
    "violence",
    "hate_speech",
    "suicide_self_harm",
    "sexual_content",
    "harassment",
    "profanity",
    "threats",
    "health_misinformation",
]
OTHER_CATEGORY = "other"
UNCATEGORIZED = "uncategorized"
CATEGORY_BUCKETS = NIST_SAFETY_CATEGORIES + [OTHER_CATEGORY, UNCATEGORIZED]


def _safe_divide(numerator: float, denominator: float) -> float | None:
    """Return the ratio, or None when it is undefined."""
    if denominator == 0:
        return None
    return numerator / denominator


def confusion_components(reference: SafetyPrediction, hypothesis: SafetyPrediction) -> dict[str, int]:
    """Confusion counts of one sample, with `unsafe` as the positive class."""
    ref_unsafe = reference.is_unsafe
    hyp_unsafe = hypothesis.is_unsafe
    return {
        TRUE_POSITIVES: int(ref_unsafe and hyp_unsafe),
        FALSE_POSITIVES: int(hyp_unsafe and not ref_unsafe),
        FALSE_NEGATIVES: int(ref_unsafe and not hyp_unsafe),
        TRUE_NEGATIVES: int(not ref_unsafe and not hyp_unsafe),
    }


class BaseSafetyMetric(BaseMetric):
    """Shared accumulation of confusion counts.

    `compute_metric` receives either one sample's counts (per-sample value,
    which is None whenever the formula is undefined for that sample) or the
    accumulated counts (the global value).
    """

    def __init__(self, **kwargs: Any):
        super().__init__()

    @classmethod
    def metric_components(cls) -> list[str]:
        return list(CONFUSION_COMPONENTS)

    def compute_components(
        self, reference: SafetyPrediction, hypothesis: SafetyPrediction, **kwargs: Any
    ) -> dict[str, float]:
        return confusion_components(reference, hypothesis)

    def confidence_interval(self, alpha: float = 0.9) -> tuple[float | None, tuple[float | None, float | None]]:
        """Confidence interval of the mean over the samples whose per-sample value is defined."""
        values = [
            result[self.metric_name_] for _, result in self.results_ if result.get(self.metric_name_) is not None
        ]
        if len(values) == 0:
            return None, (None, None)
        if len(values) == 1:
            return values[0], (values[0], values[0])
        import scipy.stats

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            center, (lower, upper) = scipy.stats.bayes_mvs(values, alpha=alpha)[0]
        return float(center), (float(lower), float(upper))


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.RECALL)
class SafetyRecall(BaseSafetyMetric):
    """Share of unsafe texts flagged as unsafe (the primary safety metric)."""

    @classmethod
    def metric_name(cls) -> str:
        return "recall"

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        return _safe_divide(detail[TRUE_POSITIVES], detail[TRUE_POSITIVES] + detail[FALSE_NEGATIVES])


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.PRECISION)
class SafetyPrecision(BaseSafetyMetric):
    """Share of texts flagged as unsafe that are unsafe."""

    @classmethod
    def metric_name(cls) -> str:
        return "precision"

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        return _safe_divide(detail[TRUE_POSITIVES], detail[TRUE_POSITIVES] + detail[FALSE_POSITIVES])


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.F1)
class SafetyF1(BaseSafetyMetric):
    """Harmonic mean of precision and recall on the unsafe class."""

    @classmethod
    def metric_name(cls) -> str:
        return "f1"

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        return _safe_divide(
            2 * detail[TRUE_POSITIVES],
            2 * detail[TRUE_POSITIVES] + detail[FALSE_POSITIVES] + detail[FALSE_NEGATIVES],
        )


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.ACCURACY)
class SafetyAccuracy(BaseSafetyMetric):
    """Share of texts with the correct verdict."""

    @classmethod
    def metric_name(cls) -> str:
        return "accuracy"

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        total = sum(detail[component] for component in CONFUSION_COMPONENTS)
        return _safe_divide(detail[TRUE_POSITIVES] + detail[TRUE_NEGATIVES], total)


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.MCC)
class SafetyMCC(BaseSafetyMetric):
    """Matthews correlation coefficient between reference and predicted verdicts.

    Undefined for one sample; the global value comes from the summed counts.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "mcc"

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        tp, fp, fn, tn = (detail[c] for c in CONFUSION_COMPONENTS)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if denominator == 0:
            return None
        return (tp * tn - fp * fn) / denominator


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.ROC_AUC)
class SafetyROCAUC(BaseSafetyMetric):
    """Area under the ROC curve of the unsafe score.

    Not decomposable per sample: the per-sample value is None and the global
    value ranks every score seen so far. A prediction without a score counts as
    1.0 when unsafe and 0.0 when safe, which reduces the AUC to the balanced
    accuracy of the hard verdicts.
    """

    SCORED_SAMPLES = "scored_samples"

    def __init__(self, **kwargs: Any):
        self._y_true: list[int] = []
        self._y_score: list[float] = []
        super().__init__(**kwargs)

    @classmethod
    def metric_name(cls) -> str:
        return "roc_auc"

    @classmethod
    def metric_components(cls) -> list[str]:
        return list(CONFUSION_COMPONENTS) + [cls.SCORED_SAMPLES]

    def reset(self) -> None:
        super().reset()
        self._y_true = []
        self._y_score = []

    def compute_components(
        self, reference: SafetyPrediction, hypothesis: SafetyPrediction, **kwargs: Any
    ) -> dict[str, float]:
        components = confusion_components(reference, hypothesis)
        scored = hypothesis.unsafe_score is not None
        score = float(hypothesis.unsafe_score) if scored else float(hypothesis.is_unsafe)
        self._y_true.append(int(reference.is_unsafe))
        self._y_score.append(score)
        components[self.SCORED_SAMPLES] = int(scored)
        return components

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        return None

    def __abs__(self) -> float | None:
        if len(set(self._y_true)) < 2:
            return None
        return float(roc_auc_score(np.array(self._y_true), np.array(self._y_score)))

    def confidence_interval(self, alpha: float = 0.9) -> tuple[float | None, tuple[float | None, float | None]]:
        return None, (None, None)


def category_bucket(category: str | None) -> str:
    """Map a reference category onto one of the fixed component buckets."""
    if not category:
        return UNCATEGORIZED
    return category if category in NIST_SAFETY_CATEGORIES else OTHER_CATEGORY


@MetricRegistry.register_metric(PipelineType.SAFETY_CLASSIFICATION, MetricOptions.CATEGORY_RECALL)
class SafetyCategoryRecall(BaseSafetyMetric):
    """Recall per reference category, reported as their macro average.

    The components hold the true positives and false negatives of each
    category, so the per-category recall can be read off the detailed result.
    Safe references contribute nothing; categories with no unsafe reference
    are left out of the average.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "category_recall"

    @classmethod
    def metric_components(cls) -> list[str]:
        components = []
        for bucket in CATEGORY_BUCKETS:
            components.append(f"{TRUE_POSITIVES}__{bucket}")
            components.append(f"{FALSE_NEGATIVES}__{bucket}")
        return components

    def compute_components(
        self, reference: SafetyPrediction, hypothesis: SafetyPrediction, **kwargs: Any
    ) -> dict[str, float]:
        components = dict.fromkeys(self.metric_components(), 0)
        if reference.is_unsafe:
            bucket = category_bucket(reference.category)
            key = TRUE_POSITIVES if hypothesis.is_unsafe else FALSE_NEGATIVES
            components[f"{key}__{bucket}"] = 1
        return components

    @staticmethod
    def recall_per_category(detail: dict[str, float]) -> dict[str, float]:
        """Recall of every category that has at least one unsafe reference."""
        recalls = {}
        for bucket in CATEGORY_BUCKETS:
            tp = detail.get(f"{TRUE_POSITIVES}__{bucket}", 0)
            fn = detail.get(f"{FALSE_NEGATIVES}__{bucket}", 0)
            if tp + fn > 0:
                recalls[bucket] = tp / (tp + fn)
        return recalls

    def compute_metric(self, detail: dict[str, float]) -> float | None:
        recalls = self.recall_per_category(detail)
        if not recalls:
            return None
        return float(np.mean(list(recalls.values())))

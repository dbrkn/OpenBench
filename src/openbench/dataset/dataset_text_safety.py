# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

from typing import Any

import numpy as np
from typing_extensions import TypedDict

from ..pipeline_prediction import SafetyLabel, SafetyPrediction
from .dataset_base import BaseDataset, BaseSample


_UNSAFE_STRINGS = {"unsafe", "un-safe", "harmful", "toxic", "1", "true", "yes"}
_SAFE_STRINGS = {"safe", "harmless", "benign", "non-toxic", "0", "false", "no"}


def parse_safety_label(value: Any) -> SafetyLabel:
    """Normalize a dataset label into a `SafetyLabel`.

    Accepts booleans, 0/1 numbers and the common string spellings.
    """
    if isinstance(value, SafetyLabel):
        return value
    if isinstance(value, bool):
        return SafetyLabel.UNSAFE if value else SafetyLabel.SAFE
    if isinstance(value, (int, float)):
        return SafetyLabel.UNSAFE if value >= 0.5 else SafetyLabel.SAFE
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _UNSAFE_STRINGS:
            return SafetyLabel.UNSAFE
        if normalized in _SAFE_STRINGS:
            return SafetyLabel.SAFE
    raise ValueError(f"Cannot interpret {value!r} as a safety label (expected safe/unsafe, 0/1 or a boolean)")


class TextSafetyExtraInfo(TypedDict, total=False):
    """Extra info for text safety samples."""

    text: str
    source: str
    source_category: str
    confidence: float


class TextSafetyRow(TypedDict):
    """Expected row structure for text safety classification.

    Requires `text` and `label` (safe/unsafe, 0/1 or a boolean). Optional
    columns: `id`, `category` (the benchmark's taxonomy category), `source`
    (the dataset the row came from), `source_category` and `confidence`.
    """

    text: str
    label: str | int | bool


class TextSafetySample(BaseSample[SafetyPrediction, TextSafetyExtraInfo]):
    """Sample for text safety classification: a text and its reference verdict.

    No audio is involved; the waveform is a placeholder so the sample fits the
    shared runner. The text lives in `extra_info` and is exposed as `text`.
    """

    @property
    def text(self) -> str:
        return self.extra_info["text"]

    @property
    def label(self) -> SafetyLabel:
        return self.reference.label

    @property
    def category(self) -> str | None:
        return self.reference.category


class TextSafetyDataset(BaseDataset[TextSafetySample]):
    """Dataset for text safety classification pipelines.

    Expects columns `text` and `label`. A placeholder waveform satisfies the
    base sample structure and is ignored downstream; the runner reports an
    audio duration of zero for this task.
    """

    _expected_columns = ["text", "label"]
    _sample_class = TextSafetySample

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        """Provide a placeholder waveform; text safety has no input audio."""
        audio_name = f"sample_{row['idx']}"
        if row.get("id"):
            audio_name = str(row["id"])
        elif row.get("audio_name"):
            audio_name = str(row["audio_name"])
        dummy_waveform = np.zeros(1, dtype=np.float32)
        dummy_sample_rate = 16000
        return audio_name, dummy_waveform, dummy_sample_rate

    def prepare_sample(self, row: TextSafetyRow) -> tuple[SafetyPrediction, TextSafetyExtraInfo]:
        """Build the reference verdict and carry the text in extra_info."""
        category = row.get("category")
        reference = SafetyPrediction(
            label=parse_safety_label(row["label"]),
            category=str(category) if category else None,
        )

        extra_info: TextSafetyExtraInfo = {"text": str(row["text"])}
        if row.get("source"):
            extra_info["source"] = str(row["source"])
        if row.get("source_category"):
            extra_info["source_category"] = str(row["source_category"])
        if row.get("confidence") is not None:
            extra_info["confidence"] = float(row["confidence"])

        return reference, extra_info

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

from typing import Any

import numpy as np
from datasets import Dataset as HfDataset
from typing_extensions import TypedDict

from ..pipeline_prediction import SafetyLabel, SafetyPrediction
from .dataset_base import BaseDataset, BaseSample


_UNSAFE_STRINGS = {"unsafe", "un-safe", "harmful", "toxic", "hate", "1", "true", "yes"}
_SAFE_STRINGS = {"safe", "harmless", "benign", "non-toxic", "non-hate", "0", "false", "no"}


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


class SafetyExtraInfo(TypedDict, total=False):
    """Extra info for safety classification samples."""

    text: str
    has_text_column: bool
    has_audio: bool
    source: str
    source_category: str
    confidence: float
    language: str


class SafetyRow(TypedDict, total=False):
    """Expected row structure for safety classification.

    Requires `label` (safe/unsafe, 0/1 or a boolean) and at least one of
    `text` (the utterance as text) or `audio` (the utterance as speech). Text
    pipelines read `text`, audio pipelines read `audio`; a row may carry both,
    for example speech with its transcript. Optional columns: `id`, `category`
    (the benchmark's taxonomy category), `source`, `source_category`,
    `confidence`, `language`.
    """

    text: str
    audio: dict
    label: str | int | bool


class SafetySample(BaseSample[SafetyPrediction, SafetyExtraInfo]):
    """One utterance, as text and/or speech, with its reference verdict.

    `waveform` holds the speech when the dataset has it; otherwise a
    one-sample placeholder keeps the shared runner happy. `text` raises when
    the dataset has no text for the sample, so text pipelines fail loudly on
    audio-only datasets instead of scoring an empty string.
    """

    @property
    def has_text(self) -> bool:
        return bool(self.extra_info.get("text"))

    @property
    def has_audio(self) -> bool:
        return bool(self.extra_info.get("has_audio", self.waveform.size > 1))

    @property
    def has_text_column(self) -> bool:
        return bool(self.extra_info.get("has_text_column", self.has_text))

    @property
    def text(self) -> str:
        """The text, which may be empty when a transcription heard nothing; raises on audio-only datasets."""
        if not self.has_text_column:
            raise ValueError(
                f"Sample {self.audio_name} has no text; this dataset is audio only. "
                "Load it with an `asr` option to transcribe it, or run an audio pipeline."
            )
        return self.extra_info.get("text") or ""

    @property
    def label(self) -> SafetyLabel:
        return self.reference.label

    @property
    def category(self) -> str | None:
        return self.reference.category


class SafetyDataset(BaseDataset[SafetySample]):
    """Dataset for safety classification pipelines.

    Expects a `label` column plus `text`, `audio` or both. Audio is decoded to
    a float waveform at its stored sample rate; without audio the waveform is a
    placeholder and the runner reports an audio duration of zero.
    """

    _expected_columns = ["label"]
    _sample_class = SafetySample

    def __init__(self, ds: HfDataset):
        super().__init__(ds)
        if "text" not in ds.column_names and "audio" not in ds.column_names:
            raise ValueError("A safety dataset needs a `text` column, an `audio` column or both")

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        audio_name = f"sample_{row['idx']}"
        if row.get("id"):
            audio_name = str(row["id"])
        elif row.get("audio_name"):
            audio_name = str(row["audio_name"])
        audio = row.get("audio")
        if audio and audio.get("array") is not None:
            waveform = np.asarray(audio["array"], dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=-1)
            return audio_name, waveform, int(audio["sampling_rate"])
        return audio_name, np.zeros(1, dtype=np.float32), 16000

    def prepare_sample(self, row: SafetyRow) -> tuple[SafetyPrediction, SafetyExtraInfo]:
        """Build the reference verdict and carry the text and provenance in extra_info."""
        category = row.get("category")
        reference = SafetyPrediction(
            label=parse_safety_label(row["label"]),
            category=str(category) if category else None,
        )

        audio = row.get("audio")
        extra_info: SafetyExtraInfo = {
            "text": str(row.get("text") or ""),
            "has_text_column": "text" in row,
            "has_audio": bool(audio and audio.get("array") is not None),
        }
        for key in ("source", "source_category", "language"):
            if row.get(key):
                extra_info[key] = str(row[key])
        if row.get("confidence") is not None:
            extra_info["confidence"] = float(row["confidence"])

        return reference, extra_info

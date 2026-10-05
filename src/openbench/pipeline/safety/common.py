# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Shared pieces of the text safety classification pipelines."""

import json
import re
from typing import Literal

import numpy as np
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import PipelineConfig, PipelineOutput


logger = get_logger(__name__)

GuardOutputStyle = Literal["auto", "safe_unsafe", "yes_no", "wildguard", "qwen3guard", "json"]


class SafetyConfig(PipelineConfig):
    """Base config for text safety classification pipelines."""

    threshold: float = Field(
        0.5,
        description="Unsafe score at or above which a text counts as unsafe, for pipelines that emit a score.",
    )


class SafetyInput(BaseModel):
    """What a safety pipeline receives for one sample: the text, the speech, or both."""

    audio_name: str
    text: str | None = None
    has_text_column: bool = True
    waveform: np.ndarray | None = None
    sample_rate: int | None = None

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_sample(cls, sample: "SafetySample") -> "SafetyInput":
        return cls(
            audio_name=sample.audio_name,
            text=sample.extra_info.get("text") or None,
            has_text_column=sample.has_text_column,
            waveform=sample.waveform if sample.has_audio else None,
            sample_rate=sample.sample_rate if sample.has_audio else None,
        )

    def require_text(self) -> str:
        """The text to classify; empty when a transcription heard nothing, an error on audio-only datasets."""
        if not self.has_text_column:
            raise ValueError(f"Sample {self.audio_name} has no text; transcribe the dataset or run an audio pipeline")
        return self.text or ""

    def require_audio(self) -> tuple[np.ndarray, int]:
        if self.waveform is None or self.sample_rate is None:
            raise ValueError(f"Sample {self.audio_name} has no audio; this pipeline needs speech")
        return self.waveform, self.sample_rate


EMPTY_TEXT_CATEGORY = "empty_text"


def empty_text_prediction() -> SafetyPrediction:
    """The verdict for a sample whose transcript is empty: nothing was said, so nothing is flagged."""
    return SafetyPrediction(label=SafetyLabel.SAFE, unsafe_score=0.0, category=EMPTY_TEXT_CATEGORY, raw_output="")


class SafetyOutput(PipelineOutput[SafetyPrediction]):
    pass


_SAFETY_KEYS = ("user safety", "response safety", "label", "verdict", "safety", "result", "classification")
_CATEGORY_KEYS = ("safety categories", "categories", "category", "violated_categories")
_LLAMA_GUARD_CODES = re.compile(r"\bS\d{1,2}\b")


def _label_from_word(word: str) -> SafetyLabel | None:
    normalized = word.strip().strip(".,;:!\"'`*").lower()
    if normalized in ("unsafe", "un-safe", "harmful", "controversial", "yes", "violation", "violates"):
        return SafetyLabel.UNSAFE
    if normalized in ("safe", "harmless", "no", "compliant"):
        return SafetyLabel.SAFE
    return None


def _parse_json(output: str) -> tuple[SafetyLabel, str | None] | None:
    start, end = output.find("{"), output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    lowered = {str(k).lower(): v for k, v in payload.items()}
    label = None
    for key in _SAFETY_KEYS:
        if key in lowered:
            value = lowered[key]
            label = _label_from_word(str(value))
            if label is None and isinstance(value, bool):
                label = SafetyLabel.UNSAFE if value else SafetyLabel.SAFE
            if label is not None:
                break
    if label is None:
        return None
    category = None
    for key in _CATEGORY_KEYS:
        if key in lowered and lowered[key]:
            value = lowered[key]
            category = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
            break
    return label, category


def _parse_wildguard(output: str) -> tuple[SafetyLabel, str | None] | None:
    match = re.search(r"harmful request:\s*(yes|no)", output, re.IGNORECASE)
    if match is None:
        return None
    return _label_from_word(match.group(1)), None


def _parse_qwen3guard(output: str) -> tuple[SafetyLabel, str | None] | None:
    match = re.search(r"safety:\s*(safe|unsafe|controversial)", output, re.IGNORECASE)
    if match is None:
        return None
    categories = re.search(r"categories:\s*(.+)", output, re.IGNORECASE)
    category = categories.group(1).strip() if categories else None
    return _label_from_word(match.group(1)), category


def _parse_safe_unsafe(output: str) -> tuple[SafetyLabel, str | None] | None:
    if re.search(r"\bunsafe\b", output, re.IGNORECASE):
        codes = _LLAMA_GUARD_CODES.findall(output)
        return SafetyLabel.UNSAFE, ",".join(codes) if codes else None
    if re.search(r"\bsafe\b", output, re.IGNORECASE):
        return SafetyLabel.SAFE, None
    return None


def _parse_yes_no(output: str) -> tuple[SafetyLabel, str | None] | None:
    match = re.match(r"\s*\**(yes|no)\b", output, re.IGNORECASE)
    if match is None:
        return None
    return _label_from_word(match.group(1)), None


_PARSERS = {
    "json": _parse_json,
    "wildguard": _parse_wildguard,
    "qwen3guard": _parse_qwen3guard,
    "safe_unsafe": _parse_safe_unsafe,
    "yes_no": _parse_yes_no,
}


def parse_guard_verdict(output: str, style: GuardOutputStyle = "auto") -> tuple[SafetyLabel, str | None]:
    """Turn a guard model's generated text into a verdict and an optional category.

    Styles: `json` (a JSON object with a label key, e.g. Nemotron Safety Guard or
    GPT-OSS Safeguard), `wildguard` ("Harmful request: yes/no"), `qwen3guard`
    ("Safety: Safe/Unsafe/Controversial", controversial counts as unsafe),
    `safe_unsafe` (the word unsafe or safe, with Llama Guard's S-codes as the
    category) and `yes_no` (ShieldGemma: Yes means a violation). `auto` tries
    them in that order. An output no style can read is logged and counts as
    safe, with `unparsed` as its category, which matches a guard that failed to
    flag the text.
    """
    styles = list(_PARSERS) if style == "auto" else [style]
    for name in styles:
        parsed = _PARSERS[name](output)
        if parsed is not None and parsed[0] is not None:
            return parsed
    logger.warning(f"Could not parse a safety verdict from: {output!r}")
    return SafetyLabel.SAFE, "unparsed"


def resolve_device(device: str | None) -> str:
    """Pick the torch device: the requested one, else CUDA, else Apple MPS, else CPU."""
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

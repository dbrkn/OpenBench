# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Roblox voice safety classifier v3 as Sova ships it: the W16A16 Core ML conversion from axon, on a Mac."""

import math
from typing import Callable, Literal

import numpy as np
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyConfig, SafetyInput, SafetyOutput
from .coreml_common import load_compiled_model, sova_bundle, threshold_shifted_probability


VOICE_HEADS = (
    "ABUSE_TYPE_PRIVACY_ASKING_FOR_PII",
    "ABUSE_TYPE_DISCRIMINATORY",
    "ABUSE_TYPE_HARASSMENT",
    "ABUSE_TYPE_SEXUAL_CONTENT",
    "ABUSE_TYPE_ILLEGAL_AND_REGULATED_CONTENT",
    "ABUSE_TYPE_DATING_AND_ROMANTIC_CONTENT",
    "ABUSE_TYPE_PROFANITY",
    "ABUSE_TYPE_DISRUPTIVE_AUDIO",
)
LANGUAGES = "ar bg cs da de el en es fi fr hr hu id it ja ko nl no pl pt ro ru sk sv th tl tr uk zh other".split()
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 240_000  # 15 s, the model's intended segment and Sova's window
HOP = 160
FRAMES = 1500
# Sova's VoiceSafetyPolicy: the heads it reads and their T0 bars
SOVA_HEADS = {
    "ABUSE_TYPE_PRIVACY_ASKING_FOR_PII": 0.5,
    "ABUSE_TYPE_DATING_AND_ROMANTIC_CONTENT": 0.5,
    "ABUSE_TYPE_SEXUAL_CONTENT": 0.5,
    "ABUSE_TYPE_HARASSMENT": 0.6,
}


def prepare_window(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sova's cut: keep the last 15 s, left-align, zero-pad, mask the frames that hold audio."""
    clip = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if clip.size > WINDOW_SAMPLES:
        clip = clip[-WINDOW_SAMPLES:]
    audio = np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)
    audio[0, : clip.size] = clip
    # An empty or sub-frame clip still gets one masked frame, so the graph sees a valid window
    valid = min(FRAMES, max(1, math.ceil(clip.size / HOP)))
    frame_mask = np.zeros((1, FRAMES), dtype=np.float32)
    frame_mask[0, :valid] = 1.0
    return audio, frame_mask


def resample_to_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == SAMPLE_RATE:
        return np.asarray(waveform, dtype=np.float32)
    import librosa

    return librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sample_rate, target_sr=SAMPLE_RATE)


class CoreMLRobloxVoiceSafetyConfig(SafetyConfig):
    model_path: str = Field(
        default_factory=lambda: str(sova_bundle("roblox_voice_safety/AudioClassifier.mlmodelc")),
        description="Compiled Core ML audio classifier (Sova's bundle by default).",
    )
    heads: list[str] = Field(
        default_factory=lambda: list(VOICE_HEADS),
        description="Heads that count towards the verdict; any of them firing makes the clip unsafe.",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Per-head thresholds; heads not listed use `threshold` (0.5 by default).",
    )
    compute_units: Literal["all", "cpu_and_ne", "cpu_and_gpu", "cpu"] = Field("cpu_and_ne")


@register_pipeline
class CoreMLRobloxVoiceSafetyPipeline(Pipeline):
    """Scores one clip through the 8-head voice safety classifier.

    The clip is unsafe when any selected head reaches its threshold. The unsafe
    score is the maximum over the heads after shifting each head so its own
    threshold maps to 0.5; the category is the head that fired highest. The
    raw output lists every head and the detected language.
    """

    _config_class = CoreMLRobloxVoiceSafetyConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        config = self.config
        unknown = set(config.heads) - set(VOICE_HEADS)
        if unknown:
            raise ValueError(f"Unknown heads {sorted(unknown)}; the model has {VOICE_HEADS}")
        model = load_compiled_model(config.model_path, config.compute_units)
        thresholds = {h: float(config.thresholds.get(h, config.threshold)) for h in config.heads}
        head_indices = {h: VOICE_HEADS.index(h) for h in config.heads}

        def classify(sample: SafetyInput) -> SafetyPrediction:
            waveform, sample_rate = sample.require_audio()
            audio, frame_mask = prepare_window(resample_to_16k(waveform, sample_rate))
            outputs = model.predict({"audio": audio, "frame_mask": frame_mask})
            probabilities = np.asarray(outputs["probabilities"], dtype=np.float32).reshape(-1)
            language_probabilities = np.asarray(outputs["language_probabilities"], dtype=np.float32).reshape(-1)
            shifted = {
                h: threshold_shifted_probability(probabilities[i], thresholds[h]) for h, i in head_indices.items()
            }
            top_head = max(shifted, key=shifted.get)
            unsafe_score = shifted[top_head]
            is_unsafe = unsafe_score >= 0.5
            language = LANGUAGES[int(language_probabilities.argmax())]
            raw = ", ".join(
                f"{h.removeprefix('ABUSE_TYPE_').lower()}={probabilities[i]:.4f}" for i, h in enumerate(VOICE_HEADS)
            )
            return SafetyPrediction(
                label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
                unsafe_score=unsafe_score,
                category=top_head.removeprefix("ABUSE_TYPE_").lower() if is_unsafe else None,
                raw_output=f"{raw}, language={language}",
            )

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

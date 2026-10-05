# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Roblox PII classifier v2 as Sova ships it: the W8A16 Core ML conversion from axon, on a Mac."""

from typing import Callable, Literal

import numpy as np
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyConfig, SafetyInput, SafetyOutput, empty_text_prediction
from .coreml_common import load_compiled_model, sova_bundle, threshold_shifted_probability


# Byte-exact with the checkpoint's `format_conversation` and Sova's ConversationFormatter
INSTRUCTION_PREFIX = (
    "Instruct: In the following chat messages from target speaker t and possibly other speakers s1, s2, etc., "
    "detect abuse by speaker t.\nQuery:"
)
PII_HEADS = ("privacy_asking_for_pii", "privacy_giving_pii", "directing_users_off_platform")
# The model card's recommended thresholds, in head order
CARD_THRESHOLDS = {"privacy_asking_for_pii": 0.60, "privacy_giving_pii": 0.55, "directing_users_off_platform": 0.10}
SEQUENCE_LENGTH = 512


def format_conversation(turns: list[tuple[str, str]]) -> str:
    """Render (speaker, text) turns the way the classifier was trained: `t` is the target, others `s1`, `s2`..."""
    names: dict[str, str] = {}
    rendered = []
    for speaker, text in turns:
        if speaker == "t":
            label = "t"
        else:
            label = names.setdefault(speaker, f"s{len(names) + 1}")
        rendered.append(f"{label}: {text}")
    return INSTRUCTION_PREFIX + "\n\n" + " </s> ".join(rendered)


class CoreMLRobloxPIIConfig(SafetyConfig):
    model_path: str = Field(
        default_factory=lambda: str(sova_bundle("roblox_pii/TextClassifier.mlmodelc")),
        description="Compiled Core ML text classifier (Sova's bundle by default).",
    )
    tokenizer_path: str = Field(
        default_factory=lambda: str(sova_bundle("roblox_pii/tokenizer")),
        description="XLM-RoBERTa tokenizer directory (Sova's bundle by default).",
    )
    heads: list[str] = Field(
        default_factory=lambda: list(PII_HEADS),
        description="Heads that count towards the verdict; any of them firing makes the text unsafe.",
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: dict(CARD_THRESHOLDS), description="Per-head thresholds (model card by default)."
    )
    compute_units: Literal["all", "cpu_and_ne", "cpu_and_gpu", "cpu"] = Field("cpu_and_ne")


@register_pipeline
class CoreMLRobloxPIIPipeline(Pipeline):
    """Scores one text as the target speaker `t` of a one-turn chat.

    The text is unsafe when any selected head reaches its threshold. The unsafe
    score is the maximum over the heads after shifting each head so its own
    threshold maps to 0.5; the category is the head that fired highest.
    """

    _config_class = CoreMLRobloxPIIConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        from transformers import AutoTokenizer

        config = self.config
        unknown = set(config.heads) - set(PII_HEADS)
        if unknown:
            raise ValueError(f"Unknown heads {sorted(unknown)}; the model has {PII_HEADS}")
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, truncation_side="left")
        model = load_compiled_model(config.model_path, config.compute_units)
        head_indices = [PII_HEADS.index(h) for h in config.heads]

        def classify(sample: SafetyInput) -> SafetyPrediction:
            text = sample.require_text()
            if not text.strip():
                return empty_text_prediction()
            text = format_conversation([("t", text)])
            encoded = tokenizer(
                text, padding="max_length", max_length=SEQUENCE_LENGTH, truncation=True, return_tensors="np"
            )
            outputs = model.predict(
                {
                    "input_ids": encoded["input_ids"].astype(np.int32),
                    "attention_mask": encoded["attention_mask"].astype(np.float32),
                }
            )
            probabilities = np.asarray(outputs["probabilities"], dtype=np.float32).reshape(-1)
            shifted = {
                PII_HEADS[i]: threshold_shifted_probability(probabilities[i], config.thresholds[PII_HEADS[i]])
                for i in head_indices
            }
            top_head = max(shifted, key=shifted.get)
            unsafe_score = shifted[top_head]
            is_unsafe = unsafe_score >= 0.5
            return SafetyPrediction(
                label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
                unsafe_score=unsafe_score,
                category=top_head if is_unsafe else None,
                raw_output=", ".join(f"{h}={probabilities[i]:.4f}" for i, h in enumerate(PII_HEADS)),
            )

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

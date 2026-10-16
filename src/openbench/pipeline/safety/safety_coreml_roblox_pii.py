# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Roblox PII classifier v2 as Sova ships it: the W8A16 Core ML conversion from axon, on a Mac."""

from typing import Callable, Literal

import numpy as np
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, empty_text_prediction
from .coreml_common import load_compiled_model, sova_bundle
from .roblox_pii_common import (
    CARD_THRESHOLDS,
    INSTRUCTION_PREFIX,
    PII_HEADS,
    SEQUENCE_LENGTH,
    RobloxPIIConfig,
    format_conversation,
    render_input,
    validate_heads,
    verdict,
)


__all__ = [
    "CARD_THRESHOLDS",
    "INSTRUCTION_PREFIX",
    "PII_HEADS",
    "CoreMLRobloxPIIConfig",
    "CoreMLRobloxPIIPipeline",
    "format_conversation",
]


class CoreMLRobloxPIIConfig(RobloxPIIConfig):
    model_path: str = Field(
        default_factory=lambda: str(sova_bundle("roblox_pii/TextClassifier.mlmodelc")),
        description="Compiled Core ML text classifier (Sova's bundle by default).",
    )
    tokenizer_path: str = Field(
        default_factory=lambda: str(sova_bundle("roblox_pii/tokenizer")),
        description="XLM-RoBERTa tokenizer directory (Sova's bundle by default).",
    )
    compute_units: Literal["all", "cpu_and_ne", "cpu_and_gpu", "cpu"] = Field("cpu_and_ne")


@register_pipeline
class CoreMLRobloxPIIPipeline(Pipeline):
    """Scores one turn as the target speaker `t`, after its preceding turns, with the Core ML bundle."""

    _config_class = CoreMLRobloxPIIConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        from transformers import AutoTokenizer

        config = self.config
        head_indices = validate_heads(config)
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path, truncation_side="left")
        model = load_compiled_model(config.model_path, config.compute_units)

        def classify(sample: SafetyInput) -> SafetyPrediction:
            text = render_input(sample, config.context_turns)
            if text is None:
                return empty_text_prediction()
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
            return verdict(probabilities, config, head_indices)

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

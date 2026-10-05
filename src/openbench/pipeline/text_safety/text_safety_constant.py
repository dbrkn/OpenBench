# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""A pipeline that returns the same verdict for every text: the floor of the benchmark."""

from typing import Callable

from pydantic import Field

from ...dataset.dataset_text_safety import TextSafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TextSafetyConfig, TextSafetyInput, TextSafetyOutput


class ConstantTextSafetyConfig(TextSafetyConfig):
    label: SafetyLabel = Field(SafetyLabel.UNSAFE, description="The verdict returned for every text.")


@register_pipeline
class ConstantTextSafetyPipeline(Pipeline):
    """Flags every text with the configured label.

    `unsafe` gives recall 1 and the precision of the dataset's unsafe share;
    any real classifier should beat its F1.
    """

    _config_class = ConstantTextSafetyConfig
    pipeline_type = PipelineType.TEXT_SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[TextSafetyInput], SafetyPrediction]:
        label = self.config.label
        score = 1.0 if label == SafetyLabel.UNSAFE else 0.0

        def classify(_: TextSafetyInput) -> SafetyPrediction:
            return SafetyPrediction(label=label, unsafe_score=score)

        return classify

    def parse_input(self, input_sample: TextSafetySample) -> TextSafetyInput:
        return TextSafetyInput(text=input_sample.text, audio_name=input_sample.audio_name)

    def parse_output(self, output: SafetyPrediction) -> TextSafetyOutput:
        return TextSafetyOutput(prediction=output)

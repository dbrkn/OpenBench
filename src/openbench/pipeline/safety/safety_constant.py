# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""A pipeline that returns the same verdict for every text: the floor of the benchmark."""

from typing import Callable

from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyConfig, SafetyInput, SafetyOutput


class ConstantSafetyConfig(SafetyConfig):
    label: SafetyLabel = Field(SafetyLabel.UNSAFE, description="The verdict returned for every text.")


@register_pipeline
class ConstantSafetyPipeline(Pipeline):
    """Flags every text with the configured label.

    `unsafe` gives recall 1 and the precision of the dataset's unsafe share;
    any real classifier should beat its F1.
    """

    _config_class = ConstantSafetyConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        label = self.config.label
        score = 1.0 if label == SafetyLabel.UNSAFE else 0.0

        def classify(_: SafetyInput) -> SafetyPrediction:
            return SafetyPrediction(label=label, unsafe_score=score)

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

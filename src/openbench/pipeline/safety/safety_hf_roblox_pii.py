# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Roblox PII classifier v2 from the Hugging Face Hub (PyTorch), for GPU machines."""

from typing import Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, dtype_kwargs, empty_text_prediction, resolve_device
from .roblox_pii_common import (
    HUB_MODEL_ID,
    SEQUENCE_LENGTH,
    RobloxPIIConfig,
    render_input,
    validate_heads,
    verdict,
)


logger = get_logger(__name__)


class HuggingFaceRobloxPIIConfig(RobloxPIIConfig):
    model_id: str = Field(HUB_MODEL_ID, description="Hub id or local directory of the checkpoint.")
    device: str | None = Field(None, description="torch device; defaults to CUDA, then MPS, then CPU.")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "float32", description="Weights dtype; float16 roughly halves the GPU time with probabilities within 1e-3."
    )


@register_pipeline
class HuggingFaceRobloxPIIPipeline(Pipeline):
    """Scores one turn as the target speaker `t`, after its preceding turns, with the PyTorch checkpoint.

    Same input format, truncation (left, 512 tokens) and head reading as the
    Core ML pipeline, so the two runtimes are comparable sample by sample.
    """

    _config_class = HuggingFaceRobloxPIIConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = self.config
        head_indices = validate_heads(config)
        device = resolve_device(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, truncation_side="left")
        model = AutoModelForSequenceClassification.from_pretrained(config.model_id, **dtype_kwargs(config.dtype))
        model.to(device).eval()
        logger.info(f"{config.model_id} on {device} ({config.dtype}), heads {config.heads}")

        def classify(sample: SafetyInput) -> SafetyPrediction:
            text = render_input(sample, config.context_turns)
            if text is None:
                return empty_text_prediction()
            encoded = tokenizer(text, truncation=True, max_length=SEQUENCE_LENGTH, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**encoded).logits[0].float()
            probabilities = torch.sigmoid(logits).cpu().numpy()
            return verdict(probabilities, config, head_indices)

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

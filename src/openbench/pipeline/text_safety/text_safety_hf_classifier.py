# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Encoder-style text classifiers from the Hugging Face Hub (sequence classification heads)."""

import re
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_text_safety import TextSafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import TextSafetyConfig, TextSafetyInput, TextSafetyOutput, resolve_device


logger = get_logger(__name__)

_UNSAFE_LABEL_PATTERN = re.compile(
    r"(unsafe|un-safe|toxic|hate|harm|offensive|impolite|abus|violat|threat|insult|sexual|profan|self[-_ ]?harm)",
    re.IGNORECASE,
)


class HuggingFaceTextClassifierConfig(TextSafetyConfig):
    model_id: str = Field(..., description="Hub id of a model with a sequence classification head.")
    unsafe_labels: list[str] | None = Field(
        None,
        description=(
            "Names (from the model's id2label) of the labels that mean unsafe. When unset, labels whose name "
            "reads as unsafe are picked, or every label for a multi-label model, or LABEL_1 for a binary one."
        ),
    )
    multi_label: bool = Field(
        False,
        description="Treat the head as independent sigmoids (one per label) instead of a softmax over labels.",
    )
    max_length: int = Field(512, description="Token budget; longer texts are truncated.")
    device: str | None = Field(None, description="torch device; defaults to CUDA, then MPS, then CPU.")
    trust_remote_code: bool = Field(False, description="Allow the model's own code to run.")


def resolve_unsafe_label_ids(
    id2label: dict[int, str], unsafe_labels: list[str] | None, multi_label: bool
) -> list[int]:
    """Indices of the labels that count as unsafe."""
    if unsafe_labels is not None:
        wanted = {label.lower() for label in unsafe_labels}
        ids = [i for i, name in id2label.items() if name.lower() in wanted]
        missing = wanted - {id2label[i].lower() for i in ids}
        if missing:
            raise ValueError(f"unsafe_labels {sorted(missing)} not in the model's labels {list(id2label.values())}")
        return ids
    ids = [i for i, name in id2label.items() if _UNSAFE_LABEL_PATTERN.search(name)]
    if ids:
        return ids
    if multi_label:
        return list(id2label)
    if len(id2label) == 2 and 1 in id2label:
        logger.warning(f"Assuming label {id2label[1]!r} means unsafe; set unsafe_labels to override")
        return [1]
    raise ValueError(f"Cannot tell which of {list(id2label.values())} mean unsafe; set unsafe_labels")


@register_pipeline
class HuggingFaceTextClassifierPipeline(Pipeline):
    """Runs a `AutoModelForSequenceClassification` model and reads the unsafe probability off its head.

    Softmax heads: the unsafe score is the summed probability of the unsafe
    labels. Multi-label heads: the highest sigmoid among the unsafe labels.
    The text is unsafe when the score reaches `threshold`; the category is the
    most probable unsafe label.
    """

    _config_class = HuggingFaceTextClassifierConfig
    pipeline_type = PipelineType.TEXT_SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[TextSafetyInput], SafetyPrediction]:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        config = self.config
        device = resolve_device(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id, trust_remote_code=config.trust_remote_code
        )
        model.to(device).eval()
        id2label = {int(i): str(name) for i, name in model.config.id2label.items()}
        unsafe_ids = resolve_unsafe_label_ids(id2label, config.unsafe_labels, config.multi_label)
        logger.info(f"{config.model_id}: unsafe labels {[id2label[i] for i in unsafe_ids]} on {device}")

        def classify(sample: TextSafetyInput) -> SafetyPrediction:
            encoded = tokenizer(
                sample.text, return_tensors="pt", truncation=True, max_length=config.max_length, padding=True
            ).to(device)
            with torch.no_grad():
                logits = model(**encoded).logits[0].float().cpu()
            if config.multi_label:
                probabilities = torch.sigmoid(logits)
                unsafe_score = float(probabilities[unsafe_ids].max())
            else:
                probabilities = torch.softmax(logits, dim=-1)
                unsafe_score = float(probabilities[unsafe_ids].sum())
            is_unsafe = unsafe_score >= config.threshold
            top_unsafe = max(unsafe_ids, key=lambda i: float(probabilities[i]))
            return SafetyPrediction(
                label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
                unsafe_score=unsafe_score,
                category=id2label[top_unsafe] if is_unsafe else None,
                raw_output=", ".join(f"{id2label[i]}={float(probabilities[i]):.4f}" for i in sorted(id2label)),
            )

        return classify

    def parse_input(self, input_sample: TextSafetySample) -> TextSafetyInput:
        return TextSafetyInput(text=input_sample.text, audio_name=input_sample.audio_name)

    def parse_output(self, output: SafetyPrediction) -> TextSafetyOutput:
        return TextSafetyOutput(prediction=output)

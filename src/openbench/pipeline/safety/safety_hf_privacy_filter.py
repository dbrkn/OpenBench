# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""OpenAI Privacy Filter from the Hugging Face Hub, read as a per-turn PII verdict.

The model tags tokens (BIOES over eight PII types). A turn counts as PII when
any of its tokens is more likely PII than `O`; the turn's score is the highest
per-token probability of not being `O`, so the ROC curve sweeps that bar. This
is the plain per-token reading of the `transformers` head, not OpenAI's
constrained Viterbi decoder with its operating points. `raw_output` is JSON
with the detected spans as [start, end, type] in the sample's text (its target
turns joined by newlines), for character-level scoring.
"""

import json
from typing import Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, dtype_kwargs, empty_text_prediction
from .span_common import SpanPipelineConfig, entity_type, overlaps, render_window, to_target_offsets


__all__ = ["HuggingFacePrivacyFilterConfig", "HuggingFacePrivacyFilterPipeline", "render_window", "spans_from_tokens"]


logger = get_logger(__name__)

HUB_MODEL_ID = "openai/privacy-filter"
MIN_TRANSFORMERS = (5, 6)


def spans_from_tokens(
    offsets: list[tuple[int, int]], labels: list[str], ranges: list[tuple[int, int]]
) -> list[tuple[int, int, str]]:
    """Contiguous runs of same-type PII tokens inside the target turns, as (start, end, type) in the target text."""
    spans: list[tuple[int, int, str]] = []
    current: tuple[int, int, int, str] | None = None  # range index, start, end, type
    for offset, label in zip(offsets, labels):
        mapped = to_target_offsets(offset, ranges) if label != "O" else None
        kind = entity_type(label) if mapped else None
        if (
            current is not None
            and kind == current[3]
            and mapped[0] == current[0]
            and not label.startswith(("B-", "S-"))
        ):
            current = (current[0], current[1], mapped[2], kind)
            continue
        if current is not None:
            spans.append((current[1], current[2], current[3]))
        current = (mapped[0], mapped[1], mapped[2], kind) if kind else None
    if current is not None:
        spans.append((current[1], current[2], current[3]))
    return spans


class HuggingFacePrivacyFilterConfig(SpanPipelineConfig):
    model_id: str = Field(HUB_MODEL_ID, description="Hub id or local directory of the checkpoint.")
    device: str | None = Field(
        None,
        description=(
            "torch device; defaults to CUDA, else CPU. Not MPS: its mixture-of-experts layers ran 16x slower there "
            "than on the CPU (1.0-1.3 s vs 75 ms per segment on an M-series Mac)."
        ),
    )
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(
        "auto", description="Weights dtype; `auto` is bf16 (as shipped) on CUDA and float32 on the CPU."
    )
    max_tokens: int = Field(8192, description="Token budget; longer inputs keep their end (the target turn).")


@register_pipeline
class HuggingFacePrivacyFilterPipeline(Pipeline):
    _config_class = HuggingFacePrivacyFilterConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        import torch
        import transformers
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version < MIN_TRANSFORMERS:
            raise RuntimeError(
                f"OpenAI Privacy Filter needs transformers >= 5.6 (found {transformers.__version__}); "
                "see docs/pii-benchmark.md for the PII environment."
            )
        config = self.config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = config.dtype if config.dtype != "auto" else ("bfloat16" if device.startswith("cuda") else "float32")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, truncation_side="left")
        model = AutoModelForTokenClassification.from_pretrained(config.model_id, **dtype_kwargs(dtype))
        model.to(device).eval()
        id2label = {int(i): str(name) for i, name in model.config.id2label.items()}
        outside_id = next(i for i, name in id2label.items() if name == "O")
        logger.info(f"{config.model_id} on {device} ({dtype}), context turns {config.context_turns}")

        def classify(sample: SafetyInput) -> SafetyPrediction:
            if not sample.require_text().strip():
                return empty_text_prediction()
            text, ranges = render_window(sample.window(config.context_turns, config.context_speakers))
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=config.max_tokens,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
            with torch.inference_mode():
                logits = model(**encoded.to(device)).logits[0].float()
            probabilities = torch.softmax(logits, dim=-1).cpu()
            predicted = probabilities.argmax(dim=-1).tolist()
            not_outside = (1.0 - probabilities[:, outside_id]).tolist()

            in_target = [i for i, offset in enumerate(offsets) if overlaps(offset, ranges)]
            if not in_target:
                return SafetyPrediction(label=SafetyLabel.SAFE, unsafe_score=0.0, raw_output=json.dumps({"spans": []}))
            top = max(in_target, key=lambda i: not_outside[i])
            unsafe_score = float(not_outside[top])
            is_unsafe = unsafe_score >= config.threshold
            top_type = probabilities[top].clone()
            top_type[outside_id] = -1.0
            spans = spans_from_tokens(offsets, [id2label[i] for i in predicted], ranges)
            return SafetyPrediction(
                label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
                unsafe_score=unsafe_score,
                category=entity_type(id2label[int(top_type.argmax())]) if is_unsafe else None,
                raw_output=json.dumps({"spans": spans}),
            )

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

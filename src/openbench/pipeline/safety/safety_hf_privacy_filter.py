# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""OpenAI Privacy Filter from the Hugging Face Hub, read as a per-turn PII verdict.

The model tags tokens (BIOES over eight PII types). A turn counts as PII when
any of its tokens is more likely PII than `O`; the turn's score is the highest
per-token probability of not being `O`, so the ROC curve sweeps that bar. This
is the plain per-token reading of the `transformers` head, not OpenAI's
constrained Viterbi decoder with its operating points.
"""

from typing import Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyConfig, SafetyInput, SafetyOutput, dtype_kwargs, empty_text_prediction, resolve_device


logger = get_logger(__name__)

HUB_MODEL_ID = "openai/privacy-filter"
MIN_TRANSFORMERS = (5, 6)


def render_window(window: list[tuple[str, str, bool]]) -> tuple[str, list[tuple[int, int]]]:
    """One text for the window and the character ranges of its target turns.

    A window of target turns only is their texts joined by newlines; with
    context, every turn is a `speaker: text` line.
    """
    with_speakers = not all(target for _, _, target in window)
    pieces, ranges, position = [], [], 0
    for speaker, text, target in window:
        prefix = f"{speaker}: " if with_speakers else ""
        start = position + len(prefix)
        if target:
            ranges.append((start, start + len(text)))
        pieces.append(prefix + text)
        position = start + len(text) + 1
    return "\n".join(pieces), ranges


def overlaps(token: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    start, end = token
    return end > start and any(end > a and start < b for a, b in ranges)


def entity_type(label: str) -> str:
    """`B-private_phone` -> `private_phone`."""
    return label.split("-", 1)[1] if "-" in label else label


def spans_from_tokens(
    offsets: list[tuple[int, int]], labels: list[str], text: str, ranges: list[tuple[int, int]]
) -> list[tuple[str, str]]:
    """Contiguous runs of same-type PII tokens inside the target ranges, as (type, text)."""
    spans: list[tuple[str, str]] = []
    current: tuple[str, int, int] | None = None
    for offset, label in zip(offsets, labels):
        kind = entity_type(label) if label != "O" and overlaps(offset, ranges) else None
        if current is not None and kind == current[0] and not label.startswith(("B-", "S-")):
            current = (current[0], current[1], offset[1])
            continue
        if current is not None:
            spans.append((current[0], text[current[1] : current[2]].strip()))
        current = (kind, offset[0], offset[1]) if kind else None
    if current is not None:
        spans.append((current[0], text[current[1] : current[2]].strip()))
    return spans


class HuggingFacePrivacyFilterConfig(SafetyConfig):
    model_id: str = Field(HUB_MODEL_ID, description="Hub id or local directory of the checkpoint.")
    context_turns: int = Field(
        0,
        ge=0,
        description="Turns kept before the first target turn; only tokens inside the target turns count.",
    )
    device: str | None = Field(None, description="torch device; defaults to CUDA, then MPS, then CPU.")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "bfloat16", description="Weights dtype (bf16 as shipped)."
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
        device = resolve_device(config.device)
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, truncation_side="left")
        model = AutoModelForTokenClassification.from_pretrained(config.model_id, **dtype_kwargs(config.dtype))
        model.to(device).eval()
        id2label = {int(i): str(name) for i, name in model.config.id2label.items()}
        outside_id = next(i for i, name in id2label.items() if name == "O")
        logger.info(f"{config.model_id} on {device} ({config.dtype}), context turns {config.context_turns}")

        def classify(sample: SafetyInput) -> SafetyPrediction:
            if not sample.require_text().strip():
                return empty_text_prediction()
            text, ranges = render_window(sample.window(config.context_turns))
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
                return SafetyPrediction(label=SafetyLabel.SAFE, unsafe_score=0.0, raw_output="")
            top = max(in_target, key=lambda i: not_outside[i])
            unsafe_score = float(not_outside[top])
            is_unsafe = unsafe_score >= config.threshold
            top_type = probabilities[top].clone()
            top_type[outside_id] = -1.0
            spans = spans_from_tokens(offsets, [id2label[i] for i in predicted], text, ranges)
            return SafetyPrediction(
                label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
                unsafe_score=unsafe_score,
                category=entity_type(id2label[int(top_type.argmax())]) if is_unsafe else None,
                raw_output="; ".join(f"{kind}={value!r}" for kind, value in spans)[:1000],
            )

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

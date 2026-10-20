# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Perplexity's PII-Tracer from the Hugging Face Hub, read as a per-segment PII verdict.

A 0.6B bidirectional Qwen3 encoder tagging BIOES over Privacy Filter's eight
types plus `other_pii`, decoded by the checkpoint's own constrained Viterbi
(`modeling_pii_masking.py`, loaded with `trust_remote_code`). The segment is PII
when the decoder keeps a span inside it; the score is the highest per-token
probability of not being `O` inside the segment, for the ROC curve.
"""

from typing import Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, dtype_kwargs, empty_text_prediction, resolve_device
from .span_common import SpanPipelineConfig, overlaps, render_window, span_prediction, target_spans


logger = get_logger(__name__)

HUB_MODEL_ID = "perplexity-ai/PII-Tracer"


class HuggingFacePIITracerConfig(SpanPipelineConfig):
    model_id: str = Field(HUB_MODEL_ID, description="Hub id or local directory of the checkpoint.")
    device: str | None = Field(None, description="torch device; defaults to CUDA, then MPS, then CPU.")
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = Field(
        "auto", description="Encoder dtype; `auto` is bf16 (as shipped) on CUDA and float32 elsewhere."
    )
    max_tokens: int = Field(4096, description="The model's window; longer inputs keep their end (the target turn).")


@register_pipeline
class HuggingFacePIITracerPipeline(Pipeline):
    _config_class = HuggingFacePIITracerConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        import torch
        from transformers import AutoModel

        config = self.config
        device = resolve_device(config.device)
        dtype = config.dtype if config.dtype != "auto" else ("bfloat16" if device.startswith("cuda") else "float32")
        model = AutoModel.from_pretrained(config.model_id, trust_remote_code=True, **dtype_kwargs(dtype))
        model.to(device).eval()
        tokenizer = model.tokenizer
        tokenizer.truncation_side = "left"
        outside_id = model.viterbi.label2id["O"]
        logger.info(f"{config.model_id} on {device} ({dtype}), context turns {config.context_turns}")

        def classify(sample: SafetyInput) -> SafetyPrediction:
            if not sample.require_text().strip():
                return empty_text_prediction()
            text, ranges = render_window(sample.window(config.context_turns, config.context_speakers))
            encoded = tokenizer(
                text, truncation=True, max_length=config.max_tokens, return_offsets_mapping=True, return_tensors="pt"
            )
            offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
            with torch.inference_mode():
                logits = (
                    model(
                        input_ids=encoded["input_ids"].to(device), attention_mask=encoded["attention_mask"].to(device)
                    )
                    .logits[0]
                    .float()
                    .cpu()
                )
            not_outside = 1.0 - torch.softmax(logits, dim=-1)[:, outside_id]
            in_target = [i for i, offset in enumerate(offsets) if overlaps(offset, ranges)]
            score = float(max((not_outside[i] for i in in_target), default=0.0))
            decoded = model.viterbi.decode(logits, offsets, text=text)
            spans = target_spans([(s.start, s.end, s.label, float(s.score)) for s in decoded], ranges)
            return span_prediction(spans, score, config.threshold, flag_on_spans=True)

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""GLiNER-family PII detectors, read as a per-segment PII verdict.

GLiNER models are zero-shot span extractors: the PII types are passed in as
label names. Two libraries cover the family: `gliner` for GLiNER v1 and its PII
fine-tunes (urchade, NVIDIA, Knowledgator) and `gliner2` for GLiNER2 and the
models built on it (Fastino's GLiNER2-PII, hivetrace's GLiNER Guard). Spans are
extracted down to `score_floor` so the ROC curve has scores to sweep; the
segment is PII when a span inside it reaches `threshold`, and only those spans
are censored; the candidates below it stay in `raw_output` with their scores.
"""

from typing import Any, Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, empty_text_prediction
from .span_common import SpanPipelineConfig, render_window, span_prediction, target_spans


logger = get_logger(__name__)


class GLiNERPIIConfig(SpanPipelineConfig):
    model_id: str = Field(..., description="Hub id of the GLiNER checkpoint.")
    library: Literal["gliner", "gliner2"] = Field("gliner", description="`gliner` (v1) or `gliner2`.")
    labels: list[str] = Field(..., description="PII label names passed to the model (its own taxonomy works best).")
    threshold: float = Field(0.3, description="Span confidence at or above which a span counts (and is censored).")
    score_floor: float = Field(0.05, description="Lowest span confidence extracted, for the score distribution.")
    device: str | None = Field(None, description="torch device; defaults to CUDA, else CPU.")


Span = tuple[int, int, str, float]


def gliner_v1_spans(
    model: Any, texts: list[str], labels: list[str], floor: float, batch_size: int
) -> list[list[Span]]:
    batches = model.inference(texts, labels, threshold=floor, batch_size=batch_size)
    return [[(int(e["start"]), int(e["end"]), str(e["label"]), float(e["score"])) for e in found] for found in batches]


def gliner2_entities(result: dict) -> list[Span]:
    spans = []
    for label, entities in (result.get("entities") or {}).items():
        for entity in entities or []:
            if isinstance(entity, dict) and entity.get("start") is not None:
                spans.append((int(entity["start"]), int(entity["end"]), str(label), float(entity["confidence"])))
    return spans


def gliner2_spans(model: Any, texts: list[str], labels: list[str], floor: float, batch_size: int) -> list[list[Span]]:
    results = model.batch_extract_entities(
        texts, labels, batch_size=batch_size, threshold=floor, include_confidence=True, include_spans=True
    )
    return [gliner2_entities(result) for result in results]


@register_pipeline
class GLiNERPIIPipeline(Pipeline):
    _config_class = GLiNERPIIConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        import torch

        config = self.config
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if config.library == "gliner":
            from gliner import GLiNER

            model = GLiNER.from_pretrained(config.model_id)
            extract = gliner_v1_spans
        else:
            from gliner2 import GLiNER2

            model = GLiNER2.from_pretrained(config.model_id)
            extract = gliner2_spans
        model.to(device)
        model.eval()
        logger.info(f"{config.model_id} ({config.library}) on {device}, {len(config.labels)} labels")

        def classify_batch(samples: list[SafetyInput]) -> list[SafetyPrediction]:
            results: list[SafetyPrediction | None] = [None] * len(samples)
            work = []
            for index, sample in enumerate(samples):
                if not sample.require_text().strip():
                    results[index] = empty_text_prediction()
                else:
                    work.append((index, *render_window(sample.window(config.context_turns, config.context_speakers))))
            if work:
                found = extract(
                    model, [text for _, text, _ in work], config.labels, config.score_floor, config.batch_size
                )
                for (index, _, ranges), spans in zip(work, found):
                    spans = target_spans(spans, ranges)
                    score = max((span[3] for span in spans), default=0.0)
                    results[index] = span_prediction(spans, score, config.threshold, bar=config.threshold)
            return results

        self.classify_batch = classify_batch

        def classify(sample: SafetyInput) -> SafetyPrediction:
            return classify_batch([sample])[0]

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""A deterministic PII floor: patterns for the structured identifiers, no model.

What a phone could run for free before any model: e-mail addresses (written or
spoken, "jo at mail dot com"), URLs, phone numbers, Luhn-checked card numbers,
US SSNs, IBANs and IPv4 addresses. Names and addresses are out of its reach,
which is the point of the comparison.
"""

import re
from typing import Callable

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import SafetyInput, SafetyOutput, empty_text_prediction
from .span_common import SpanPipelineConfig, render_window, span_prediction, target_spans


PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "spoken_email": re.compile(r"\b[\w.]+ at [\w-]+ dot (?:com|net|org|edu|gov|co|io|de|fr|uk)\b", re.IGNORECASE),
    "url": re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "card_number": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "phone": re.compile(r"(?<![\w-])\+?\(?\d{1,4}\)?(?:[ .-]?\(?\d{2,4}\)?){2,5}(?![\w-])"),
}


def luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value = value * 2 - 9 if value > 4 else value * 2
        total += value
    return total % 10 == 0


def regex_spans(text: str) -> list[tuple[int, int, str, float]]:
    """Non-overlapping pattern hits, earlier and longer matches first."""
    hits = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            digits = re.sub(r"\D", "", value)
            if kind == "card_number" and not (13 <= len(digits) <= 19 and luhn_valid(digits)):
                continue
            if kind == "phone" and not 7 <= len(digits) <= 15:
                continue
            hits.append((match.start(), match.end(), kind, 1.0))
    hits.sort(key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    kept, last_end = [], -1
    for hit in hits:
        if hit[0] >= last_end:
            kept.append(hit)
            last_end = hit[1]
    return kept


@register_pipeline
class RegexPIIPipeline(Pipeline):
    _config_class = SpanPipelineConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        config = self.config

        def classify(sample: SafetyInput) -> SafetyPrediction:
            if not sample.require_text().strip():
                return empty_text_prediction()
            text, ranges = render_window(sample.window(config.context_turns, config.context_speakers))
            spans = target_spans(regex_spans(text), ranges)
            return span_prediction(spans, 1.0 if spans else 0.0, config.threshold)

        self.classify_batch = lambda samples: [classify(sample) for sample in samples]
        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

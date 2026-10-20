# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""What the span-level PII pipelines share: the window they read, the target ranges inside it, and the verdict.

A span model reads the target turns, optionally after the speaker's earlier
turns, and only spans inside the target turns count. The verdict is PII when
any span reaches the pipeline's bar; the score is the model's highest
confidence inside the target turns, so the ROC curve sweeps that bar.
`raw_output` is JSON with the spans as [start, end, type] in the sample's text
(its target turns joined by newlines), for character-level scoring.
"""

import json

from pydantic import Field

from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from .common import ContextSpeakers, SafetyConfig


class SpanPipelineConfig(SafetyConfig):
    context_turns: int = Field(
        0,
        ge=0,
        description="Turns kept before the first target turn; only spans inside the target turns count.",
    )
    context_speakers: ContextSpeakers = Field(
        "all", description="`target` keeps only the speaker under test in the context (an owner-only transcript)."
    )
    batch_size: int = Field(16, ge=1, description="Segments per forward pass in `classify_batch`.")


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


def overlaps(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    start, end = span
    return end > start and any(end > a and start < b for a, b in ranges)


def entity_type(label: str) -> str:
    """`B-private_phone` -> `private_phone`."""
    return label.split("-", 1)[1] if "-" in label else label


def to_target_offsets(span: tuple[int, int], ranges: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    """Map character offsets in the rendered window to the target text (the target turns joined by newlines).

    Returns (range index, start, end) for the first target range the span overlaps, or None.
    """
    start, end = span
    position = 0
    for index, (a, b) in enumerate(ranges):
        if end > start and end > a and start < b:
            return index, position + max(start, a) - a, position + min(end, b) - a
        position += b - a + 1
    return None


def target_spans(
    spans: list[tuple[int, int, str, float]], ranges: list[tuple[int, int]]
) -> list[tuple[int, int, str, float]]:
    """Character spans (start, end, type, score) in the window, kept and mapped when they fall in a target turn."""
    kept = []
    for start, end, kind, score in spans:
        mapped = to_target_offsets((start, end), ranges)
        if mapped is not None:
            kept.append((mapped[1], mapped[2], kind, score))
    return kept


def encode_batch(tokenizer, texts: list[str], max_tokens: int):
    """Tokenize a batch with padding and character offsets; returns (encoded tensors, per-item offsets).

    Offsets are cut to each item's real length, so padding never reaches a decoder.
    """
    encoded = tokenizer(
        texts, truncation=True, max_length=max_tokens, padding=True, return_offsets_mapping=True, return_tensors="pt"
    )
    offset_mapping = encoded.pop("offset_mapping").tolist()
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    if tokenizer.padding_side == "left":
        offsets = [[tuple(o) for o in row[len(row) - n :]] for row, n in zip(offset_mapping, lengths)]
    else:
        offsets = [[tuple(o) for o in row[:n]] for row, n in zip(offset_mapping, lengths)]
    return encoded, offsets


def item_logits(logits, index: int, length: int, padding_side: str):
    """One item's [length, C] logits out of a padded batch."""
    row = logits[index]
    return row[row.shape[0] - length :] if padding_side == "left" else row[:length]


def span_prediction(
    spans: list[tuple[int, int, str, float]],
    score: float,
    threshold: float,
    *,
    flag_on_spans: bool = False,
    bar: float = 0.0,
) -> SafetyPrediction:
    """The verdict from target spans and the target's best score.

    PII when the score reaches `threshold`, or, with `flag_on_spans`, whenever
    the model's own decoder kept a span (models whose decoder is the verdict).
    `raw_output` lists every span as [start, end, type, score] with `bar`, the
    span score at or above which a span is censored; spans below it are kept
    as candidates, so a reader can re-score the segment under a different
    choice of PII types or a different bar.
    """
    censored = [span for span in spans if span[3] >= bar]
    is_unsafe = bool(censored) if flag_on_spans else score >= threshold
    top = max(censored, key=lambda span: span[3]) if censored else None
    return SafetyPrediction(
        label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
        unsafe_score=float(score),
        category=top[2] if is_unsafe and top else None,
        raw_output=json.dumps(
            {"bar": bar, "spans": [[start, end, kind, round(float(sc), 4)] for start, end, kind, sc in spans]}
        ),
    )


def parse_spans(raw_output: str | None) -> tuple[list[tuple[int, int, str, float]], float] | None:
    """(spans with scores, bar) from a span pipeline's `raw_output`, or None for other outputs.

    Spans written before scores were kept ([start, end, type]) read as score 1.
    """
    try:
        payload = json.loads(raw_output or "")
        spans = payload["spans"]
    except (ValueError, KeyError, TypeError):
        return None
    parsed = [(int(s[0]), int(s[1]), str(s[2]), float(s[3]) if len(s) > 3 else 1.0) for s in spans]
    return parsed, float(payload.get("bar", 0.0))

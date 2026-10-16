# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""What the Roblox PII classifier v2 pipelines share: the checkpoint's input format and the reading of its heads."""

import numpy as np
from pydantic import Field

from ...pipeline_prediction import SafetyLabel, SafetyPrediction
from .common import SafetyConfig, SafetyInput
from .coreml_common import threshold_shifted_probability


# Byte-exact with the checkpoint's `format_conversation` and Sova's ConversationFormatter
INSTRUCTION_PREFIX = (
    "Instruct: In the following chat messages from target speaker t and possibly other speakers s1, s2, etc., "
    "detect abuse by speaker t.\nQuery:"
)
PII_HEADS = ("privacy_asking_for_pii", "privacy_giving_pii", "directing_users_off_platform")
# The model card's recommended thresholds, in head order
CARD_THRESHOLDS = {"privacy_asking_for_pii": 0.60, "privacy_giving_pii": 0.55, "directing_users_off_platform": 0.10}
SEQUENCE_LENGTH = 512
HUB_MODEL_ID = "Roblox/roblox-pii-classifier-v2"


def format_conversation(turns: list[tuple[str, str]]) -> str:
    """Render (speaker, text) turns the way the classifier was trained: `t` is the target, others `s1`, `s2`..."""
    names: dict[str, str] = {}
    rendered = []
    for speaker, text in turns:
        if speaker == "t":
            label = "t"
        else:
            label = names.setdefault(speaker, f"s{len(names) + 1}")
        rendered.append(f"{label}: {text}")
    return INSTRUCTION_PREFIX + "\n\n" + " </s> ".join(rendered)


class RobloxPIIConfig(SafetyConfig):
    heads: list[str] = Field(
        default_factory=lambda: ["privacy_giving_pii"],
        description="Heads that count towards the verdict; any of them firing flags the turn.",
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: dict(CARD_THRESHOLDS), description="Per-head thresholds (model card by default)."
    )
    context_turns: int = Field(
        16,
        ge=0,
        description=(
            "Turns kept before the first target turn (the oldest are dropped first by the 512-token left "
            "truncation); 0 scores the target turns alone."
        ),
    )


def validate_heads(config: RobloxPIIConfig) -> list[int]:
    unknown = set(config.heads) - set(PII_HEADS)
    if unknown:
        raise ValueError(f"Unknown heads {sorted(unknown)}; the model has {PII_HEADS}")
    return [PII_HEADS.index(h) for h in config.heads]


def render_input(sample: SafetyInput, context_turns: int) -> str | None:
    """The formatted window for one sample, or None when the target turns are empty."""
    if not sample.require_text().strip():
        return None
    return format_conversation([(speaker, text) for speaker, text, _ in sample.window(context_turns)])


def verdict(probabilities: np.ndarray, config: RobloxPIIConfig, head_indices: list[int]) -> SafetyPrediction:
    """Flag the turn when any selected head reaches its threshold.

    The score is the maximum over the selected heads after shifting each so
    its own threshold maps to 0.5; the category is the head that fired highest.
    """
    shifted = {
        PII_HEADS[i]: threshold_shifted_probability(probabilities[i], config.thresholds[PII_HEADS[i]])
        for i in head_indices
    }
    top_head = max(shifted, key=shifted.get)
    unsafe_score = shifted[top_head]
    is_unsafe = unsafe_score >= 0.5
    return SafetyPrediction(
        label=SafetyLabel.UNSAFE if is_unsafe else SafetyLabel.SAFE,
        unsafe_score=unsafe_score,
        category=top_head if is_unsafe else None,
        raw_output=", ".join(f"{h}={probabilities[i]:.4f}" for i, h in enumerate(PII_HEADS)),
    )

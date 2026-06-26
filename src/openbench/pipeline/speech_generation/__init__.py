# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from .speech_generation_argmax_oss import (
    ArgmaxOpenSourceSpeechGenerationConfig,
    ArgmaxOpenSourceSpeechGenerationPipeline,
)
from .speech_generation_argmax_prototype import (
    ArgmaxPrototypeSpeechGenerationConfig,
    ArgmaxPrototypeSpeechGenerationPipeline,
)


__all__ = [
    "ArgmaxOpenSourceSpeechGenerationConfig",
    "ArgmaxOpenSourceSpeechGenerationPipeline",
    "ArgmaxPrototypeSpeechGenerationConfig",
    "ArgmaxPrototypeSpeechGenerationPipeline",
]

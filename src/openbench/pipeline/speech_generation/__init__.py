# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from .common import SpeechGenerationConfig, SpeechGenerationOutput
from .speech_generation_cartesia import (
    CartesiaSpeechGenerationConfig,
    CartesiaSpeechGenerationPipeline,
)
from .speech_generation_elevenlabs import (
    ElevenLabsSpeechGenerationConfig,
    ElevenLabsSpeechGenerationPipeline,
)
from .speech_generation_gemini import (
    GeminiSpeechGenerationConfig,
    GeminiSpeechGenerationPipeline,
)
from .speech_generation_openai import (
    OpenAISpeechGenerationConfig,
    OpenAISpeechGenerationPipeline,
)
from .speech_generation_wkp import WhisperKitSpeechGenerationPipeline


__all__ = [
    "CartesiaSpeechGenerationConfig",
    "CartesiaSpeechGenerationPipeline",
    "ElevenLabsSpeechGenerationConfig",
    "ElevenLabsSpeechGenerationPipeline",
    "GeminiSpeechGenerationConfig",
    "GeminiSpeechGenerationPipeline",
    "OpenAISpeechGenerationConfig",
    "OpenAISpeechGenerationPipeline",
    "SpeechGenerationConfig",
    "SpeechGenerationOutput",
    "WhisperKitSpeechGenerationPipeline",
]

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from .apple_speech_analyzer import SpeechAnalyzerConfig, SpeechAnalyzerPipeline
from .common import TranscriptionOutput
from .transcription_groq import GroqTranscriptionConfig, GroqTranscriptionPipeline
from .transcription_whisperkitpro import WhisperKitProTranscriptionConfig, WhisperKitProTranscriptionPipeline
from .whisperkit import WhisperKitTranscriptionConfig, WhisperKitTranscriptionPipeline
from .transcription_opena import OpenAITranscriptionPipelineConfig, OpenAITranscriptionPipeline
from .transcription_deepgram import DeepgramTranscriptionPipelineConfig, DeepgramTranscriptionPipeline
from .transcription_nemo import NeMoTranscriptionPipelineConfig, NeMoTranscriptionPipeline


__all__ = [
    "TranscriptionOutput",
    "SpeechAnalyzerPipeline",
    "SpeechAnalyzerConfig",
    "WhisperKitTranscriptionPipeline",
    "WhisperKitTranscriptionConfig",
    "WhisperKitProTranscriptionPipeline",
    "WhisperKitProTranscriptionConfig",
    "OpenAITranscriptionPipeline",
    "OpenAITranscriptionPipelineConfig",
    "DeepgramTranscriptionPipeline",
    "DeepgramTranscriptionPipelineConfig",
    "NeMoTranscriptionPipeline",
    "NeMoTranscriptionPipelineConfig",
]

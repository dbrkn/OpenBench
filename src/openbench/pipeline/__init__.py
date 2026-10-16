# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

# ruff: noqa
"""Pipelines. The audio families (diarization, orchestration, speech generation, streaming and batch transcription)
need the audio stack (pyannote.audio, NeMo, whisperx, the vendor SDKs); on an environment without it each missing
family is skipped with one warning and only the safety pipelines register, which is all the text safety benchmarks
need. `UNAVAILABLE_FAMILIES` names the skipped families and why."""
import importlib
import warnings

from .base import PIPELINE_REGISTRY, Pipeline, register_pipeline
from .pipeline_registry import PipelineRegistry
from .safety import *

UNAVAILABLE_FAMILIES: dict[str, str] = {}
for _family in ("diarization", "orchestration", "speech_generation", "streaming_transcription", "transcription"):
    try:
        _module = importlib.import_module(f".{_family}", __name__)
    except ImportError as error:  # a missing audio dependency; the safety pipelines do not need it
        UNAVAILABLE_FAMILIES[_family] = str(error)
        warnings.warn(f"openbench.pipeline.{_family} unavailable ({error}); its pipelines are not registered")
        continue
    globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})

# Import pipeline aliases to register them
# needs to be imported at the end to avoid circular imports
from . import pipeline_aliases

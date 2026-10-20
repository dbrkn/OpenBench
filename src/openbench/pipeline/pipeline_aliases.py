# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Pipeline alias registrations for common configurations."""

# The audio pipeline classes are bound through globals() below (their families are optional), which ruff cannot see.
# ruff: noqa: F821

import importlib
import os
from typing import Any

from .pipeline_registry import PipelineRegistry
from .safety import (
    ConstantSafetyPipeline,
    CoreMLRobloxPIIPipeline,
    GLiNERPIIPipeline,
    HuggingFacePIITracerPipeline,
    HuggingFacePrivacyFilterPipeline,
    HuggingFaceRobloxPIIPipeline,
    RegexPIIPipeline,
)


# The audio families are optional (see `openbench.pipeline`): a family whose imports fail binds its pipeline classes
# to None and `_register_alias` skips their aliases.
_FAMILY_PIPELINES = {
    "diarization": (
        "ArgmaxOpenSourceDiarizationPipeline",
        "AWSTranscribePipeline",
        "DeepgramDiarizationPipeline",
        "ElevenLabsDiarizationPipeline",
        "NeMoSortformerPipeline",
        "PicovoicePipeline",
        "PyannoteApiPipeline",
        "PyAnnotePipeline",
        "SpeakerKitPipeline",
    ),
    "orchestration": (
        "ArgmaxOpenSourceOrchestrationPipeline",
        "DeepgramOrchestrationPipeline",
        "ElevenLabsOrchestrationPipeline",
        "NeMoMTParakeetPipeline",
        "OpenAIOrchestrationPipeline",
        "PyannoteOrchestrationPipeline",
        "WhisperKitProOrchestrationPipeline",
        "WhisperXPipeline",
    ),
    "speech_generation": ("ArgmaxOpenSourceSpeechGenerationPipeline",),
    "streaming_transcription": (
        "AssemblyAIStreamingPipeline",
        "DeepgramStreamingPipeline",
        "FireworksStreamingPipeline",
        "GladiaStreamingPipeline",
        "OpenAIStreamingPipeline",
    ),
    "transcription": (
        "ArgmaxOpenSourceTranscriptionPipeline",
        "AssemblyAITranscriptionPipeline",
        "DeepgramTranscriptionPipeline",
        "ElevenLabsTranscriptionPipeline",
        "GroqTranscriptionPipeline",
        "NeMoTranscriptionPipeline",
        "OpenAITranscriptionPipeline",
        "PyannoteTranscriptionPipeline",
        "SpeechAnalyzerPipeline",
        "WhisperKitProTranscriptionPipeline",
        "WhisperOSSTranscriptionPipeline",
    ),
}
for _family, _names in _FAMILY_PIPELINES.items():
    try:
        _module = importlib.import_module(f".{_family}", __package__)
    except ImportError:  # reported once by `openbench.pipeline`
        _module = None
    for _name in _names:
        globals()[_name] = getattr(_module, _name, None)


def _register_alias(alias: str, pipeline_class: Any, default_config: dict[str, Any], description: str = "") -> None:
    """`PipelineRegistry.register_alias`, skipping the aliases of a pipeline family that did not import."""
    if pipeline_class is None:
        return
    PipelineRegistry.register_alias(alias, pipeline_class, default_config=default_config, description=description)


def register_pipeline_aliases() -> None:
    """Register all pipeline aliases with their configurations."""

    ################# DIARIZATION PIPELINES #################
    _register_alias(
        "aws-diarization",
        AWSTranscribePipeline,
        default_config={
            "out_dir": "./aws_diarization_results",
            "bucket_name": "diarization-benchmarks",
            "region_name": "us-east-2",
            "max_speakers": 30,
            "num_worker_processes": 8,
            "per_worker_chunk_size": 1,
        },
        description="AWS Transcribe with speaker diarization. Requires AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) and S3 bucket setup.",
    )

    _register_alias(
        "pyannote",
        PyAnnotePipeline,
        default_config={
            "out_dir": "./pyannote_logs",
            "num_speakers": None,
            "min_speakers": None,
            "max_speakers": None,
            "use_oracle_clustering": False,
            "use_oracle_segmentation": False,
            "use_float16": True,
        },
        description="Pyannote open-source speaker diarization pipeline.",
    )

    _register_alias(
        "nemo-sortformer",
        NeMoSortformerPipeline,
        default_config={
            "out_dir": "./nemo_sortformer_logs",
            "use_float16": True,
            "chunk_size": 340,
            "right_context": 40,
            "fifo_size": 40,
            "update_period": 300,
            "speaker_cache_size": 188,
        },
        description="NeMo Sortformer speaker diarization pipeline.",
    )

    _register_alias(
        "pyannote-api",
        PyannoteApiPipeline,
        default_config={
            "out_dir": "./pyannoteapi",
            "timeout": 3600,
            "request_buffer": 30,
            "model": "precision-3",
        },
        description="PyannoteAI API speaker diarization pipeline using the precision-3 model. Requires `PYANNOTE_TOKEN` env var from https://www.pyannote.ai/.",
    )

    _register_alias(
        "speakerkit",
        SpeakerKitPipeline,
        default_config={
            "out_dir": "./speakerkit-report",
            "cli_path": os.getenv("SPEAKERKIT_CLI_PATH"),
            "engine": "pyannote",
        },
        description="SpeakerKit speaker diarization pipeline using community-1 model from pyannote. Requires CLI installation and API key. Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. For access to the CLI binary contact speakerkitpro@argmaxinc.com",
    )

    _register_alias(
        "speakerkit-sortformer-compressed",
        SpeakerKitPipeline,
        default_config={
            "out_dir": "./speakerkit-sortformer-report",
            "cli_path": os.getenv("SPEAKERKIT_CLI_PATH"),
            "engine": "sortformer",
            # Pinned explicitly: speakerkitpro-cli >= 3.1.6 defaults to Nemotron 3 Diarization.
            "sortformer_model_version": "v2-1",
            "sortformer_model_variant": "384_94MB",
        },
        description=(
            "SpeakerKit speaker diarization pipeline using the Sortformer v2 model compressed to 94MB "
            "(`--sortformer-model-version v2-1 --sortformer-model-variant 384_94MB`). Requires CLI installation and API key. "
            "Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. For access to the CLI binary contact speakerkitpro@argmaxinc.com."
        ),
    )

    _register_alias(
        "speakerkit-nemotron-3-diarization",
        SpeakerKitPipeline,
        default_config={
            "out_dir": "./speakerkit-nemotron-3-diarization-report",
            "cli_path": os.getenv("SPEAKERKIT_CLI_PATH"),
            "engine": "sortformer",
            "sortformer_model_version": "nemotron-3-diarization",
            "sortformer_model_variant": "684_74MB",
        },
        description=(
            "SpeakerKit speaker diarization pipeline using NVIDIA Nemotron 3 Diarization (Sortformer v3, 74MB variant; "
            "`--sortformer-model-version nemotron-3-diarization --sortformer-model-variant 684_74MB`). Requires CLI installation "
            "and API key. Set `SPEAKERKIT_CLI_PATH` and `SPEAKERKIT_API_KEY` env vars. "
            "For access to the CLI binary contact speakerkitpro@argmaxinc.com."
        ),
    )

    _register_alias(
        "argmax-oss-diarization",
        ArgmaxOpenSourceDiarizationPipeline,
        default_config={
            "out_dir": "./argmax_oss_diarization_reports",
        },
        description=(
            "Argmax SDK open-source diarization via `argmax-cli diarize`. "
            "Clone/build under ARGMAX_OSS_CACHE_DIR (default ~/.cache/openbench/argmax-oss) unless `cli_path` is set."
            "Uses pyannote's community-1 model for diarization."
        ),
    )

    _register_alias(
        "picovoice-diarization",
        PicovoicePipeline,
        default_config={
            "out_dir": "./picovoice_logs",
        },
        description="Picovoice diarization pipeline. Requires API key from https://www.picovoice.ai/. Set `PICOVOICE_API_KEY` env var.",
    )

    _register_alias(
        "deepgram-diarization",
        DeepgramDiarizationPipeline,
        default_config={
            "out_dir": "./deepgram_diarization_results",
            "model_version": "nova-3",
        },
        description="Deepgram diarization pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    _register_alias(
        "elevenlabs-diarization",
        ElevenLabsDiarizationPipeline,
        default_config={
            "out_dir": "./elevenlabs_diarization_results",
            "model_id": "scribe_v2",
        },
        description="ElevenLabs diarization pipeline. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    ################# ORCHESTRATION PIPELINES #################

    _register_alias(
        "whisperx-tiny",
        WhisperXPipeline,
        default_config={
            "out_dir": "./whisperx_output",
            "model_name": "tiny",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 16,
            "threads": 8,
        },
        description="WhisperX diarized transcription pipeline from https://github.com/m-bain/whisperX running with the tiny model",
    )

    _register_alias(
        "whisperx-large-v3-turbo",
        WhisperXPipeline,
        default_config={
            "out_dir": "./whisperx_output",
            "model_name": "large-v3-turbo",
            "device": "cpu",
            "compute_type": "int8",
            "batch_size": 16,
            "threads": 8,
        },
        description="WhisperX diarzed transcription pipeline from https://github.com/m-bain/whisperX running with the large-v3-turbo model",
    )

    _register_alias(
        "deepgram-orchestration",
        DeepgramOrchestrationPipeline,
        default_config={
            "out_dir": "./deepgram_orchestration_results",
            "model_version": "nova-3",
        },
        description="Deepgram orchestration pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    _register_alias(
        "elevenlabs-orchestration",
        ElevenLabsOrchestrationPipeline,
        default_config={
            "out_dir": "./elevenlabs_orchestration_results",
            "model_id": "scribe_v2",
        },
        description="ElevenLabs orchestration pipeline with diarization. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-tiny",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-tiny",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the tiny version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-large-v3",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-large-v3-turbo",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3-v20240930 version of the model (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-large-v3-turbo-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930_626MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the large-v3-v20240930 version of the model compressed to 626MB (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v2",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v2-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model compressed to 476MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v2-compressed-sortformer-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "sortformer",
            "diarization_mode": "prerecorded",
        },
        description=(
            "WhisperKitPro orchestration pipeline using the parakeet-v2 version of the model compressed to 476MB and using Sortformer for diarization. "
            "Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var."
        ),
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v3",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v3-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "pyannote",
            "use_exclusive_reconciliation": True,
        },
        description="WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model compressed to 494MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-orchestration-parakeet-v3-compressed-sortformer-compressed",
        WhisperKitProOrchestrationPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
            "orchestration_strategy": "subsegment",
            "engine": "sortformer",
            "diarization_mode": "prerecorded",
        },
        description=(
            "WhisperKitPro orchestration pipeline using the parakeet-v3 version of the model compressed to 494MB and using Sortformer for diarization. "
            "Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var."
        ),
    )

    _register_alias(
        "openai-orchestration",
        OpenAIOrchestrationPipeline,
        default_config={
            "model_version": "gpt-4o-transcribe-diarize",
        },
        description="OpenAI orchestration pipeline using the `gpt-4o-transcribe-diarize` version of the model. Requires `OPENAI_API_KEY` env var.",
    )

    _register_alias(
        "nemo-mt-parakeet",
        NeMoMTParakeetPipeline,
        default_config={
            "out_dir": "./nemo_mt_parakeet_logs",
            "diar_model_id": "nvidia/diar_streaming_sortformer_4spk-v2.1",
            "asr_model_id": "nvidia/multitalker-parakeet-streaming-0.6b-v1",
        },
        description="NeMo Multi-Talker Parakeet orchestration pipeline (diarization + transcription).",
    )

    _register_alias(
        "pyannote-orchestration",
        PyannoteOrchestrationPipeline,
        default_config={
            "out_dir": "./pyannote_orchestration_results",
            "timeout": 3600,
            "request_buffer": 30,
            "model": "precision-3",
        },
        description="PyannoteAI orchestration pipeline (diarization + transcription). Uses the precision-3 model with Nvidia Parakeet STT. Requires `PYANNOTE_TOKEN` env var from https://www.pyannote.ai/.",
    )

    _register_alias(
        "argmax-oss-orchestration-tiny",
        ArgmaxOpenSourceOrchestrationPipeline,
        default_config={
            "out_dir": "./argmax_oss_orchestration_reports",
            "model_version": "tiny",
            "word_timestamps": False,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (OSS): `argmax-cli transcribe --diarization` with verbose RTTM-like diarization log "
        "as the transcript (word + speaker per token). Cache: ARGMAX_OSS_CACHE_DIR or default; optional `cli_path`.",
    )

    ################# TRANSCRIPTION PIPELINES #################

    _register_alias(
        "whisperkit-tiny",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "tiny",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (Swift release build, not debug). Model tiny. "
        "Cache: ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss unless `cli_path` is set. "
        "For `openbench-cli evaluate`, prefer `-d earnings22-3hours` over `librispeech-200`: longer clips amortize "
        "first-run model load so speed factor is less misleading than on very short utterances.",
    )

    _register_alias(
        "whisperkit-large-v3",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "large-v3",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (release build). Model large-v3. "
        "Cache: ARGMAX_OSS_CACHE_DIR or default. For transcription benchmarks use `-d earnings22-3hours`; "
        "`librispeech-200` is better reserved for quick WER smoke tests.",
    )

    _register_alias(
        "whisperkit-large-v3-turbo",
        ArgmaxOpenSourceTranscriptionPipeline,
        default_config={
            "model_version": "large-v3-v20240930",
            "word_timestamps": True,
            "chunking_strategy": "vad",
        },
        description="Argmax SDK (open source) transcription via `argmax-cli` (release build). Model large-v3-v20240930. "
        "Cache: ARGMAX_OSS_CACHE_DIR or default. Prefer `-d earnings22-3hours` for evaluate; short LibriSpeech "
        "clips skew speed factor because of fixed startup cost per file.",
    )

    _register_alias(
        "speech-analyzer",
        SpeechAnalyzerPipeline,
        default_config={
            "clone_dir": "./speech_analyzer_repo",
        },
        description="Speech Analyzer transcription pipeline (open-source version). Requires Swift and Xcode installed.",
    )

    # Legacy mode without manual local model download
    # PipelineRegistry.register_alias(
    #     "whisperkitpro-tiny",
    #     WhisperKitProTranscriptionPipeline,
    #     default_config={
    #         "model_version": "tiny",
    #         "model_prefix": "openai",
    #         "model_repo_name": "argmaxinc/whisperkit-pro",
    #         "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
    #         "use_keywords": False
    #     },
    #     description="WhisperKitPro transcription pipeline using the tiny version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    # )

    # New mode with manual local model download
    # to avoid HF Download Rate Limiting
    _register_alias(
        "whisperkitpro-tiny",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-tiny",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the tiny version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-large-v3",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3 version of the model. Requires Swift and Xcode installed. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-large-v3-turbo",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3-v20240930 version of the model (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-large-v3-turbo-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/whisperkit-pro",
            "model_variant": "openai_whisper-large-v3-v20240930_626MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the large-v3-v20240930 version of the model compressed to 626MB (which is the same as large-v3-turbo from OpenAI). Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-parakeet-v2",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v2 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-parakeet-v2-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v2_476MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v2 version of the model compressed to 476MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-parakeet-v3",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v3 version of the model. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-parakeet-v3-compressed",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "repo_id": "argmaxinc/parakeetkit-pro",
            "model_variant": "nvidia_parakeet-v3_494MB",
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description="WhisperKitPro transcription pipeline using the parakeet-v3 version of the model compressed to 494MB. Requires `WHISPERKITPRO_CLI_PATH` env var and depending on your permissions also `WHISPERKITPRO_API_KEY` env var.",
    )

    _register_alias(
        "whisperkitpro-local-model",
        WhisperKitProTranscriptionPipeline,
        default_config={
            "model_dir": os.getenv("WHISPERKITPRO_LOCAL_MODEL_PATH"),
            "cli_path": os.getenv("WHISPERKITPRO_CLI_PATH"),
        },
        description=(
            "WhisperKitPro transcription using only a local model directory (no default Hugging Face repo). "
            "Set `WHISPERKITPRO_LOCAL_MODEL_PATH` to the folder passed as `--model-path` on the CLI; it must exist. "
            "Requires `WHISPERKITPRO_CLI_PATH` and may require `WHISPERKITPRO_API_KEY`."
        ),
    )

    _register_alias(
        "groq-whisper-large-v3-turbo",
        GroqTranscriptionPipeline,
        default_config={
            "model_id": "whisper-large-v3-turbo",
            "temperature": 0.0,
            "force_language": False,
        },
        description="Groq transcription pipeline using the whisper-large-v3-turbo model. Requires `GROQ_API_KEY` env var.",
    )

    _register_alias(
        "groq-whisper-large-v3",
        GroqTranscriptionPipeline,
        default_config={
            "model_id": "whisper-large-v3",
            "temperature": 0.0,
            "force_language": False,
        },
        description="Groq transcription pipeline using the whisper-large-v3 model. Requires `GROQ_API_KEY` env var.",
    )

    ################# KEYWORD-ENABLED TRANSCRIPTION PIPELINES #################

    _register_alias(
        "openai-transcription",
        OpenAITranscriptionPipeline,
        default_config={
            "out_dir": "./openai_keywords_results",
            "model_version": "whisper-1",
            "use_keywords": False,
        },
        description="OpenAI Whisper transcription with keyword boosting. Requires `OPENAI_API_KEY` env var.",
    )

    _register_alias(
        "whisper-tiny-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "tiny",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    _register_alias(
        "whisper-base-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "base",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    _register_alias(
        "whisper-large-v2-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v2",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    _register_alias(
        "whisper-large-v3-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v3",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    _register_alias(
        "whisper-turbo-oss-transcription",
        WhisperOSSTranscriptionPipeline,
        default_config={
            "out_dir": "./whisper_oss_results",
            "model_version": "large-v3-turbo",
            "device": None,
            "use_keywords": False,
        },
        description="Open-source OpenAI Whisper transcription (local)",
    )

    _register_alias(
        "deepgram-transcription",
        DeepgramTranscriptionPipeline,
        default_config={
            "out_dir": "./deepgram_keywords_results",
            "model_version": "nova-3",
            "use_keywords": False,
        },
        description="Deepgram transcription with keyword boosting. Requires `DEEPGRAM_API_KEY` env var.",
    )

    _register_alias(
        "nemo-transcription-ctc-large",
        NeMoTranscriptionPipeline,
        default_config={
            "out_dir": "./nemo_keywords_results",
            "nemo_model_file": "nvidia/stt_en_fastconformer_ctc_large",
            "decoder_type": "ctc",
            "device": "cpu",
            "acoustic_batch_size": 32,
            "beam_threshold": 7.0,
            "context_score": 3.0,
            "ctc_ali_token_weight": 0.5,
            "use_keywords": False,
            "spelling_separator": "_",
        },
        description="NeMo CTC transcription with context biasing for keyword spotting. Local model, no API key required.",
    )

    _register_alias(
        "assemblyai-transcription",
        AssemblyAITranscriptionPipeline,
        default_config={
            "model_version": "universal",
            "use_keywords": False,
        },
        description="AssemblyAI transcription pipeline with keyword boosting support. Requires API key from https://www.assemblyai.com/. Set `ASSEMBLYAI_API_KEY` env var.",
    )

    _register_alias(
        "elevenlabs-transcription",
        ElevenLabsTranscriptionPipeline,
        default_config={
            "model_id": "scribe_v2",
            "use_keywords": False,
        },
        description="ElevenLabs transcription pipeline with keyterm prompting support. Requires API key from https://elevenlabs.io/. Set `ELEVENLABS_API_KEY` env var.",
    )

    _register_alias(
        "pyannote-transcription",
        PyannoteTranscriptionPipeline,
        default_config={
            "out_dir": "./pyannote_transcription_results",
            "timeout": 3600,
            "request_buffer": 30,
            "model": "precision-3",
        },
        description="PyannoteAI transcription pipeline (ignores speaker attribution). Uses the precision-3 model with Nvidia Parakeet STT. Requires `PYANNOTE_TOKEN` env var from https://www.pyannote.ai/.",
    )

    ################# SPEECH GENERATION PIPELINES #################

    _register_alias(
        "argmax-oss-speech-generation",
        ArgmaxOpenSourceSpeechGenerationPipeline,
        default_config={
            "out_dir": "./speech_generation_results",
            "speaker": "aiden",
            "language": "english",
            "output_format": "wav",
            "seed": 10,
            "temperature": 0.9,
            "top_k": 50,
            "max_new_tokens": 245,
        },
        description=(
            "Argmax SDK (open source) speech-generation pipeline. Synthesizes audio from text prompts via "
            "`argmax-cli tts`. Clone/build under ARGMAX_OSS_CACHE_DIR (default ~/.cache/openbench/argmax-oss) "
            "unless `cli_path` is set. WER is computed by `SpeechGenerationWordErrorRate`, which transcribes "
            "the generated audio with WhisperKitPro / parakeet-v2 by default and compares against the original "
            "prompt; that metric requires `WHISPERKITPRO_CLI_PATH`."
        ),
    )

    ################# PII CLASSIFICATION PIPELINES #################
    # Two systems, each asked one question per sample: does this turn give out PII (mask the whole turn)?
    # Roblox PII v2 reads the turn as speaker `t` with the preceding turns as context; Privacy Filter tags
    # spans in the turn and the turn counts as PII when any span lands in it.

    _register_alias(
        "roblox-pii-v2",
        HuggingFaceRobloxPIIPipeline,
        default_config={"out_dir": "./pii_results"},
        description=(
            "Roblox PII classifier v2 (PyTorch, Hub weights). The turn is speaker `t`, up to 16 preceding turns are "
            "context, left-truncated to 512 tokens. PII when the giving head reaches 0.55 (the model card's threshold)."
        ),
    )

    _register_alias(
        "roblox-pii-v2-target-only",
        HuggingFaceRobloxPIIPipeline,
        default_config={"out_dir": "./pii_results", "context_turns": 0},
        description="Roblox PII classifier v2 on the turn alone, no context (how the model card ran the other models).",
    )

    _register_alias(
        "roblox-pii-v2-owner-context",
        HuggingFaceRobloxPIIPipeline,
        default_config={"out_dir": "./pii_results", "context_speakers": "target"},
        description=(
            "Roblox PII classifier v2 with only the speaker's own earlier turns as context (up to 16), what an "
            "owner-only recorder such as fil transcribes."
        ),
    )

    _register_alias(
        "roblox-pii-v2-owner-context-2",
        HuggingFaceRobloxPIIPipeline,
        default_config={"out_dir": "./pii_results", "context_speakers": "target", "context_turns": 2},
        description="Roblox PII classifier v2 with the speaker's own last 2 turns as context.",
    )

    _register_alias(
        "roblox-pii-v2-any-head",
        HuggingFaceRobloxPIIPipeline,
        default_config={
            "out_dir": "./pii_results",
            "heads": ["privacy_asking_for_pii", "privacy_giving_pii", "directing_users_off_platform"],
        },
        description=(
            "Roblox PII classifier v2, flagged when asking (0.60), giving (0.55) or off-platform (0.10) fires: "
            "the reading of the Roblox chat benchmark, whose unsafe rows carry any of the three labels."
        ),
    )

    _register_alias(
        "roblox-pii-v2-coreml",
        CoreMLRobloxPIIPipeline,
        default_config={"out_dir": "./pii_results"},
        description=(
            "Roblox PII classifier v2 as Sova ships it, the W8A16 Core ML bundle (macOS only; set SOVA_ROOT or "
            "`-pc model_path=...`). Same context and giving-head reading as `roblox-pii-v2`."
        ),
    )

    _register_alias(
        "openai-privacy-filter",
        HuggingFacePrivacyFilterPipeline,
        default_config={"out_dir": "./pii_results"},
        description=(
            "OpenAI Privacy Filter (1.5B total / 50M active, needs transformers >= 5.6) on the turn alone. PII when any "
            "token's most likely label is a PII label; the score is the highest per-token probability of not being `O`."
        ),
    )

    _register_alias(
        "openai-privacy-filter-context",
        HuggingFacePrivacyFilterPipeline,
        default_config={"out_dir": "./pii_results", "context_turns": 16},
        description=(
            "OpenAI Privacy Filter over up to 16 preceding turns plus the turn; only tokens inside the turn count, so it "
            "sees the same context as `roblox-pii-v2`."
        ),
    )

    _register_alias(
        "openai-privacy-filter-owner-context",
        HuggingFacePrivacyFilterPipeline,
        default_config={"out_dir": "./pii_results", "context_turns": 16, "context_speakers": "target"},
        description="OpenAI Privacy Filter with only the speaker's own earlier turns (up to 16) before the turn.",
    )

    _register_alias(
        "openai-privacy-filter-owner-context-2",
        HuggingFacePrivacyFilterPipeline,
        default_config={"out_dir": "./pii_results", "context_turns": 2, "context_speakers": "target"},
        description="OpenAI Privacy Filter with the speaker's own last 2 turns before the turn.",
    )

    for suffix, extra, note in (
        ("", {}, "on the turn alone"),
        (
            "-owner-context-2",
            {"context_turns": 2, "context_speakers": "target"},
            "with the speaker's own last 2 turns",
        ),
    ):
        _register_alias(
            f"openmed-privacy-filter-multilingual-v2{suffix}",
            HuggingFacePrivacyFilterPipeline,
            default_config={"out_dir": "./pii_results", "model_id": "OpenMed/privacy-filter-multilingual-v2", **extra},
            description=(
                "OpenMed's fine-tune of OpenAI Privacy Filter (54 types, 16 languages incl. Turkish; Apache-2.0; "
                f"trained on ai4privacy, Nemotron, Gretel and Privy-style data), {note}."
            ),
        )

    _register_alias(
        "regex-pii",
        RegexPIIPipeline,
        default_config={"out_dir": "./pii_results"},
        description="Patterns only (e-mail written or spoken, URL, phone, Luhn card, SSN, IBAN, IPv4): the free floor.",
    )

    _register_alias(
        "pii-tracer",
        HuggingFacePIITracerPipeline,
        default_config={"out_dir": "./pii_results"},
        description=(
            "Perplexity PII-Tracer (0.6B bidirectional Qwen3 encoder, MIT) on the turn alone; PII when its own "
            "constrained Viterbi keeps a span in the turn. Needs transformers >= 5.2."
        ),
    )

    _register_alias(
        "pii-tracer-owner-context-2",
        HuggingFacePIITracerPipeline,
        default_config={"out_dir": "./pii_results", "context_turns": 2, "context_speakers": "target"},
        description="PII-Tracer with the speaker's own last 2 turns before the turn.",
    )

    # GLiNER family: zero-shot span extractors, each given the PII labels of its own taxonomy (broad types such as
    # city, country or company are left out: the gold labels do not count them).
    gliner_v1_labels = [
        "person",
        "name",
        "email address",
        "phone number",
        "street address",
        "postal code",
        "social security number",
        "credit card number",
        "bank account number",
        "account number",
        "identity document number",
        "passport number",
        "driver's license number",
        "username",
        "password",
        "url",
        "date of birth",
        "license plate number",
    ]
    nvidia_labels = [
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "street_address",
        "postcode",
        "ssn",
        "credit_debit_card",
        "account_number",
        "customer_id",
        "user_name",
        "password",
        "url",
        "date_of_birth",
        "license_plate",
        "national_id",
    ]
    gliner2_pii_labels = [
        "person",
        "full_name",
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "address",
        "street_address",
        "postal_code",
        "government_id",
        "national_id_number",
        "passport_number",
        "drivers_license_number",
        "account_number",
        "card_number",
        "username",
        "account_id",
        "password",
        "secret",
        "date_of_birth",
    ]
    guard_labels = [
        "person",
        "first_name",
        "last_name",
        "address",
        "street",
        "postal_code",
        "email",
        "phone",
        "social_account",
        "messenger",
        "passport",
        "national_id",
        "document_id",
        "date_of_birth",
        "card_number",
    ]
    gliner_models = [
        (
            "gliner-pii-urchade-multi",
            "urchade/gliner_multi_pii-v1",
            "gliner",
            gliner_v1_labels,
            0.5,
            "GLiNER v1 multilingual PII fine-tune by GLiNER's author (Apache-2.0)",
        ),
        (
            "gliner-pii-nvidia",
            "nvidia/gliner-PII",
            "gliner",
            nvidia_labels,
            0.5,
            "NVIDIA GLiNER-PII on GLiNER large-v2.1, trained on Nemotron-PII (NVIDIA Open Model License)",
        ),
        (
            "gliner-pii-knowledgator-edge",
            "knowledgator/gliner-pii-edge-v1.0",
            "gliner",
            gliner_v1_labels,
            0.3,
            "Knowledgator GLiNER-PII edge (Apache-2.0)",
        ),
        (
            "gliner-pii-knowledgator-small",
            "knowledgator/gliner-pii-small-v1.0",
            "gliner",
            gliner_v1_labels,
            0.3,
            "Knowledgator GLiNER-PII small (Apache-2.0)",
        ),
        (
            "gliner-pii-knowledgator-base",
            "knowledgator/gliner-pii-base-v1.0",
            "gliner",
            gliner_v1_labels,
            0.3,
            "Knowledgator GLiNER-PII base (Apache-2.0)",
        ),
        (
            "gliner-pii-knowledgator-large",
            "knowledgator/gliner-pii-large-v1.0",
            "gliner",
            gliner_v1_labels,
            0.3,
            "Knowledgator GLiNER-PII large (Apache-2.0)",
        ),
        (
            "gliner2-pii-fastino",
            "fastino/gliner2-privacy-filter-PII-multi",
            "gliner2",
            gliner2_pii_labels,
            0.5,
            "Fastino GLiNER2-PII, 0.3B, 7 languages (Apache-2.0)",
        ),
        (
            "gliner-guard-omni",
            "hivetrace/gliner-guard-omni",
            "gliner2",
            guard_labels,
            0.4,
            "hivetrace GLiNER Guard omni, 0.3B, safety + PII (Apache-2.0)",
        ),
        (
            "gliner-guard-uni",
            "hivetrace/gliner-guard-uniencoder",
            "gliner2",
            guard_labels,
            0.4,
            "hivetrace GLiNER Guard uni-encoder, 0.15B (Apache-2.0)",
        ),
        (
            "gliner-guard-bi",
            "hivetrace/gliner-guard-biencoder",
            "gliner2",
            guard_labels,
            0.4,
            "hivetrace GLiNER Guard bi-encoder, 0.15B (Apache-2.0)",
        ),
    ]
    for alias, model_id, library, labels, threshold, description in gliner_models:
        base = {"out_dir": "./pii_results", "model_id": model_id, "library": library, "labels": labels}
        _register_alias(
            alias,
            GLiNERPIIPipeline,
            default_config={**base, "threshold": threshold},
            description=f"{description}; the turn alone, a span at or above {threshold} flags it.",
        )
        _register_alias(
            f"{alias}-owner-context-2",
            GLiNERPIIPipeline,
            default_config={**base, "threshold": threshold, "context_turns": 2, "context_speakers": "target"},
            description=f"{description}; the speaker's own last 2 turns before the turn.",
        )

    _register_alias(
        "constant-pii",
        ConstantSafetyPipeline,
        default_config={"out_dir": "./pii_results", "label": "unsafe"},
        description="Masks every turn: recall 1 at the dataset's PII share, the F1 floor both models must beat.",
    )

    ################# STREAMING TRANSCRIPTION PIPELINES #################

    _register_alias(
        "deepgram-streaming",
        DeepgramStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "realtime_resolution": 0.02,
            "model_version": "nova-3",
            "endpoint_url": "wss://api.deepgram.com/v1/listen?model={model_version}&channels={channels}&sample_rate={sample_rate}&encoding=linear16&interim_results=true",
        },
        description="Deepgram streaming transcription pipeline. Requires API key from https://www.deepgram.com/. Set `DEEPGRAM_API_KEY` env var.",
    )

    _register_alias(
        "fireworks-streaming",
        FireworksStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "ws://audio-streaming.us-virginia-1.direct.fireworks.ai/v1/audio/transcriptions/streaming",
            "model": "whisper-v3-turbo",
        },
        description="Fireworks streaming transcription pipeline. Requires API key from https://www.fireworks.ai/. Set `FIREWORKS_API_KEY` env var.",
    )

    _register_alias(
        "gladia-streaming",
        GladiaStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "https://api.gladia.io/v2/live",
        },
        description="Gladia streaming transcription pipeline. Requires API key from https://www.gladia.io/. Set `GLADIA_API_KEY` env var.",
    )

    _register_alias(
        "openai-streaming",
        OpenAIStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "realtime_resolution": 0.02,
            "endpoint_url": "https://api.openai.com/v1/realtime/transcription_sessions",
            "model": "gpt-4o-transcribe",
        },
        description="OpenAI streaming transcription pipeline. Requires API key from https://www.openai.com/. Set `OPENAI_API_KEY` env var.",
    )

    _register_alias(
        "assemblyai-streaming",
        AssemblyAIStreamingPipeline,
        default_config={
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "chunksize_ms": 50,
            "endpoint_url": "wss://streaming.assemblyai.com/v3/ws",
        },
        description="AssemblyAI streaming transcription pipeline. Requires API key from https://www.assemblyai.com/. Set `ASSEMBLYAI_API_KEY` env var.",
    )


register_pipeline_aliases()

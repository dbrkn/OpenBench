# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Word Error Rate for speech-generation pipelines.

The metric receives a `GeneratedAudio` hypothesis (path + duration), runs
ASR on the audio to obtain a `Transcript`, and then delegates to the
standard `WordErrorRate.compute_components` against the reference. This
keeps the TTS pipeline free of ASR concerns — including from its timing
budget — while still letting WER score generated audio.

Transcription can be specified either as a `TranscriptionConfig` instance
(resolved via `PipelineRegistry.create_pipeline_from_config`) or as a
pipeline alias string (resolved via `PipelineRegistry.create_pipeline`).
When nothing is supplied, the metric falls back to the
`whisperkit-large-v3-turbo` Argmax OSS alias so users don't have to wire
anything up.
"""

from pathlib import Path

import librosa
from argmaxtools.utils import get_logger

from ...dataset import TranscriptionSample
from ...pipeline import Pipeline
from ...pipeline.pipeline_registry import PipelineRegistry
from ...pipeline.transcription.common import TranscriptionConfig
from ...pipeline_prediction import GeneratedAudio, Transcript
from ...types import PipelineType
from ..metric import MetricOptions
from ..registry import MetricRegistry
from .word_error_metrics import WordErrorRate


logger = get_logger(__name__)

DEFAULT_TRANSCRIPTION_ALIAS = "whisperkit-large-v3-turbo"


def _build_transcription_sample(audio_path: str, language: str | None = None) -> TranscriptionSample:
    """Wrap a WAV on disk in a TranscriptionSample so a pipeline can consume it.

    The reference is a no-op `Transcript` — transcription pipelines only
    read it via `parse_input` for fields like keyword boosting (which we
    aren't using here), not as ground truth.

    `language` (when provided) is exposed via `TranscriptionSample.language`
    so a `force_language` transcription pipeline can hint the ASR with the
    dataset's target language instead of relying on auto-detection — important
    for multilingual TTS eval sets (e.g. Seed-TTS EN/ZH).
    """
    waveform, sample_rate = librosa.load(audio_path, sr=None)
    extra_info: dict = {}
    if language:
        extra_info["language"] = language
    return TranscriptionSample(
        audio_name=Path(audio_path).stem,
        waveform=waveform,
        sample_rate=int(sample_rate),
        reference=Transcript(words=[]),
        extra_info=extra_info,
    )


@MetricRegistry.register_metric(PipelineType.SPEECH_GENERATION, MetricOptions.WER)
class SpeechGenerationWordErrorRate(WordErrorRate):
    """WER metric for speech-generation pipelines.

    Args:
        transcription_config: Either a `TranscriptionConfig` instance (the
            matching `Pipeline` is built via
            `PipelineRegistry.create_pipeline_from_config`) or a pipeline
            alias string (resolved via `PipelineRegistry.create_pipeline`).
            If `None`, defaults to the `whisperkit-large-v3-turbo` Argmax OSS
            alias. The pipeline is built lazily on first use.
        **kwargs: Forwarded to `WordErrorRate.__init__`
            (e.g. `use_text_normalizer`, `english_spelling_mapping`).
    """

    def __init__(
        self,
        transcription_config: TranscriptionConfig | str | None = None,
        force_language: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if transcription_config is None:
            transcription_config = DEFAULT_TRANSCRIPTION_ALIAS
        if not isinstance(transcription_config, (TranscriptionConfig, str)):
            raise TypeError(
                "SpeechGenerationWordErrorRate.transcription_config must be a "
                "TranscriptionConfig instance or a pipeline alias string, got "
                f"{type(transcription_config).__name__}"
            )
        self._transcription_config = transcription_config
        # When True, the dataset's per-sample language is forwarded to the ASR
        # (via TranscriptionSample.language + the pipeline's force_language path)
        # rather than letting Whisper auto-detect — the right default for
        # multilingual TTS eval sets where the target language is known.
        self._force_language = force_language
        self._pipeline: Pipeline | None = None

    def _get_pipeline(self) -> Pipeline:
        if self._pipeline is not None:
            return self._pipeline

        spec = self._transcription_config
        # Resolve pipeline class first so we can validate pipeline_type before
        # paying the build cost (which for argmax-oss aliases means clone + swift build).
        if isinstance(spec, str):
            pipeline_class = PipelineRegistry.get_pipeline_class(spec)
            label = f"alias={spec!r}"
        else:
            pipeline_class = PipelineRegistry.get_pipeline_class_for_config(type(spec))
            label = type(spec).__name__

        if pipeline_class.pipeline_type != PipelineType.TRANSCRIPTION:
            raise ValueError(
                f"{pipeline_class.__name__} has pipeline_type {pipeline_class.pipeline_type}, expected TRANSCRIPTION."
            )

        logger.info("Building transcription pipeline for speech-generation WER: %s", label)
        if isinstance(spec, str):
            self._pipeline = PipelineRegistry.create_pipeline(spec)
        else:
            self._pipeline = pipeline_class(spec)

        # Force the ASR to honour the dataset's per-sample language. The flag
        # lives on TranscriptionConfig; pipelines that don't support forcing
        # (e.g. NeMo/AssemblyAI) just warn and ignore it.
        if self._force_language and hasattr(self._pipeline.config, "force_language"):
            self._pipeline.config.force_language = True
        return self._pipeline

    def compute_components(self, reference: Transcript, hypothesis: GeneratedAudio, **kwargs) -> dict[str, int]:
        """Transcribe the generated audio, then delegate to WER.

        Args:
            reference: Reference transcript built from the original prompt.
            hypothesis: Generated audio (path + duration) produced by a TTS pipeline.
        """
        # The runner unpacks the dataset sample's extra_info into kwargs, so the
        # target language (when the dataset provides one) arrives here.
        language = kwargs.get("language") if self._force_language else None
        sample = _build_transcription_sample(hypothesis.audio_path, language=language)
        pipeline_output = self._get_pipeline()(sample)
        hypothesis_transcript: Transcript = pipeline_output.prediction
        logger.debug("TTS WER hypothesis transcript: " + hypothesis_transcript.get_transcript_string()[:120] + "...")
        return super().compute_components(reference, hypothesis_transcript, **kwargs)

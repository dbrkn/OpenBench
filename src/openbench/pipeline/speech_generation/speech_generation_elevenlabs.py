# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""
Speech generation pipeline using ElevenLabs TTS API.

Generates TTS audio from text prompts via ElevenLabs,
then transcribes the generated audio back to text using
WhisperKitPro (Parakeet) for WER evaluation against the
original prompt.
"""

import os
import time
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_base import BaseSample
from ...dataset.dataset_speech_generation import SpeechGenerationSample
from ...engine.whisperkitpro_engine import (
    WhisperKitPro,
    WhisperKitProConfig,
    WhisperKitProInput,
)
from ...pipeline_prediction import Transcript
from ..base import (
    Pipeline,
    PipelineConfig,
    PipelineOutput,
    PipelineType,
    register_pipeline,
)
from .common import SpeechGenerationOutput

logger = get_logger(__name__)

TEMP_TTS_AUDIO_DIR = Path("./temp_tts_audio")


class ElevenLabsSpeechGenerationConfig(PipelineConfig):
    """Config for the ElevenLabs speech generation pipeline."""

    # ElevenLabs TTS parameters
    api_key: str | None = Field(
        default=None,
        description=(
            "ElevenLabs API key. Falls back to "
            "ELEVENLABS_API_KEY env var."
        ),
    )
    voice_id: str = Field(
        default="JBFqnCBsd6RMkjVDRZzb",
        description="ElevenLabs voice ID.",
    )
    model_id: str = Field(
        default="eleven_v3", #"eleven_multilingual_v2",
        description="ElevenLabs model ID.",
    )
    output_format: str = Field(
        default="mp3_44100_128",
        description="ElevenLabs output audio format.",
    )

    # Transcription parameters (WhisperKitPro / Parakeet)
    transcription_cli_path: str = Field(
        ...,
        description=(
            "Path to the whisperkit-cli binary "
            "used for transcription."
        ),
    )
    transcription_repo_id: str | None = Field(
        default=None,
        description=(
            "HuggingFace repo ID for transcription "
            "model (e.g. argmaxinc/parakeetkit-pro)."
        ),
    )
    transcription_model_variant: str | None = Field(
        default=None,
        description=(
            "Model variant folder within the repo "
            "(e.g. nvidia_parakeet-v2_476MB)."
        ),
    )
    transcription_model_path: str | None = Field(
        default=None,
        description=(
            "Local path to ASR model dir. "
            "Overrides repo_id/model_variant."
        ),
    )
    transcription_word_timestamps: bool = Field(
        default=True,
        description="Include word timestamps.",
    )
    transcription_chunking_strategy: str = Field(
        default="vad",
        description="Chunking strategy (none or vad).",
    )

    keep_generated_audio: bool = Field(
        default=False,
        description=(
            "If True, keep the generated TTS audio "
            "files instead of deleting them."
        ),
    )


class ElevenLabsSpeechGenerationInput(BaseModel):
    """Input for the ElevenLabs speech generation pipeline."""

    text: str = Field(
        ...,
        description="Text prompt to generate speech from.",
    )
    audio_name: str = Field(
        ...,
        description=(
            "Unique identifier for this sample "
            "(used for temp file naming)."
        ),
    )


@register_pipeline
class ElevenLabsSpeechGenerationPipeline(Pipeline):
    """Speech generation pipeline using ElevenLabs TTS API.

    This pipeline:
    1. Generates audio from text via ElevenLabs text-to-speech API
    2. Transcribes audio via WhisperKitPro engine (Parakeet)
    3. Returns transcription as Transcript for WER eval
    4. Cleans up temporary audio and report files
    """

    _config_class = ElevenLabsSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(
        self,
    ) -> Callable[[ElevenLabsSpeechGenerationInput], Transcript]:
        config = self.config
        pipeline_ref = self

        transcription_engine = self._build_transcription_engine()

        api_key = config.api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key must be provided "
                "via config or ELEVENLABS_API_KEY env var."
            )

        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        def generate_and_transcribe(
            input: ElevenLabsSpeechGenerationInput,
        ) -> Transcript:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            ext = config.output_format.split("_")[0]
            audio_path = TEMP_TTS_AUDIO_DIR / f"{input.audio_name}.{ext}"

            # -- Step 1: Generate audio via ElevenLabs API --
            audio_iter = client.text_to_speech.convert(
                text=input.text,
                voice_id=config.voice_id,
                model_id=config.model_id,
                output_format=config.output_format,
            )

            with open(audio_path, "wb") as f:
                for chunk in audio_iter:
                    f.write(chunk)

            if not audio_path.exists() or audio_path.stat().st_size == 0:
                raise RuntimeError(
                    "ElevenLabs TTS failed: audio file "
                    f"missing or empty at {audio_path}"
                )

            logger.info(
                f"Generated ElevenLabs TTS audio: {audio_path}"
            )

            # -- Step 2: Read audio duration --
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                pipeline_ref._last_generated_duration = info.duration
            except Exception as e:
                logger.warning(f"Audio duration read failed: {e}")
                pipeline_ref._last_generated_duration = None

            # -- Step 3: Transcribe via WhisperKitPro (Parakeet) --
            engine_input = WhisperKitProInput(
                audio_path=audio_path,
                keep_audio=config.keep_generated_audio,
            )
            engine_output = transcription_engine(engine_input)

            # -- Step 4: Parse transcription report --
            json_path = engine_output.json_report_path
            if json_path.exists():
                import json

                with json_path.open("r") as f:
                    data = json.load(f)
                all_words, all_starts, all_ends = [], [], []
                for seg in data.get("segments", []):
                    for w in seg.get("words", []):
                        all_words.append(w["word"])
                        if "start" in w:
                            all_starts.append(w["start"])
                        if "end" in w:
                            all_ends.append(w["end"])
                transcript = Transcript.from_words_info(
                    words=all_words,
                    start=all_starts if all_starts else None,
                    end=all_ends if all_ends else None,
                )
                json_path.unlink(missing_ok=True)
                srt_path = engine_output.srt_report_path
                if srt_path:
                    srt_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "Transcription report not found "
                    f"at {json_path}"
                )

            text_preview = transcript.get_transcript_string()[:100]
            logger.info(f"Transcription: {text_preview}...")

            return transcript

        return generate_and_transcribe

    def _build_transcription_engine(self) -> WhisperKitPro:
        """Create WhisperKitPro engine for transcription (Parakeet)."""
        config = self.config

        import coremltools as ct

        compute = ct.ComputeUnit.CPU_AND_NE
        engine_config = WhisperKitProConfig(
            repo_id=config.transcription_repo_id,
            model_variant=config.transcription_model_variant,
            model_dir=config.transcription_model_path,
            word_timestamps=config.transcription_word_timestamps,
            chunking_strategy=config.transcription_chunking_strategy,
            audio_encoder_compute_units=compute,
            text_decoder_compute_units=compute,
        )

        return WhisperKitPro(
            cli_path=config.transcription_cli_path,
            transcription_config=engine_config,
        )

    def __call__(self, input_sample: BaseSample) -> PipelineOutput:
        """Run pipeline and set generated audio duration."""
        self._last_generated_duration: float | None = None
        parsed_input = self.parse_input(input_sample)
        start_time = time.perf_counter()
        output = self.pipeline(parsed_input)
        end_time = time.perf_counter()
        prediction_time = end_time - start_time
        parsed_output = self.parse_output(output)
        if parsed_output.prediction_time is None:
            parsed_output.prediction_time = prediction_time

        dur = self._last_generated_duration
        logger.debug(f"Generated audio duration: {dur}s")
        is_sg = isinstance(input_sample, SpeechGenerationSample)
        if is_sg and dur is not None:
            input_sample.generated_audio_duration = dur
            dur_val = input_sample.generated_audio_duration
            logger.debug(f"Set sample duration to {dur_val}s")

        return parsed_output

    def parse_input(
        self, input_sample: SpeechGenerationSample
    ) -> ElevenLabsSpeechGenerationInput:
        """Extract text prompt from the sample."""
        text = input_sample.reference.get_transcript_string()
        return ElevenLabsSpeechGenerationInput(
            text=text,
            audio_name=input_sample.audio_name,
        )

    def parse_output(self, output: Transcript) -> SpeechGenerationOutput:
        """Wrap transcription into output."""
        return SpeechGenerationOutput(prediction=output)

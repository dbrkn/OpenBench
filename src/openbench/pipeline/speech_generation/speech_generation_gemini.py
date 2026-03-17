# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""
Speech generation pipeline using Google Gemini TTS API.

Generates TTS audio from text prompts via Google Cloud
Text-to-Speech, then transcribes the generated audio back
to text using WhisperKitPro (Parakeet) for WER evaluation
against the original prompt.
"""

import time
from pathlib import Path
from typing import Callable

from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_base import BaseSample
from ...dataset.dataset_speech_generation import (
    SpeechGenerationSample,
)
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


class GeminiSpeechGenerationConfig(PipelineConfig):
    """Config for the Gemini speech generation pipeline."""

    # Google Cloud TTS parameters
    project_id: str | None = Field(
        default=None,
        description=(
            "Google Cloud project ID. Falls back to "
            "GOOGLE_CLOUD_PROJECT env var."
        ),
    )
    voice_name: str = Field(
        default="Charon",
        description="Google Cloud TTS voice name.",
    )
    language_code: str = Field(
        default="en-US",
        description="BCP-47 language code.",
    )
    model_name: str = Field(
        default="gemini-2.5-pro-tts",
        description="Google Cloud TTS model name.",
    )
    prompt: str | None = Field(
        default=None,
        description=(
            "Styling instructions for how to "
            "synthesize the speech."
        ),
    )
    audio_encoding: str = Field(
        default="MP3",
        description=(
            "Audio encoding format "
            "(MP3, LINEAR16, OGG_OPUS, MULAW, ALAW)."
        ),
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


ENCODING_TO_EXT = {
    "MP3": "mp3",
    "LINEAR16": "wav",
    "OGG_OPUS": "ogg",
    "MULAW": "wav",
    "ALAW": "wav",
}


class GeminiSpeechGenerationInput(BaseModel):
    """Input for the Gemini speech generation pipeline."""

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
class GeminiSpeechGenerationPipeline(Pipeline):
    """Speech generation pipeline using Google Gemini TTS.

    This pipeline:
    1. Generates audio from text via Google Cloud TTS
    2. Transcribes audio via WhisperKitPro engine (Parakeet)
    3. Returns transcription as Transcript for WER eval
    4. Cleans up temporary audio and report files
    """

    _config_class = GeminiSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(
        self,
    ) -> Callable[[GeminiSpeechGenerationInput], Transcript]:
        config = self.config
        pipeline_ref = self

        transcription_engine = self._build_transcription_engine()

        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()

        encoding_enum = getattr(
            texttospeech.AudioEncoding,
            config.audio_encoding,
        )

        def generate_and_transcribe(
            input: GeminiSpeechGenerationInput,
        ) -> Transcript:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

            ext = ENCODING_TO_EXT.get(
                config.audio_encoding, "mp3"
            )
            audio_path = (
                TEMP_TTS_AUDIO_DIR
                / f"{input.audio_name}.{ext}"
            )

            # -- Step 1: Generate audio via Google Cloud TTS --
            synth_kwargs = {"text": input.text}
            if config.prompt is not None:
                synth_kwargs["prompt"] = config.prompt

            synthesis_input = texttospeech.SynthesisInput(
                **synth_kwargs
            )

            voice = texttospeech.VoiceSelectionParams(
                language_code=config.language_code,
                name=config.voice_name,
                model_name=config.model_name,
            )

            audio_config = texttospeech.AudioConfig(
                audio_encoding=encoding_enum,
            )

            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )

            with open(audio_path, "wb") as f:
                f.write(response.audio_content)

            if (
                not audio_path.exists()
                or audio_path.stat().st_size == 0
            ):
                raise RuntimeError(
                    "Gemini TTS failed: audio file "
                    f"missing or empty at {audio_path}"
                )

            logger.info(
                f"Generated Gemini TTS audio: {audio_path}"
            )

            # -- Step 2: Read audio duration --
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                pipeline_ref._last_generated_duration = (
                    info.duration
                )
            except Exception as e:
                logger.warning(
                    f"Audio duration read failed: {e}"
                )
                pipeline_ref._last_generated_duration = None

            # -- Step 3: Transcribe via WhisperKitPro --
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
                    start=(
                        all_starts if all_starts else None
                    ),
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

            text_preview = (
                transcript.get_transcript_string()[:100]
            )
            logger.info(f"Transcription: {text_preview}...")

            return transcript

        return generate_and_transcribe

    def _build_transcription_engine(self) -> WhisperKitPro:
        """Create WhisperKitPro engine for transcription."""
        config = self.config

        import coremltools as ct

        compute = ct.ComputeUnit.CPU_AND_NE
        engine_config = WhisperKitProConfig(
            repo_id=config.transcription_repo_id,
            model_variant=config.transcription_model_variant,
            model_dir=config.transcription_model_path,
            word_timestamps=(
                config.transcription_word_timestamps
            ),
            chunking_strategy=(
                config.transcription_chunking_strategy
            ),
            audio_encoder_compute_units=compute,
            text_decoder_compute_units=compute,
        )

        return WhisperKitPro(
            cli_path=config.transcription_cli_path,
            transcription_config=engine_config,
        )

    def __call__(
        self, input_sample: BaseSample
    ) -> PipelineOutput:
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
        is_sg = isinstance(
            input_sample, SpeechGenerationSample
        )
        if is_sg and dur is not None:
            input_sample.generated_audio_duration = dur
            dur_val = input_sample.generated_audio_duration
            logger.debug(
                f"Set sample duration to {dur_val}s"
            )

        return parsed_output

    def parse_input(
        self, input_sample: SpeechGenerationSample
    ) -> GeminiSpeechGenerationInput:
        """Extract text prompt from the sample."""
        text = input_sample.reference.get_transcript_string()
        return GeminiSpeechGenerationInput(
            text=text,
            audio_name=input_sample.audio_name,
        )

    def parse_output(
        self, output: Transcript
    ) -> SpeechGenerationOutput:
        """Wrap transcription into output."""
        return SpeechGenerationOutput(prediction=output)

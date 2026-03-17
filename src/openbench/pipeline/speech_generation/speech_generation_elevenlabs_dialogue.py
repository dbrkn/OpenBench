# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""
Speech generation pipeline using ElevenLabs text-to-dialogue API.

Generates multi-speaker conversational audio from dialogue turns
via ElevenLabs, then transcribes the generated audio back to text
using WhisperKitPro (Parakeet) for WER evaluation against the
original dialogue text.

Long dialogues that exceed the API character limit are automatically
split into chunks, generated separately, and stitched together with
configurable silence gaps between chunks.
"""

import io
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
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

DEFAULT_SPEAKER_VOICE_MAP = {
    "doctor": "9BWtsMINqrJLrRacOk9x",
    "patient": "IKne3meq5aSn9XLyUdCD",
    "assistant": "pFZP5JQG7iQjIQuC4Bku",
}

MAX_CHARS_PER_CHUNK = 4500


def _chunk_dialogue_turns(
    turns: list[dict],
    speaker_voice_map: dict[str, str],
    default_voice_id: str,
    max_chars: int = MAX_CHARS_PER_CHUNK,
) -> list[list[dict]]:
    """Split dialogue turns into chunks that fit under the char limit.

    Each chunk is a list of ElevenLabs input dicts ({text, voice_id}).
    Splits on turn boundaries so no individual turn is broken.
    """
    chunks: list[list[dict]] = []
    current_chunk: list[dict] = []
    current_chars = 0

    for turn in turns:
        speaker = turn.get("speaker", "")
        voice_id = speaker_voice_map.get(speaker, default_voice_id)
        entry = {"text": turn["text"], "voice_id": voice_id}
        turn_chars = len(turn["text"])

        if current_chars + turn_chars > max_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0

        current_chunk.append(entry)
        current_chars += turn_chars

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _stitch_audio_files(
    chunk_paths: list[Path],
    output_path: Path,
    silence_duration: float = 0.75,
) -> None:
    """Concatenate audio files with silence gaps between them.

    Decodes each chunk, inserts silence, and writes as WAV.
    """
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for i, path in enumerate(chunk_paths):
        segment = AudioSegment.from_file(str(path))
        if i > 0:
            silence_ms = int(silence_duration * 1000)
            combined += AudioSegment.silent(
                duration=silence_ms,
                frame_rate=segment.frame_rate,
            )
        combined += segment

    combined.export(str(output_path), format="wav")


class ElevenLabsDialogueGenerationConfig(PipelineConfig):
    """Config for the ElevenLabs dialogue generation pipeline."""

    api_key: str | None = Field(
        default=None,
        description=(
            "ElevenLabs API key. Falls back to "
            "ELEVENLABS_API_KEY env var."
        ),
    )
    model_id: str = Field(
        default="eleven_v3",
        description="ElevenLabs model ID for dialogue.",
    )
    speaker_voice_map: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SPEAKER_VOICE_MAP),
        description=(
            "Mapping of speaker names to ElevenLabs voice IDs. "
            "Speakers not in this map use the default_voice_id."
        ),
    )
    default_voice_id: str = Field(
        default="9BWtsMINqrJLrRacOk9x",
        description="Fallback voice ID for unmapped speakers.",
    )
    max_chars_per_chunk: int = Field(
        default=MAX_CHARS_PER_CHUNK,
        description=(
            "Max characters per API call. Dialogues exceeding "
            "this are split into multiple chunks."
        ),
    )
    chunk_silence_duration: float = Field(
        default=0.75,
        description=(
            "Silence duration (seconds) inserted between "
            "stitched chunks. Range 0.5-1.0 recommended."
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


class ElevenLabsDialogueGenerationInput(BaseModel):
    """Input for the ElevenLabs dialogue generation pipeline."""

    text: str = Field(
        ...,
        description="Full concatenated dialogue text (for reference).",
    )
    dialogue: list[dict] = Field(
        ...,
        description="List of dialogue turns with speaker and text.",
    )
    audio_name: str = Field(
        ...,
        description=(
            "Unique identifier for this sample "
            "(used for temp file naming)."
        ),
    )


@register_pipeline
class ElevenLabsDialogueGenerationPipeline(Pipeline):
    """Speech generation pipeline using ElevenLabs text-to-dialogue API.

    This pipeline:
    1. Chunks dialogue turns to fit API character limits
    2. Generates audio per chunk via ElevenLabs text_to_dialogue
    3. Saves chunk audio files under temp_tts_audio/chunks/
    4. Stitches chunks with silence gaps into final audio
    5. Transcribes audio via WhisperKitPro engine (Parakeet)
    6. Returns transcription as Transcript for WER eval
    """

    _config_class = ElevenLabsDialogueGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(
        self,
    ) -> Callable[[ElevenLabsDialogueGenerationInput], Transcript]:
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

        def _generate_chunk(
            chunk_inputs: list[dict],
            chunk_path: Path,
        ) -> Path:
            """Generate audio for a single chunk of dialogue turns."""
            audio_iter = client.text_to_dialogue.convert(
                inputs=chunk_inputs,
            )
            with open(chunk_path, "wb") as f:
                for data in audio_iter:
                    f.write(data)

            if not chunk_path.exists() or chunk_path.stat().st_size == 0:
                raise RuntimeError(
                    "ElevenLabs dialogue TTS failed: "
                    f"chunk empty at {chunk_path}"
                )
            return chunk_path

        def generate_and_transcribe(
            input: ElevenLabsDialogueGenerationInput,
        ) -> Transcript:
            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            chunks_dir = TEMP_TTS_AUDIO_DIR / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            # -- Step 1: Chunk dialogue turns --
            chunks = _chunk_dialogue_turns(
                input.dialogue,
                config.speaker_voice_map,
                config.default_voice_id,
                max_chars=config.max_chars_per_chunk,
            )

            total_turns = sum(len(c) for c in chunks)
            logger.info(
                f"Generating dialogue for {input.audio_name}: "
                f"{total_turns} turns in {len(chunks)} chunk(s)"
            )

            # -- Step 2: Generate audio per chunk --
            chunk_paths: list[Path] = []
            for i, chunk_inputs in enumerate(chunks):
                chunk_chars = sum(len(e["text"]) for e in chunk_inputs)
                chunk_path = (
                    chunks_dir
                    / f"{input.audio_name}_chunk_{i}.mp3"
                )
                logger.info(
                    f"  Chunk {i}: {len(chunk_inputs)} turns, "
                    f"{chunk_chars} chars -> {chunk_path.name}"
                )
                _generate_chunk(chunk_inputs, chunk_path)
                chunk_paths.append(chunk_path)

            # -- Step 3: Stitch or use single chunk --
            if len(chunk_paths) == 1:
                audio_path = (
                    TEMP_TTS_AUDIO_DIR / f"{input.audio_name}.mp3"
                )
                chunk_paths[0].rename(audio_path)
            else:
                audio_path = (
                    TEMP_TTS_AUDIO_DIR / f"{input.audio_name}.wav"
                )
                _stitch_audio_files(
                    chunk_paths,
                    audio_path,
                    silence_duration=config.chunk_silence_duration,
                )
                logger.info(
                    f"Stitched {len(chunk_paths)} chunks -> "
                    f"{audio_path.name}"
                )

            # -- Step 4: Read audio duration --
            try:
                info = sf.info(str(audio_path))
                pipeline_ref._last_generated_duration = info.duration
            except Exception as e:
                logger.warning(f"Audio duration read failed: {e}")
                pipeline_ref._last_generated_duration = None

            # -- Step 5: Transcribe via WhisperKitPro (Parakeet) --
            engine_input = WhisperKitProInput(
                audio_path=audio_path,
                keep_audio=config.keep_generated_audio,
            )
            engine_output = transcription_engine(engine_input)

            # -- Step 6: Parse transcription report --
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
    ) -> ElevenLabsDialogueGenerationInput:
        """Extract dialogue and text from the sample."""
        text = input_sample.reference.get_transcript_string()
        dialogue = input_sample.extra_info.get("dialogue", [])

        if not dialogue:
            raise ValueError(
                f"Sample {input_sample.audio_name} has no dialogue data. "
                "This pipeline requires a dialogue dataset."
            )

        return ElevenLabsDialogueGenerationInput(
            text=text,
            dialogue=dialogue,
            audio_name=input_sample.audio_name,
        )

    def parse_output(self, output: Transcript) -> SpeechGenerationOutput:
        """Wrap transcription into output."""
        return SpeechGenerationOutput(prediction=output)

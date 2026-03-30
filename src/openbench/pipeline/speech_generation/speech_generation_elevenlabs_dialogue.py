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
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import tqdm
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
    "doctor": "JBFqnCBsd6RMkjVDRZzb",
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
        default="JBFqnCBsd6RMkjVDRZzb",
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
    transcription_cli_path: str | None = Field(
        default=None,
        description=(
            "Path to the whisperkit-cli binary "
            "used for transcription. Required unless "
            "generate_only=True."
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

    concurrency: int = Field(
        default=1,
        description=(
            "Number of concurrent TTS API calls. "
            "Values > 1 use a thread pool for I/O-bound generation."
        ),
    )
    warm_start: bool = Field(
        default=False,
        description=(
            "If True, skip TTS generation for samples that already "
            "have audio files in the output directory and go straight "
            "to transcription."
        ),
    )
    audio_output_dir: str | None = Field(
        default=None,
        description=(
            "Persistent directory for saving generated audio files "
            "with sample_id names. If None, uses ./temp_tts_audio."
        ),
    )
    generate_only: bool = Field(
        default=False,
        description=(
            "If True, only generate TTS audio without "
            "transcription. Requires audio_output_dir."
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

        if not config.generate_only:
            if not config.transcription_cli_path:
                raise ValueError(
                    "transcription_cli_path is required "
                    "unless generate_only=True."
                )
            transcription_engine = (
                self._build_transcription_engine()
            )
        else:
            transcription_engine = None

        api_key = config.api_key or os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key must be provided "
                "via config or ELEVENLABS_API_KEY env var."
            )

        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=api_key)

        needs_persistent_dir = (
            config.warm_start
            or config.concurrency > 1
            or config.generate_only
        )
        if config.audio_output_dir:
            output_dir = Path(config.audio_output_dir).resolve()
        elif needs_persistent_dir:
            raise ValueError(
                "warm_start, concurrency > 1, or generate_only "
                "requires audio_output_dir to be set to an "
                "absolute path. The default relative path "
                "changes with each run's timestamped output "
                "directory."
            )
        else:
            output_dir = TEMP_TTS_AUDIO_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks_dir = output_dir / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Audio output directory (absolute): {output_dir}"
        )

        staging_dir = output_dir / ".staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        for d in (chunks_dir, staging_dir):
            stale = [
                f for f in d.iterdir()
                if f.is_file() and f.stat().st_size == 0
            ]
            for f in stale:
                f.unlink()
        for f in staging_dir.iterdir():
            if f.is_file():
                f.unlink()

        def _generate_chunk(
            chunk_inputs: list[dict],
            chunk_path: Path,
        ) -> Path:
            """Generate audio for a single chunk of dialogue turns.

            Writes to a staging file first, then moves to the
            final path so chunks/ only contains completed files.
            """
            audio_iter = client.text_to_dialogue.convert(
                inputs=chunk_inputs,
            )
            fd, tmp = tempfile.mkstemp(
                suffix=".mp3", dir=staging_dir
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    for data in audio_iter:
                        f.write(data)
                tmp_path = Path(tmp)
                if tmp_path.stat().st_size == 0:
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "ElevenLabs dialogue TTS returned "
                        f"empty audio for {chunk_path.name}"
                    )
                tmp_path.rename(chunk_path)
            except Exception:
                Path(tmp).unlink(missing_ok=True)
                raise
            return chunk_path

        def _find_existing_audio(audio_name: str) -> Path | None:
            """Check if audio already exists for warm start."""
            for ext in (".wav", ".mp3"):
                candidate = output_dir / f"{audio_name}{ext}"
                if candidate.exists() and candidate.stat().st_size > 0:
                    return candidate
            return None

        def _generate_audio(
            input: ElevenLabsDialogueGenerationInput,
        ) -> Path:
            """Generate TTS audio for a single sample. Returns audio path."""
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

            chunk_paths: list[Path] = []
            for i, chunk_inputs in enumerate(chunks):
                chunk_chars = sum(
                    len(e["text"]) for e in chunk_inputs
                )
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

            if len(chunk_paths) == 1:
                audio_path = (
                    output_dir / f"{input.audio_name}.mp3"
                )
                shutil.copy2(chunk_paths[0], audio_path)
            else:
                audio_path = (
                    output_dir / f"{input.audio_name}.wav"
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

            return audio_path

        def _transcribe_audio(audio_path: Path) -> Transcript:
            """Transcribe an audio file and return Transcript."""
            engine_input = WhisperKitProInput(
                audio_path=audio_path,
                keep_audio=config.keep_generated_audio,
            )
            engine_output = transcription_engine(engine_input)

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

        def generate_and_transcribe(
            input: ElevenLabsDialogueGenerationInput,
        ) -> Transcript:
            # Skip generation when warm_start is enabled or when
            # pre_generate_all already created the audio file
            # (concurrency > 1 triggers pre_generate_all in the runner)
            should_check = (
                config.warm_start
                or config.concurrency > 1
                or config.generate_only
            )
            existing = (
                _find_existing_audio(input.audio_name)
                if should_check
                else None
            )

            if existing:
                logger.info(
                    f"Reusing existing audio {existing} "
                    f"for {input.audio_name}"
                )
                audio_path = existing
            else:
                audio_path = _generate_audio(input)

            try:
                info = sf.info(str(audio_path))
                pipeline_ref._last_generated_duration = info.duration
            except Exception as e:
                logger.warning(
                    f"Audio duration read failed: {e}"
                )
                pipeline_ref._last_generated_duration = None

            if config.generate_only:
                logger.info(
                    f"generate_only: skipping transcription "
                    f"for {input.audio_name}"
                )
                return Transcript.from_words_info(
                    words=input.text.split()
                )

            return _transcribe_audio(audio_path)

        # Store references for pre_generate_all
        self._generate_audio = _generate_audio
        self._find_existing_audio = _find_existing_audio
        self._output_dir = output_dir

        return generate_and_transcribe

    def pre_generate_all(
        self, samples: list[SpeechGenerationSample]
    ) -> None:
        """Pre-generate all TTS audio concurrently.

        Logs warm-start stats and shows a progress bar.
        """
        config = self.config
        if config.concurrency <= 1:
            return

        total = len(samples)
        skipped = 0
        no_dialogue = 0
        inputs = []

        for sample in samples:
            dialogue = sample.extra_info.get("dialogue", [])
            if not dialogue:
                no_dialogue += 1
                continue

            should_check = (
                config.warm_start or config.generate_only
            )
            if should_check:
                existing = self._find_existing_audio(
                    sample.audio_name
                )
                if existing:
                    skipped += 1
                    continue

            text = sample.reference.get_transcript_string()
            inputs.append(
                ElevenLabsDialogueGenerationInput(
                    text=text,
                    dialogue=dialogue,
                    audio_name=sample.audio_name,
                )
            )

        to_generate = len(inputs)
        logger.info(
            f"Audio generation plan: {total} total, "
            f"{skipped} already exist, "
            f"{no_dialogue} have no dialogue, "
            f"{to_generate} to generate "
            f"(concurrency={config.concurrency})"
        )

        if not inputs:
            logger.info("Nothing to generate.")
            return

        completed = 0
        failed = 0
        pbar = tqdm.tqdm(
            total=to_generate,
            desc="Generating TTS audio",
            unit="sample",
        )

        with ThreadPoolExecutor(
            max_workers=config.concurrency
        ) as executor:
            futures = {
                executor.submit(self._generate_audio, inp): inp
                for inp in inputs
            }
            for future in as_completed(futures):
                inp = futures[future]
                try:
                    path = future.result()
                    completed += 1
                    pbar.set_postfix_str(
                        f"last={inp.audio_name}"
                    )
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Failed: {inp.audio_name}: {e}"
                    )
                    pbar.close()
                    raise
                finally:
                    pbar.update(1)

        pbar.close()
        logger.info(
            f"Generation complete: {completed} succeeded, "
            f"{failed} failed, {skipped} reused"
        )

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
        """Run pipeline and set generated audio duration.

        When warm_start is enabled, audio generation is skipped for
        samples that already have audio files in audio_output_dir.
        """
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

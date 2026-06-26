# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

from pathlib import Path

import numpy as np
from typing_extensions import TypedDict

from ..pipeline_prediction import Transcript
from .dataset_base import BaseDataset, BaseSample


class SpeechGenerationExtraInfo(TypedDict, total=False):
    """Extra info for speech generation samples.

    For voice-cloning datasets (e.g. seedTTS), `ref_audio` points at the
    target-speaker prompt clip and `ref_text` is its transcript. The prototype
    pipeline consumes these to drive `tts-cli --mode voice_clone`, and the SIM
    metric compares the generated clip against `ref_audio`.
    """

    language: str
    ref_audio: str
    ref_text: str


class SpeechGenerationRow(TypedDict):
    """Expected row structure for speech generation.

    Requires 'text' (the prompt string). No audio needed.
    """

    text: str


class SpeechGenerationSample(BaseSample[Transcript, SpeechGenerationExtraInfo]):
    """Sample for speech-generation tasks.

    The reference `Transcript` is constructed from the prompt text. The
    pipeline synthesizes audio from this prompt and returns a
    `GeneratedAudio` prediction; the WER metric transcribes that audio
    and compares against this reference.
    """

    @property
    def text(self) -> str:
        """The original text prompt."""
        return self.reference.get_transcript_string()


class SpeechGenerationDataset(BaseDataset[SpeechGenerationSample]):
    """Dataset for speech-generation pipelines.

    Expects column: 'text' (the prompt string). No audio column is
    required — audio is produced by the pipeline, and a dummy waveform is
    supplied to satisfy the base sample structure.

    For voice-cloning eval datasets that also ship a reference clip in an
    `audio` column, that clip is loaded into the sample waveform (and its
    transcript exposed as `ref_text`) so cloning pipelines can use it as the
    target speaker. The runner still reads the generated-audio duration off the
    pipeline output, not this input audio.
    """

    _expected_columns = ["text"]
    _sample_class = SpeechGenerationSample

    @staticmethod
    def _has_reference_audio(row: dict) -> bool:
        audio = row.get("audio")
        return isinstance(audio, dict) and audio.get("array") is not None

    def _extract_audio_info(self, row: dict) -> tuple[str, np.ndarray, int]:
        """Load a reference clip if present, else a placeholder waveform."""
        audio_name = f"sample_{row['idx']}"
        # Use audio_name from the row if available
        if "audio_name" in row and row["audio_name"]:
            audio_name = str(row["audio_name"])

        if self._has_reference_audio(row):
            audio = row["audio"]
            if audio.get("path"):
                audio_name = Path(audio["path"]).stem
            return audio_name, np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])

        dummy_waveform = np.zeros(1, dtype=np.float32)
        dummy_sample_rate = 16000
        return audio_name, dummy_waveform, dummy_sample_rate

    def prepare_sample(self, row: SpeechGenerationRow) -> tuple[Transcript, SpeechGenerationExtraInfo]:
        """Build the reference transcript from the prompt text."""
        text = row["text"]
        words = text.split()
        reference = Transcript.from_words_info(
            words=words,
        )

        extra_info: SpeechGenerationExtraInfo = {}
        if "language" in row:
            extra_info["language"] = row["language"]
        # When the dataset ships a reference clip, its transcript doubles as the
        # voice-clone `ref_text` (the pipeline materializes the clip as ref_audio).
        if self._has_reference_audio(row):
            extra_info["ref_text"] = text

        return reference, extra_info

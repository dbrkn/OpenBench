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
    metric compares the generated clip against `ref_audio`. `sample_idx` carries
    the dataset's stable per-sample id (e.g. the source file name).
    """

    language: str
    ref_audio: str
    ref_text: str
    sample_idx: str


class SpeechGenerationRow(TypedDict, total=False):
    """Expected row structure for speech generation.

    The text to synthesize (also the WER ground truth) comes from `prompt_text`
    (voice-clone datasets) or, for plain TTS datasets, the legacy `text` column.
    For voice cloning, `target_text` is the transcript of the reference clip
    (`audio`), used as `--ref-text`.
    """

    prompt_text: str
    text: str
    target_text: str
    language: str
    sample_idx: str


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

    # No hard-required column: the synthesis text may be `prompt_text` (voice-clone
    # datasets) or the legacy `text` (plain TTS datasets); validated in prepare_sample.
    _expected_columns: list[str] = []
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
        """Build the synthesis reference transcript and the per-sample extra info.

        The reference `Transcript` is the text to synthesize and the WER ground
        truth: it comes from `prompt_text` (voice-clone datasets) or the legacy
        `text` column. For voice cloning, `target_text` is the transcript of the
        reference clip and is exposed as `ref_text` (`tts-cli --ref-text`).
        """
        synth_text = row.get("prompt_text")
        if synth_text is None:
            synth_text = row.get("text")
        if synth_text is None:
            raise ValueError(
                "Speech-generation dataset row must provide a 'prompt_text' (or legacy 'text') column "
                "with the text to synthesize."
            )
        reference = Transcript.from_words_info(words=synth_text.split())

        extra_info: SpeechGenerationExtraInfo = {}
        if row.get("language") is not None:
            extra_info["language"] = row["language"]
        # Stable per-sample id (e.g. source file name) for downstream result rows.
        if row.get("sample_idx") is not None:
            extra_info["sample_idx"] = str(row["sample_idx"])
        # When the dataset ships a reference clip, its transcript is the voice-clone
        # `ref_text`. New voice-clone schema uses `target_text`; plain datasets reuse
        # the synthesis text.
        if self._has_reference_audio(row):
            extra_info["ref_text"] = row.get("target_text") or synth_text

        return reference, extra_info

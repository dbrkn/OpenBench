# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Speech-generation pipeline via ArgmaxPrototypes `tts-cli` (voice cloning).

This mirrors `speech_generation_argmax_oss.py`, but generates audio through the
prototype `tts-cli` (from the `argmax_prototypes` package) instead of the
production Swift `argmax-cli tts`. The key capability `tts-cli` adds is voice
cloning: with `--mode voice_clone` it conditions synthesis on a reference
speaker clip (`--ref-audio`, optionally `--ref-text` for ICL mode).

For voice-cloning datasets such as seedTTS, each sample carries `ref_audio`
(and `ref_text`) in its `extra_info`. The pipeline threads the reference-audio
path onto the `GeneratedAudio` prediction so the speaker-similarity (SIM) metric
can compare the generated clip against the intended speaker without depending on
the dataset's reference *transcript* (which WER consumes).
"""

from enum import StrEnum
from pathlib import Path
from typing import Callable, Literal

import librosa
import soundfile as sf
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field, model_validator

from ...dataset.dataset_speech_generation import SpeechGenerationSample
from ...engine.argmax_prototype_engine import (
    ArgmaxPrototypeEngine,
    ArgmaxPrototypeEngineConfig,
    PrototypeTtsInput,
    PrototypeTtsOutput,
)
from ...pipeline_prediction import GeneratedAudio
from ..base import (
    Pipeline,
    PipelineConfig,
    PipelineOutput,
    PipelineType,
    register_pipeline,
)


logger = get_logger(__name__)

TEMP_TTS_AUDIO_DIR = Path("./temp_tts_prototype_audio")
# Generated clips live in their own subdir so they never collide with the
# per-sample reference clips (which share the same base filename) and are kept
# after the run. Reference clips go under `ref/` and are removed once scored.
GENERATED_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "generated"
TEMP_REF_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "ref"

# Voice-clone assets (SpeakerEncoder/SpeechEncoder etc.) ship under the "base"
# model family, not the CLI's default "customvoice" variant. So when cloning we
# default `--version-dir` to the base variant (still overridable per config).
DEFAULT_VOICE_CLONE_VERSION_DIR = "12hz-0.6b-base"


class PrototypeTtsSpeaker(StrEnum):
    """`tts-cli --speaker` allowed values (only used in custom_voice mode)."""

    RYAN = "RYAN"
    AIDEN = "AIDEN"
    ONO_ANNA = "ONO_ANNA"
    SOHEE = "SOHEE"
    ERIC = "ERIC"
    DYLAN = "DYLAN"
    SERENA = "SERENA"
    VIVIAN = "VIVIAN"
    UNCLE_FU = "UNCLE_FU"


class PrototypeTtsLanguage(StrEnum):
    """`tts-cli --language` allowed values."""

    ENGLISH = "ENGLISH"
    CHINESE = "CHINESE"
    JAPANESE = "JAPANESE"
    KOREAN = "KOREAN"
    GERMAN = "GERMAN"
    FRENCH = "FRENCH"
    RUSSIAN = "RUSSIAN"
    PORTUGUESE = "PORTUGUESE"
    SPANISH = "SPANISH"
    ITALIAN = "ITALIAN"


class TtsMode(StrEnum):
    """`tts-cli --mode` values this pipeline supports."""

    CUSTOM_VOICE = "custom_voice"
    VOICE_CLONE = "voice_clone"


class ArgmaxPrototypeSpeechGenerationConfig(PipelineConfig):
    """Config for the ArgmaxPrototypes (`tts-cli`) speech-generation pipeline.

    `mode` selects between a predefined `speaker` (custom_voice) and cloning a
    per-sample reference clip (voice_clone). In voice_clone mode the reference
    audio/text are read from each sample's `extra_info` (`ref_audio`, `ref_text`),
    not from this config, since they vary per sample.
    """

    cli_path: str | None = Field(
        default=None,
        description="Path to (or name on PATH of) the `tts-cli` executable. Defaults to `tts-cli`.",
    )
    mode: TtsMode = Field(
        default=TtsMode.VOICE_CLONE,
        description="--mode. `voice_clone` clones each sample's reference clip; `custom_voice` uses `speaker`.",
    )
    x_vector_only: bool = Field(
        default=False,
        description="--x-vector-only. Voice-clone with speaker embedding only (no ICL); `ref_text` not required.",
    )
    speaker: PrototypeTtsSpeaker = Field(
        default=PrototypeTtsSpeaker.AIDEN,
        description="--speaker. Only used in custom_voice mode.",
    )
    language: PrototypeTtsLanguage = Field(
        default=PrototypeTtsLanguage.ENGLISH, description="--language."
    )
    version_dir: str | None = Field(
        default=None,
        description=(
            "--version-dir (model variant). In voice_clone mode this defaults to the base "
            f"variant ({DEFAULT_VOICE_CLONE_VERSION_DIR!r}), which carries the voice-clone assets; "
            "in custom_voice mode None lets the CLI use its own default. Set any explicit variant to override."
        ),
    )
    models_path: str | None = Field(
        default=None,
        description="--models-path (HF repo or local dir). None uses the CLI default.",
    )
    instruction: str | None = Field(
        default=None,
        description="--instruction (style hint, e.g. 'Speak slowly'). 1.7B model variant only.",
    )
    code_decoder_backend: Literal["coreml", "mlx"] | None = Field(
        default="mlx",
        description="--code-decoder-backend. Defaults to 'mlx'; set 'coreml' for the CoreML asset, or None for the CLI default.",
    )
    speaker_encoder_variant: str | None = Field(
        default="W16A16-10s",
        description="--speaker-encoder-variant (SpeakerEncoder voice-clone CoreML asset). Only used in voice_clone mode; None uses the CLI default.",
    )
    speech_encoder_variant: str | None = Field(
        default="W16A16-10s",
        description="--speech-encoder-variant (SpeechEncoder voice-clone CoreML asset). Only used in ICL voice_clone mode; None uses the CLI default.",
    )
    speech_encoder_rvq_variant: str | None = Field(
        default="W16A16-10s",
        description="--speech-encoder-rvq-variant (SpeechEncoderRVQ voice-clone CoreML asset). Only used in ICL voice_clone mode; None uses the CLI default.",
    )

    @model_validator(mode="after")
    def _default_voice_clone_version_dir(self) -> "ArgmaxPrototypeSpeechGenerationConfig":
        """Default `version_dir` to the base variant for voice cloning.

        Only fills it when the user left it unset, so an explicit `version_dir`
        always wins. custom_voice mode is left untouched (None → CLI default).
        """
        if self.version_dir is None and self.mode == TtsMode.VOICE_CLONE:
            self.version_dir = DEFAULT_VOICE_CLONE_VERSION_DIR
        return self

    def generate_tts_cli_args(self) -> list[str]:
        """Build the per-config (sample-independent) `tts-cli` flag list.

        Per-sample voice-clone flags (`--ref-audio`, `--ref-text`) are appended
        at synthesis time by the pipeline, not here.
        """
        args: list[str] = [
            "--language",
            str(self.language),
            "--mode",
            str(self.mode),
        ]
        if self.mode == TtsMode.CUSTOM_VOICE:
            args.extend(["--speaker", str(self.speaker)])
        if self.mode == TtsMode.VOICE_CLONE:
            if self.x_vector_only:
                args.append("--x-vector-only")
            # SpeakerEncoder is used for any voice_clone (x-vector or ICL).
            if self.speaker_encoder_variant is not None:
                args.extend(["--speaker-encoder-variant", self.speaker_encoder_variant])
            # SpeechEncoder(+RVQ) assets are ICL-only (skipped with --x-vector-only).
            if not self.x_vector_only:
                if self.speech_encoder_variant is not None:
                    args.extend(["--speech-encoder-variant", self.speech_encoder_variant])
                if self.speech_encoder_rvq_variant is not None:
                    args.extend(["--speech-encoder-rvq-variant", self.speech_encoder_rvq_variant])
        if self.version_dir is not None:
            args.extend(["--version-dir", self.version_dir])
        if self.models_path is not None:
            args.extend(["--models-path", self.models_path])
        if self.instruction is not None:
            args.extend(["--instruction", self.instruction])
        if self.code_decoder_backend is not None:
            args.extend(["--code-decoder-backend", self.code_decoder_backend])
        return args


class PrototypeSpeechGenerationInput(BaseModel):
    """Input for the prototype speech-generation pipeline."""

    text: str = Field(..., description="Text prompt to generate speech from.")
    audio_name: str = Field(..., description="Unique identifier for this sample (used for temp file naming).")
    ref_audio: str | None = Field(
        default=None,
        description="Path to the reference-speaker clip to clone (required in voice_clone mode).",
    )
    ref_text: str | None = Field(
        default=None,
        description="Transcript of `ref_audio` (required for ICL voice_clone, i.e. not --x-vector-only).",
    )


@register_pipeline
class ArgmaxPrototypeSpeechGenerationPipeline(Pipeline):
    """Speech-generation pipeline using `tts-cli` via `ArgmaxPrototypeEngine`.

    For each sample, the pipeline:

    1. Builds a `tts-cli` flag list from the config (+ per-sample clone refs).
    2. Synthesizes audio to a temp directory under cwd.
    3. Measures the duration via librosa.
    4. Returns a `GeneratedAudio` prediction (path + duration + reference path).

    WER and SIM scoring are performed by their respective metrics, not here, so
    the pipeline's reported `prediction_time` reflects TTS only.
    """

    _config_class = ArgmaxPrototypeSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(self) -> Callable[[PrototypeSpeechGenerationInput], GeneratedAudio]:
        engine = ArgmaxPrototypeEngine(ArgmaxPrototypeEngineConfig(cli_path=self.config.cli_path))
        base_args = self.config.generate_tts_cli_args()
        mode = self.config.mode
        x_vector_only = self.config.x_vector_only

        def generate(inp: PrototypeSpeechGenerationInput) -> GeneratedAudio:
            tts_args = list(base_args)
            if mode == TtsMode.VOICE_CLONE:
                if not inp.ref_audio:
                    raise ValueError(
                        f"voice_clone mode requires a reference clip, but sample {inp.audio_name!r} "
                        "has no `ref_audio` in its extra_info."
                    )
                tts_args.extend(["--ref-audio", inp.ref_audio])
                if not x_vector_only:
                    if not inp.ref_text:
                        raise ValueError(
                            f"ICL voice_clone requires `ref_text` for sample {inp.audio_name!r}; "
                            "set `x_vector_only=true` to clone without it."
                        )
                    tts_args.extend(["--ref-text", inp.ref_text])

            GENERATED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            try:
                output: PrototypeTtsOutput = engine.tts(
                    PrototypeTtsInput(
                        text=inp.text,
                        output_dir=GENERATED_AUDIO_DIR,
                        output_filename=inp.audio_name,
                    ),
                    tts_args,
                )
                duration = float(librosa.get_duration(path=str(output.audio_path)))
                logger.debug("Generated TTS audio: %s (%.2fs)", output.audio_path, duration)
                return GeneratedAudio(
                    audio_path=str(output.audio_path),
                    duration=duration,
                    reference_audio_path=inp.ref_audio,
                )
            except Exception:
                # Clean up partial outputs so the temp dir doesn't grow across retries.
                (GENERATED_AUDIO_DIR / f"{inp.audio_name}.wav").unlink(missing_ok=True)
                (GENERATED_AUDIO_DIR / f"{inp.audio_name}.npy").unlink(missing_ok=True)
                raise

        return generate

    def parse_input(self, input_sample: SpeechGenerationSample) -> PrototypeSpeechGenerationInput:
        extra_info = input_sample.extra_info or {}
        ref_audio = extra_info.get("ref_audio")
        # If the dataset ships the reference clip as the sample waveform (rather
        # than a path), materialize it to a temp WAV so `tts-cli --ref-audio` and
        # the SIM metric can consume it. This is independent of TTS mode: the SIM
        # metric compares against this clip even in custom_voice mode (it is the
        # dataset's ground-truth target speaker). A length-1 waveform is the
        # placeholder used by prompt-only datasets, so skip it.
        if (
            not ref_audio
            and input_sample.waveform is not None
            and len(input_sample.waveform) > 1
        ):
            TEMP_REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            ref_path = TEMP_REF_AUDIO_DIR / f"{input_sample.audio_name}.wav"
            sf.write(str(ref_path), input_sample.waveform, input_sample.sample_rate)
            ref_audio = str(ref_path)
        return PrototypeSpeechGenerationInput(
            text=input_sample.reference.get_transcript_string(),
            audio_name=input_sample.audio_name,
            ref_audio=ref_audio,
            ref_text=extra_info.get("ref_text"),
        )

    def parse_output(self, output: GeneratedAudio) -> PipelineOutput[GeneratedAudio]:
        return PipelineOutput[GeneratedAudio](prediction=output)

    def cleanup_sample(self, output: PipelineOutput[GeneratedAudio]) -> None:
        """Drop per-sample byproducts, keeping only the single generated WAV.

        Once a sample is fully scored we remove:

        * the RVQ-frames ``.npy`` that ``tts-cli`` writes next to the generated
          WAV (nothing downstream reads it), and
        * the reference clip we materialized under ``TEMP_REF_AUDIO_DIR`` (in
          voice_clone mode `parse_input` writes the sample's target-speaker
          waveform there so both `tts-cli` and the SIM metric can read it).

        The generated WAV is intentionally left in place, and dataset-provided
        reference paths (outside our temp dir) are never touched.
        """
        pred = output.prediction

        # 1) The generated WAV's sidecar .npy (frames) — keep only the audio.
        try:
            Path(pred.audio_path).with_suffix(".npy").unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Could not remove frames file for %s: %s", pred.audio_path, e)

        # 2) The reference clip we materialized for this sample.
        ref_path = getattr(pred, "reference_audio_path", None)
        if not ref_path:
            return
        ref = Path(ref_path)
        try:
            if ref.resolve().is_relative_to(TEMP_REF_AUDIO_DIR.resolve()):
                ref.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Could not remove reference clip %s: %s", ref, e)

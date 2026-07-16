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
# SIM yardstick clips (refclone real target) go under `sim/` when materialized
# from the sample waveform.
GENERATED_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "generated"
TEMP_REF_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "ref"
TEMP_SIM_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "sim"

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
    voice_clone_backend: Literal["coreml", "mlx"] | None = Field(
        default=None,
        description=(
            "--voice-clone-backend. None (default) omits the flag for tts-cli builds that "
            "don't support it (e.g. rd-689). Set 'mlx' for variable-length encoders via "
            "mlx-audio (refclone study), or 'coreml' for the fixed-window CoreML "
            "SpeakerEncoder/SpeechEncoder assets."
        ),
    )
    mlx_voice_clone_repo_id: str | None = Field(
        default=None,
        description=(
            "--mlx-voice-clone-repo-id. Only used when voice_clone_backend=mlx. "
            "None lets tts-cli use its Base-family default."
        ),
    )
    mlx_max_sequence_length: int | None = Field(
        default=None,
        description=(
            "--mlx-max-sequence-length. KV budget for the MLX talker (ICL reference prefix "
            "+ text + generated frames, ~12.5 frames/s). None (default) defers to the "
            "tts-cli branch's own default (512 on berkin/mlx-voice-clone). Set 256 to "
            "reinstate the rd-689-era cap matching the CoreML SpeechDecoder's kv_len_256 "
            "cache (prevents an IndexError overflow there); the refclone study used 8192 "
            "for very long ICL references. Only applied when code_decoder_backend='mlx'."
        ),
    )
    speaker_encoder_variant: str | None = Field(
        default=None,
        description=(
            "--speaker-encoder-variant (CoreML SpeakerEncoder). Only used when "
            "voice_clone_backend=coreml (or unset). Example: W16A16-10s."
        ),
    )
    speech_encoder_variant: str | None = Field(
        default=None,
        description=(
            "--speech-encoder-variant (CoreML SpeechEncoder). Only used when "
            "voice_clone_backend=coreml (or unset). Example: W16A16-10s."
        ),
    )
    speech_encoder_rvq_variant: str | None = Field(
        default=None,
        description=(
            "--speech-encoder-rvq-variant (CoreML SpeechEncoderRVQ). Only used when "
            "voice_clone_backend=coreml (or unset). Example: W16A16-10s."
        ),
    )
    # Refclone-study alignment: opt-in pass-throughs to tts-cli. All default to
    # "omit the flag" so tts-cli builds without them (e.g. rd-689) keep working;
    # the refclone study sets no_chunk=true max_new_tokens=2500
    # mlx_repo_id=mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
    # mlx_max_sequence_length=8192 streaming=true via --pipeline-config.
    no_chunk: bool = Field(
        default=False,
        description=(
            "--no-chunk. Synthesize the full sample text in one ICL call "
            "(disables tts-cli's 35-word TextChunker) so long voice-clone "
            "targets are not truncated mid-script. Refclone uses True."
        ),
    )
    max_new_tokens: int | None = Field(
        default=None,
        description=(
            "--max-new-tokens (RVQ frames, ~12/s). Refclone uses 2500; None "
            "leaves the tts-cli / library default (no explicit cap)."
        ),
    )
    mlx_repo_id: str | None = Field(
        default=None,
        description=(
            "--mlx-repo-id. Only used with code_decoder_backend=mlx. Refclone uses "
            "the Base-bf16 talker to match voice_clone version_dir=12hz-0.6b-base."
        ),
    )
    streaming: bool = Field(
        default=False,
        description=(
            "--streaming. Passed for CLI compatibility with the refclone study; "
            "current tts-cli ICL is non-streaming (full text in prefix)."
        ),
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
            if self.voice_clone_backend is not None:
                args.extend(["--voice-clone-backend", self.voice_clone_backend])
            if self.voice_clone_backend == "mlx":
                if self.mlx_voice_clone_repo_id is not None:
                    args.extend(["--mlx-voice-clone-repo-id", self.mlx_voice_clone_repo_id])
            else:
                # CoreML encoder variants only apply when not using MLX VC encoders.
                if self.speaker_encoder_variant is not None:
                    args.extend(["--speaker-encoder-variant", self.speaker_encoder_variant])
                if not self.x_vector_only:
                    if self.speech_encoder_variant is not None:
                        args.extend(["--speech-encoder-variant", self.speech_encoder_variant])
                    if self.speech_encoder_rvq_variant is not None:
                        args.extend(
                            ["--speech-encoder-rvq-variant", self.speech_encoder_rvq_variant]
                        )
        if self.version_dir is not None:
            args.extend(["--version-dir", self.version_dir])
        if self.models_path is not None:
            args.extend(["--models-path", self.models_path])
        if self.instruction is not None:
            args.extend(["--instruction", self.instruction])
        if self.code_decoder_backend is not None:
            args.extend(["--code-decoder-backend", self.code_decoder_backend])
        if self.mlx_repo_id is not None:
            args.extend(["--mlx-repo-id", self.mlx_repo_id])
        # Cap the MLX talker so it can't out-generate the CoreML SpeechDecoder's kv cache.
        if self.code_decoder_backend == "mlx" and self.mlx_max_sequence_length is not None:
            args.extend(["--mlx-max-sequence-length", str(self.mlx_max_sequence_length)])
        if self.no_chunk:
            args.append("--no-chunk")
        if self.max_new_tokens is not None:
            args.extend(["--max-new-tokens", str(self.max_new_tokens)])
        if self.streaming:
            args.append("--streaming")
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
    sim_audio: str | None = Field(
        default=None,
        description=(
            "Path to the SIM yardstick clip. Refclone uses the REAL target wav here so SIM is "
            "comparable across reference lengths; seedTTS leaves this unset and falls back to ref_audio."
        ),
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
                # SIM yardstick: prefer explicit sim_audio (real target); else clone prompt.
                return GeneratedAudio(
                    audio_path=str(output.audio_path),
                    duration=duration,
                    reference_audio_path=inp.sim_audio or inp.ref_audio,
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
        sim_audio = extra_info.get("sim_audio")
        # If the dataset ships the reference clip as the sample waveform (rather
        # than a path), materialize it to a temp WAV so `tts-cli --ref-audio` and
        # the SIM metric can consume it. This is independent of TTS mode: the SIM
        # metric compares against this clip even in custom_voice mode (it is the
        # dataset's ground-truth target speaker). A length-1 waveform is the
        # placeholder used by prompt-only datasets, so skip it.
        #
        # Refclone rows also carry an explicit `ref_audio` path (clone prompt) while
        # the sample waveform is the REAL target — then materialize the waveform as
        # `sim_audio` (SIM yardstick) and keep `ref_audio` for cloning.
        has_waveform = (
            input_sample.waveform is not None and len(input_sample.waveform) > 1
        )
        if has_waveform and ref_audio and not sim_audio:
            TEMP_SIM_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            sim_path = TEMP_SIM_AUDIO_DIR / f"{input_sample.audio_name}.wav"
            sf.write(str(sim_path), input_sample.waveform, input_sample.sample_rate)
            sim_audio = str(sim_path)
        elif has_waveform and not ref_audio:
            TEMP_REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            ref_path = TEMP_REF_AUDIO_DIR / f"{input_sample.audio_name}.wav"
            sf.write(str(ref_path), input_sample.waveform, input_sample.sample_rate)
            ref_audio = str(ref_path)
        return PrototypeSpeechGenerationInput(
            text=input_sample.reference.get_transcript_string(),
            audio_name=input_sample.audio_name,
            ref_audio=ref_audio,
            ref_text=extra_info.get("ref_text"),
            sim_audio=sim_audio,
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

        # 2) The reference / SIM clips we materialized for this sample.
        for path_str in (
            getattr(pred, "reference_audio_path", None),
            # Clone prompts may still sit under TEMP_REF_AUDIO_DIR even when SIM
            # uses a different yardstick path on the prediction.
        ):
            if not path_str:
                continue
            ref = Path(path_str)
            try:
                resolved = ref.resolve()
                if resolved.is_relative_to(TEMP_REF_AUDIO_DIR.resolve()) or resolved.is_relative_to(
                    TEMP_SIM_AUDIO_DIR.resolve()
                ):
                    ref.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not remove temp clip %s: %s", ref, e)

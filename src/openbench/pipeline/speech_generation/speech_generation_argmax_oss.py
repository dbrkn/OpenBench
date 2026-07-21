# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Speech-generation pipeline via Argmax SDK open-source `argmax-cli tts`."""

from enum import StrEnum
import shutil
from pathlib import Path
from typing import Callable, Literal

import librosa
from argmaxtools.utils import get_logger
from pydantic import BaseModel, Field

from ...dataset.dataset_speech_generation import SpeechGenerationSample
from ...engine.argmax_oss_engine import (
    ArgmaxOpenSourceEngine,
    ArgmaxOpenSourceEngineConfig,
    TtsCliInput,
    TtsCliOutput,
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

TEMP_TTS_AUDIO_DIR = Path("./temp_tts_audio")
# Voice-clone reference clips (materialized from sample waveforms) and SIM
# yardstick clips live in their own subdirs, mirroring the prototype pipeline.
TEMP_REF_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "ref"
TEMP_SIM_AUDIO_DIR = TEMP_TTS_AUDIO_DIR / "sim"


class TtsSpeaker(StrEnum):
    """`argmax-cli tts --speaker` allowed values."""

    RYAN = "ryan"
    AIDEN = "aiden"
    ONO_ANNA = "ono-anna"
    SOHEE = "sohee"
    ERIC = "eric"
    DYLAN = "dylan"
    SERENA = "serena"
    VIVIAN = "vivian"
    UNCLE_FU = "uncle-fu"


class TtsLanguage(StrEnum):
    """`argmax-cli tts --language` allowed values."""

    ENGLISH = "english"
    CHINESE = "chinese"
    JAPANESE = "japanese"
    KOREAN = "korean"
    GERMAN = "german"
    FRENCH = "french"
    RUSSIAN = "russian"
    PORTUGUESE = "portuguese"
    SPANISH = "spanish"
    ITALIAN = "italian"


class ArgmaxOpenSourceSpeechGenerationConfig(PipelineConfig):
    """Config for the Argmax OSS speech-generation pipeline.

    Engine fields (cache_dir / commit_hash / cli_path) mirror the
    transcription / diarization argmax-oss configs. TTS-specific fields
    follow `argmax-cli tts` flags.
    """

    cache_dir: str | None = Field(
        default=None,
        description="Cache directory for argmax-oss clone + CLI build. "
        "Defaults to ARGMAX_OSS_CACHE_DIR or ~/.cache/openbench/argmax-oss.",
    )
    commit_hash: str | None = Field(
        default=None,
        description="Optional git commit pin for the clone.",
    )
    cli_path: str | None = Field(
        default=None,
        description="Prebuilt argmax-cli path; skips clone/build.",
    )
    speaker: TtsSpeaker = Field(default=TtsSpeaker.AIDEN, description="--speaker.")
    language: TtsLanguage = Field(default=TtsLanguage.ENGLISH, description="--language.")
    output_format: Literal["wav", "m4a"] = Field(
        default="wav",
        description="--output-format. WAV is preferred so the WER metric can decode without extra deps.",
    )
    seed: int | None = Field(default=None, description="--seed for reproducible output.")
    temperature: float = Field(default=0.9, description="--temperature.")
    top_k: int = Field(default=50, description="--top-k.")
    max_new_tokens: int = Field(default=245, description="--max-new-tokens (RVQ frames).")
    instruction: str | None = Field(
        default=None,
        description="--instruction (style hint, e.g. 'Speak slowly'). 1.7B model only.",
    )
    model: Literal["0.6b", "1.7b"] | None = Field(
        default=None,
        description="--model preset (0.6b or 1.7b). Leave None to use CLI default.",
    )
    models_path: str | None = Field(default=None, description="--models-path (local model dir).")
    model_repo: str | None = Field(default=None, description="--model-repo (HF repo).")
    version_dir: str | None = Field(default=None, description="--version-dir (overrides --model preset).")
    tokenizer: str | None = Field(default=None, description="--tokenizer (HF repo or local path).")
    repo_url: str | None = Field(
        default=None,
        description="Git repo to clone/build argmax-cli from (e.g. a fork carrying voice cloning).",
    )
    mode: Literal["custom_voice", "voice_clone"] = Field(
        default="custom_voice",
        description=(
            "voice_clone conditions each sample on its reference clip (ref_audio/ref_text from "
            "extra_info, passed as --ref-audio/--ref-text); custom_voice uses --speaker."
        ),
    )
    x_vector_only: bool = Field(
        default=False,
        description="--x-vector-only. Clone with the speaker embedding only (no ICL); ref_text not required.",
    )
    code_decoder_variant: str | None = Field(default=None, description="--code-decoder-variant.")
    multi_code_decoder_variant: str | None = Field(default=None, description="--multi-code-decoder-variant.")
    code_embedder_variant: str | None = Field(default=None, description="--code-embedder-variant.")
    multi_code_embedder_variant: str | None = Field(default=None, description="--multi-code-embedder-variant.")
    text_projector_variant: str | None = Field(default=None, description="--text-projector-variant.")
    speech_decoder_variant: str | None = Field(default=None, description="--speech-decoder-variant.")
    speech_decoder_mode: str | None = Field(
        default=None,
        description="--speech-decoder-mode (latencyOptimized | throughputOptimized | singleFunction).",
    )
    talker_backend: Literal["coreml", "mlx"] = Field(
        default="coreml",
        description=(
            "CodeDecoder (talker) backend, passed as argmax-cli --code-decoder-backend. "
            "mlx: the MLX talker (batched ICL prefill, no KV cap; macOS 14+, requires the "
            "Base-family mlx-community checkpoint in the local HF cache and a build with "
            "the TTSKitMLX target). Everything else stays CoreML."
        ),
    )
    encoder_backend: Literal["coreml", "mlx"] = Field(
        default="coreml",
        description=(
            "Voice-clone reference encoder, passed as argmax-cli --voice-clone-encoder-backend. "
            "coreml: fixed 10/15s windows, ANE. mlx: variable-length references, GPU (macOS 14+, "
            "requires the Base-family mlx-community checkpoint in the local HF cache and a build "
            "with the TTSKitMLX target)."
        ),
    )
    speaker_encoder_variant: str | None = Field(default=None, description="--speaker-encoder-variant.")
    speech_encoder_variant: str | None = Field(default=None, description="--speech-encoder-variant.")
    speech_encoder_rvq_variant: str | None = Field(default=None, description="--speech-encoder-rvq-variant.")
    target_chunk_size: int | None = Field(
        default=None,
        description=(
            "--target-chunk-size (characters). The CLI default (42) splits eval texts into many "
            "more chunks than the Python prototype's 35-word chunker (~190 chars); pass ~190 for "
            "protocol parity — fewer seams and fewer full ICL prefills per sample."
        ),
    )
    mlx_max_sequence_length: int | None = Field(
        default=None,
        description=(
            "--mlx-max-sequence-length (talker_backend=mlx only): KV budget in positions for the MLX "
            "talker (ICL prefix + generated frames; ~12.5 positions per reference second). The CLI "
            "default is 1024; long-reference sweeps need e.g. 6144."
        ),
    )
    max_reference_seconds: float | None = Field(
        default=None,
        description=(
            "--max-reference-seconds (encoder_backend=mlx only): reference-duration cap. The CLI "
            "default is 120 s; encode peak Metal memory scales ~90 MB per reference second, so raise "
            "it only on runners with enough unified memory (the reflen sweep needs ~450)."
        ),
    )

    def generate_tts_cli_args(self) -> list[str]:
        args: list[str] = [
            "--speaker",
            self.speaker,
            "--language",
            self.language,
            "--output-format",
            self.output_format,
            "--temperature",
            str(self.temperature),
            "--top-k",
            str(self.top_k),
            "--max-new-tokens",
            str(self.max_new_tokens),
        ]
        if self.seed is not None:
            args.extend(["--seed", str(self.seed)])
        if self.target_chunk_size is not None:
            args.extend(["--target-chunk-size", str(self.target_chunk_size)])
        if self.instruction is not None:
            args.extend(["--instruction", self.instruction])
        if self.model is not None:
            args.extend(["--model", self.model])
        if self.models_path is not None:
            args.extend(["--models-path", self.models_path])
        if self.model_repo is not None:
            args.extend(["--model-repo", self.model_repo])
        if self.version_dir is not None:
            args.extend(["--version-dir", self.version_dir])
        if self.tokenizer is not None:
            args.extend(["--tokenizer", self.tokenizer])
        for flag, value in [
            ("--code-decoder-variant", self.code_decoder_variant),
            ("--multi-code-decoder-variant", self.multi_code_decoder_variant),
            ("--code-embedder-variant", self.code_embedder_variant),
            ("--multi-code-embedder-variant", self.multi_code_embedder_variant),
            ("--text-projector-variant", self.text_projector_variant),
            ("--speech-decoder-variant", self.speech_decoder_variant),
            ("--speech-decoder-mode", self.speech_decoder_mode),
            ("--speaker-encoder-variant", self.speaker_encoder_variant),
            ("--speech-encoder-variant", self.speech_encoder_variant),
            ("--speech-encoder-rvq-variant", self.speech_encoder_rvq_variant),
        ]:
            if value is not None:
                args.extend([flag, value])
        return args


class SpeechGenerationInput(BaseModel):
    """Input for the speech-generation pipeline."""

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
        description="SIM yardstick clip path (held-out real target); falls back to ref_audio when unset.",
    )


@register_pipeline
class ArgmaxOpenSourceSpeechGenerationPipeline(Pipeline):
    """Speech-generation pipeline using `argmax-cli tts` via `ArgmaxOpenSourceEngine`.

    For each sample, the pipeline:

    1. Builds an `argmax-cli tts` flag list from the config.
    2. Synthesizes audio to a temp directory under cwd.
    3. Measures the duration via librosa.
    4. Returns a `GeneratedAudio` prediction (path + duration).

    WER scoring is performed by `SpeechGenerationWordErrorRate`, not here,
    so the pipeline's reported `prediction_time` reflects TTS only.
    """

    _config_class = ArgmaxOpenSourceSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(self) -> Callable[[SpeechGenerationInput], GeneratedAudio]:
        engine = ArgmaxOpenSourceEngine(
            ArgmaxOpenSourceEngineConfig(
                cache_dir=self.config.cache_dir,
                commit_hash=self.config.commit_hash,
                cli_path=self.config.cli_path,
                repo_url=self.config.repo_url,
            )
        )
        suffix = f".{self.config.output_format}"
        mode = self.config.mode
        x_vector_only = self.config.x_vector_only
        encoder_backend = self.config.encoder_backend
        talker_backend = self.config.talker_backend

        tts_args = self.config.generate_tts_cli_args()
        if talker_backend == "mlx":
            # Single CLI: the MLX talker is a flag on argmax-cli. Command-line
            # SwiftPM can't compile mlx-swift's Metal shaders, so graft the
            # xcodebuild-produced bundle next to the built binary once.
            self._graft_mlx_metallib(engine)
            tts_args.extend(["--code-decoder-backend", "mlx"])
            if self.config.mlx_max_sequence_length is not None:
                tts_args.extend(["--mlx-max-sequence-length", str(self.config.mlx_max_sequence_length)])
        if encoder_backend == "mlx":
            self._graft_mlx_metallib(engine)
            tts_args.extend(["--voice-clone-encoder-backend", "mlx"])
            if self.config.max_reference_seconds is not None:
                tts_args.extend(["--max-reference-seconds", str(self.config.max_reference_seconds)])

        def generate(inp: SpeechGenerationInput) -> GeneratedAudio:
            sample_args = list(tts_args)
            if mode == "voice_clone":
                if not inp.ref_audio:
                    raise ValueError(
                        f"voice_clone mode requires a reference clip, but sample {inp.audio_name!r} "
                        "has no `ref_audio` in its extra_info."
                    )
                if not x_vector_only and not inp.ref_text:
                    raise ValueError(
                        f"ICL voice_clone requires `ref_text` for sample {inp.audio_name!r}; "
                        "set `x_vector_only=true` to clone without it."
                    )
                sample_args.extend(["--ref-audio", inp.ref_audio])
                if x_vector_only:
                    sample_args.append("--x-vector-only")
                else:
                    sample_args.extend(["--ref-text", inp.ref_text])

            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = TEMP_TTS_AUDIO_DIR / f"{inp.audio_name}{suffix}"
            try:
                output: TtsCliOutput = engine.tts(
                    TtsCliInput(text=inp.text, output_path=audio_path),
                    sample_args,
                )
                duration = float(librosa.get_duration(path=str(output.audio_path)))
                logger.debug("Generated TTS audio: %s (%.2fs)", output.audio_path, duration)
                # SIM yardstick: prefer the explicit held-out target; else the clone prompt.
                return GeneratedAudio(
                    audio_path=str(output.audio_path),
                    duration=duration,
                    reference_audio_path=inp.sim_audio or inp.ref_audio,
                )
            except Exception:
                # Clean up partial output so the temp dir doesn't grow across retries.
                audio_path.unlink(missing_ok=True)
                raise

        return generate

    def _graft_mlx_metallib(self, engine: ArgmaxOpenSourceEngine) -> None:
        """Place mlx-swift's Metal shader bundle next to the built argmax-cli.

        Command-line SwiftPM cannot compile mlx-swift's Metal shaders (runtime
        'Failed to load the default metallib'); build the bundle once via
        xcodebuild and copy it into the SwiftPM release bin dir.
        """
        import subprocess

        bin_dir = Path(engine.cli_path).resolve().parent
        bundle = bin_dir / "mlx-swift_Cmlx.bundle"
        if bundle.exists():
            return
        repo_dir = bin_dir
        while repo_dir.name != ".build" and repo_dir.parent != repo_dir:
            repo_dir = repo_dir.parent
        repo_dir = repo_dir.parent
        logger.info("Grafting mlx-swift Metal bundle via xcodebuild (one-time per checkout)")
        subprocess.run(
            "xcodebuild build -scheme argmax-cli -destination platform=macOS "
            "-derivedDataPath .build/xcode -quiet",
            cwd=repo_dir,
            shell=True,
            check=True,
        )
        built = repo_dir / ".build" / "xcode" / "Build" / "Products" / "Debug" / "mlx-swift_Cmlx.bundle"
        if not built.exists():
            raise RuntimeError(f"xcodebuild did not produce {built}")
        shutil.copytree(built, bundle)

    def parse_input(self, input_sample: SpeechGenerationSample) -> SpeechGenerationInput:
        import soundfile as sf

        extra_info = input_sample.extra_info or {}
        ref_audio = extra_info.get("ref_audio")
        sim_audio = extra_info.get("sim_audio")
        # Mirror the prototype pipeline: when the dataset ships the reference
        # clip as the sample waveform, materialize it to a temp WAV for
        # `--ref-audio` / the SIM metric. When an explicit ref_audio path exists
        # (refclone-style datasets), the waveform is the REAL target instead and
        # becomes the SIM yardstick.
        has_waveform = input_sample.waveform is not None and len(input_sample.waveform) > 1
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

        return SpeechGenerationInput(
            text=input_sample.reference.get_transcript_string(),
            audio_name=input_sample.audio_name,
            ref_audio=ref_audio,
            ref_text=extra_info.get("ref_text"),
            sim_audio=sim_audio,
        )

    def parse_output(self, output: GeneratedAudio) -> PipelineOutput[GeneratedAudio]:
        return PipelineOutput[GeneratedAudio](prediction=output)

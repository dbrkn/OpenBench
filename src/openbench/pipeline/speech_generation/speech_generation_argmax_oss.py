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
    encoder_backend: Literal["coreml", "mlx"] = Field(
        default="coreml",
        description=(
            "Voice-clone reference encoder. coreml: argmax-cli encodes in-process via --ref-audio "
            "(fixed 10/15s windows). mlx: encode via the TTSKitMLX extension's ttskit-mlx-cli "
            "(variable-length references), then drive argmax-cli with --voice-clone-prompt. "
            "mlx requires the repo branch to contain Extensions/TTSKitMLX."
        ),
    )
    speaker_encoder_variant: str | None = Field(default=None, description="--speaker-encoder-variant.")
    speech_encoder_variant: str | None = Field(default=None, description="--speech-encoder-variant.")
    speech_encoder_rvq_variant: str | None = Field(default=None, description="--speech-encoder-rvq-variant.")

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
        tts_args = self.config.generate_tts_cli_args()
        suffix = f".{self.config.output_format}"
        mode = self.config.mode
        x_vector_only = self.config.x_vector_only
        encoder_backend = self.config.encoder_backend

        mlx_encode_cli: str | None = None
        if mode == "voice_clone" and encoder_backend == "mlx":
            mlx_encode_cli = self._build_mlx_encode_cli(engine)

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
                if mlx_encode_cli is not None:
                    # Two-step MLX path: variable-length encode to a prompt JSON,
                    # then drive argmax-cli with the precomputed prompt.
                    import subprocess

                    TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                    prompt_path = TEMP_TTS_AUDIO_DIR / f"{inp.audio_name}.prompt.json"
                    encode_cmd = [mlx_encode_cli, "encode", "--ref-audio", inp.ref_audio, "--output", str(prompt_path)]
                    if inp.ref_text:
                        encode_cmd.extend(["--ref-text", inp.ref_text])
                    if x_vector_only:
                        encode_cmd.append("--x-vector-only")
                    try:
                        subprocess.run(encode_cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"ttskit-mlx-cli encode failed: {e.stderr}") from e
                    sample_args.extend(["--voice-clone-prompt", str(prompt_path)])
                else:
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

    def _build_mlx_encode_cli(self, engine: ArgmaxOpenSourceEngine) -> str:
        """Build `ttskit-mlx-cli` from the Extensions/TTSKitMLX package in the
        same checkout argmax-cli was built from."""
        import subprocess

        repo_dir = Path(engine.cli_path).resolve()
        # engine.cli_path = <repo>/.build/<triple>/release/argmax-cli
        while repo_dir.name != ".build" and repo_dir.parent != repo_dir:
            repo_dir = repo_dir.parent
        repo_dir = repo_dir.parent
        ext_dir = repo_dir / "Extensions" / "TTSKitMLX"
        if not ext_dir.is_dir():
            raise RuntimeError(
                f"encoder_backend=mlx requires Extensions/TTSKitMLX in the argmax-oss checkout ({ext_dir} missing). "
                "Use a repo/commit that carries the MLX extension (e.g. berkin/voice-clone-mlx)."
            )
        logger.info("Building ttskit-mlx-cli in %s", ext_dir)
        build_cmd = "swift build -c release --product ttskit-mlx-cli"
        subprocess.run(build_cmd, cwd=ext_dir, shell=True, check=True)
        result = subprocess.run(
            f"{build_cmd} --show-bin-path", cwd=ext_dir, stdout=subprocess.PIPE, shell=True, text=True, check=True
        )
        bin_dir = Path(result.stdout.strip())
        cli = bin_dir / "ttskit-mlx-cli"
        if not cli.is_file():
            raise RuntimeError(f"ttskit-mlx-cli not found after build: {cli}")

        # Command-line SwiftPM can't compile mlx-swift's Metal shaders (see the
        # extension README): produce the Cmlx bundle via xcodebuild once and
        # graft it next to the SwiftPM binary so the kernels load at runtime.
        bundle = bin_dir / "mlx-swift_Cmlx.bundle"
        if not bundle.exists():
            logger.info("Grafting mlx-swift Metal bundle via xcodebuild (one-time per checkout)")
            subprocess.run(
                "xcodebuild build -scheme ttskit-mlx-cli -destination platform=macOS "
                "-derivedDataPath .build/xcode -quiet",
                cwd=ext_dir,
                shell=True,
                check=True,
            )
            built = ext_dir / ".build" / "xcode" / "Build" / "Products" / "Debug" / "mlx-swift_Cmlx.bundle"
            if not built.exists():
                raise RuntimeError(f"xcodebuild did not produce {built}")
            shutil.copytree(built, bundle)
        return str(cli)

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

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Speech-generation pipeline via the upstream Qwen3-TTS reference implementation.

Wraps the official PyTorch package (https://github.com/QwenLM/Qwen3-TTS,
``pip install qwen-tts``) so voice-clone evals can be scored against the
original released pipeline — zeroing out every Argmax implementation factor
(CoreML/MLX backends, chunking, fixed encoder windows, KV caps). The upstream
model handles the whole text in one generation call: no chunking, variable-
length reference encoding, and its own vocoder.
"""

from pathlib import Path
from typing import Callable

import librosa
from argmaxtools.utils import get_logger
from pydantic import Field

from ...pipeline_prediction import GeneratedAudio
from ..base import Pipeline, PipelineConfig, PipelineOutput, PipelineType, register_pipeline
from .speech_generation_argmax_oss import SpeechGenerationInput, TEMP_TTS_AUDIO_DIR, TEMP_REF_AUDIO_DIR, TEMP_SIM_AUDIO_DIR


logger = get_logger(__name__)


class QwenTTSInput(SpeechGenerationInput):
    """SpeechGenerationInput plus the sample's language code."""

    language: str | None = None

# `language` column values -> upstream language names.
_LANGUAGES = {
    "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian",
}


class QwenTTSSpeechGenerationConfig(PipelineConfig):
    """Config for the upstream Qwen3-TTS voice-clone pipeline."""

    model_id: str = Field(
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        description="HuggingFace model id (Base family required for voice cloning).",
    )
    device: str = Field(
        default="mps",
        description="torch device_map target: cuda:0 | mps | cpu.",
    )
    dtype: str = Field(
        default="float32",
        description="torch dtype: float32 | float16 | bfloat16 (bf16/fp16 need CUDA or recent MPS).",
    )
    attn_implementation: str = Field(
        default="sdpa",
        description="Attention backend: sdpa (portable) or flash_attention_2 (CUDA only).",
    )
    language: str = Field(
        default="English",
        description="Fallback language name when the sample carries none.",
    )
    x_vector_only: bool = Field(
        default=False,
        description="Use x-vector-only cloning (skips ICL reference codes; ref_text unused).",
    )
    skip_sample_file: str | None = Field(
        default=None,
        description=(
            "Path to a newline-separated list of sample names to skip (raises a fast per-sample "
            "error the runner logs and moves past). Crash-resume aid: fill it with the ids "
            "already uploaded to the results repo so restarts never regenerate them."
        ),
    )


@register_pipeline
class QwenTTSSpeechGenerationPipeline(Pipeline):
    """Voice-clone speech generation through the official qwen_tts package.

    Mirrors the ArgmaxOpenSource pipeline's dataset protocol (ref_audio /
    ref_text conditioning, sim_audio yardstick) so results are row-comparable,
    while generation runs entirely in upstream PyTorch.
    """

    _config_class = QwenTTSSpeechGenerationConfig
    pipeline_type = PipelineType.SPEECH_GENERATION

    def build_pipeline(self) -> Callable[[QwenTTSInput], GeneratedAudio]:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, self.config.dtype)
        logger.info(
            "Loading %s on %s (%s, attn=%s)",
            self.config.model_id, self.config.device, self.config.dtype, self.config.attn_implementation,
        )
        model = Qwen3TTSModel.from_pretrained(
            self.config.model_id,
            device_map=self.config.device,
            dtype=dtype,
            attn_implementation=self.config.attn_implementation,
        )
        fallback_language = self.config.language
        x_vector_only = self.config.x_vector_only
        skip = set()
        if self.config.skip_sample_file:
            skip = {l.strip() for l in open(self.config.skip_sample_file) if l.strip()}
            logger.info("Skipping %d already-scored samples", len(skip))

        def generate(inp: QwenTTSInput) -> GeneratedAudio:
            if inp.audio_name in skip:
                raise RuntimeError(f"skipped: {inp.audio_name} already scored in a previous attempt")
            if not inp.ref_audio:
                raise ValueError(f"voice_clone requires a reference clip for sample {inp.audio_name!r}")
            if not x_vector_only and not inp.ref_text:
                raise ValueError(f"ICL voice_clone requires ref_text for sample {inp.audio_name!r}")

            language = _LANGUAGES.get((inp.language or "").lower(), None) or fallback_language
            kwargs = dict(ref_audio=inp.ref_audio)
            if x_vector_only:
                kwargs["x_vector_only_mode"] = True
            else:
                kwargs["ref_text"] = inp.ref_text
            wavs, sr = model.generate_voice_clone(text=inp.text, language=language, **kwargs)

            TEMP_TTS_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            audio_path = TEMP_TTS_AUDIO_DIR / f"{inp.audio_name}.wav"
            sf.write(str(audio_path), wavs[0], sr)
            # MPS accumulates allocator/command-buffer state across long
            # autoregressive generations and eventually aborts the process
            # silently; flush it after every sample.
            if self.config.device.startswith("mps"):
                torch.mps.synchronize()
                torch.mps.empty_cache()
            duration = float(librosa.get_duration(path=str(audio_path)))
            return GeneratedAudio(
                audio_path=str(audio_path),
                duration=duration,
                reference_audio_path=inp.sim_audio or inp.ref_audio,
            )

        return generate

    # Same materialization protocol as the Argmax OSS pipeline so that
    # ref_audio / sim_audio semantics (and therefore SIM/WER comparability)
    # are identical across implementations.
    def parse_input(self, input_sample) -> SpeechGenerationInput:
        import soundfile as sf

        extra_info = input_sample.extra_info or {}
        ref_audio = extra_info.get("ref_audio")
        sim_audio = extra_info.get("sim_audio")
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

        return QwenTTSInput(
            text=input_sample.reference.get_transcript_string(),
            audio_name=input_sample.audio_name,
            ref_audio=ref_audio,
            ref_text=extra_info.get("ref_text"),
            sim_audio=sim_audio,
            language=extra_info.get("language"),
        )

    def parse_output(self, output: GeneratedAudio) -> PipelineOutput[GeneratedAudio]:
        return PipelineOutput[GeneratedAudio](prediction=output)

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Generative guard models from the Hugging Face Hub (Llama Guard, Qwen3Guard, Granite Guardian and the like)."""

from typing import Any, Callable, Literal

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_text_safety import TextSafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import (
    GuardOutputStyle,
    TextSafetyConfig,
    TextSafetyInput,
    TextSafetyOutput,
    parse_guard_verdict,
    resolve_device,
)


logger = get_logger(__name__)


class HuggingFaceGuardModelConfig(TextSafetyConfig):
    model_id: str = Field(..., description="Hub id of a causal language model trained as a safety guard.")
    system_prompt: str | None = Field(None, description="Optional system message.")
    prompt_template: str = Field(
        "{text}",
        description="User message, formatted with `{text}`. Guards whose chat template adds the policy keep the default.",
    )
    chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments for `apply_chat_template`, e.g. Granite Guardian's `guardian_config`.",
    )
    use_processor: bool = Field(
        False,
        description="Build the prompt with `AutoProcessor` instead of `AutoTokenizer` (multimodal guards such as Llama Guard 4).",
    )
    output_style: GuardOutputStyle = Field("auto", description="How to read the generated verdict.")
    max_new_tokens: int = Field(64, description="Generation budget; verdicts are short.")
    score_mode: Literal["label", "first_token"] = Field(
        "label",
        description=(
            "`label`: the unsafe score is 1 or 0 from the parsed verdict. `first_token`: the probability mass of "
            "`unsafe_tokens` against `safe_tokens` at the first generated position, for a graded ROC-AUC."
        ),
    )
    unsafe_tokens: list[str] = Field(default_factory=lambda: ["unsafe", "Unsafe", "Yes", "yes"])
    safe_tokens: list[str] = Field(default_factory=lambda: ["safe", "Safe", "No", "no"])
    device: str | None = Field(None, description="torch device; defaults to CUDA, then MPS, then CPU.")
    torch_dtype: str | None = Field(None, description="e.g. `bfloat16`; defaults to the checkpoint's dtype.")
    trust_remote_code: bool = Field(False, description="Allow the model's own code to run.")


def _first_token_ids(tokenizer: Any, words: list[str]) -> list[int]:
    """First token id of each word, with and without a leading space."""
    ids = set()
    for word in words:
        for variant in (word, " " + word):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                ids.add(encoded[0])
    return sorted(ids)


@register_pipeline
class HuggingFaceGuardModelPipeline(Pipeline):
    """Prompts a generative guard model with the text and parses the verdict it writes.

    Greedy decoding (temperature 0, as in the paper). The chat template of the
    checkpoint builds the prompt; `prompt_template`, `system_prompt` and
    `chat_template_kwargs` cover guards that need a policy or a config.
    """

    _config_class = HuggingFaceGuardModelConfig
    pipeline_type = PipelineType.TEXT_SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[TextSafetyInput], SafetyPrediction]:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        config = self.config
        device = resolve_device(config.device)
        dtype = getattr(torch, config.torch_dtype) if config.torch_dtype else None
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
        templater = (
            AutoProcessor.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
            if config.use_processor
            else tokenizer
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id, torch_dtype=dtype, trust_remote_code=config.trust_remote_code
        )
        model.to(device).eval()
        unsafe_ids = _first_token_ids(tokenizer, config.unsafe_tokens)
        safe_ids = _first_token_ids(tokenizer, config.safe_tokens)
        logger.info(f"{config.model_id} on {device}, output style {config.output_style}")

        def classify(sample: TextSafetyInput) -> SafetyPrediction:
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": config.prompt_template.format(text=sample.text)})
            inputs = templater.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **config.chat_template_kwargs,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            output = tokenizer.decode(generated.sequences[0][prompt_length:], skip_special_tokens=True)
            label, category = parse_guard_verdict(output, config.output_style)
            if config.score_mode == "first_token" and generated.scores:
                probabilities = torch.softmax(generated.scores[0][0].float(), dim=-1)
                unsafe_mass = float(probabilities[unsafe_ids].sum()) if unsafe_ids else 0.0
                safe_mass = float(probabilities[safe_ids].sum()) if safe_ids else 0.0
                total = unsafe_mass + safe_mass
                unsafe_score = unsafe_mass / total if total > 0 else float(label.value == "unsafe")
            else:
                unsafe_score = float(label.value == "unsafe")
            return SafetyPrediction(label=label, unsafe_score=unsafe_score, category=category, raw_output=output)

        return classify

    def parse_input(self, input_sample: TextSafetySample) -> TextSafetyInput:
        return TextSafetyInput(text=input_sample.text, audio_name=input_sample.audio_name)

    def parse_output(self, output: SafetyPrediction) -> TextSafetyOutput:
        return TextSafetyOutput(prediction=output)

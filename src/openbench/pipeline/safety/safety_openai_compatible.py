# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Guard models behind an OpenAI-compatible chat endpoint (Ollama, vLLM, LM Studio, OpenAI)."""

import os
from typing import Any, Callable

from argmaxtools.utils import get_logger
from pydantic import Field

from ...dataset.dataset_safety import SafetySample
from ...pipeline_prediction import SafetyPrediction
from ..base import Pipeline, PipelineType, register_pipeline
from .common import (
    GuardOutputStyle,
    SafetyConfig,
    SafetyInput,
    SafetyOutput,
    empty_text_prediction,
    parse_guard_verdict,
)


logger = get_logger(__name__)

LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0")


class OpenAICompatibleGuardConfig(SafetyConfig):
    model: str = Field(..., description="Model name as the endpoint knows it, e.g. `qwen3guard:4b` on Ollama.")
    base_url: str | None = Field(
        None,
        description="Endpoint; defaults to `OPENAI_BASE_URL`, then to Ollama at http://localhost:11434/v1.",
    )
    api_key_env: str = Field("OPENAI_API_KEY", description="Environment variable holding the API key.")
    system_prompt: str | None = Field(None, description="Optional system message.")
    prompt_template: str = Field("{text}", description="User message, formatted with `{text}`.")
    output_style: GuardOutputStyle = Field("auto", description="How to read the generated verdict.")
    max_tokens: int = Field(64, description="Generation budget; verdicts are short.")
    temperature: float = Field(0.0, description="Sampling temperature; 0 for the paper's setting.")
    extra_body: dict[str, Any] | None = Field(
        None,
        description='Extra request fields, e.g. `{"chat_template_kwargs": {"enable_thinking": false}}` for vLLM.',
    )


@register_pipeline
class OpenAICompatibleGuardPipeline(Pipeline):
    """Sends each text as a chat completion and parses the verdict from the reply.

    Lets the benchmark run against a local server (Ollama serves Qwen3Guard,
    Llama Guard and ShieldGemma builds) or a hosted API without loading
    weights in the process.
    """

    _config_class = OpenAICompatibleGuardConfig
    pipeline_type = PipelineType.SAFETY_CLASSIFICATION

    def build_pipeline(self) -> Callable[[SafetyInput], SafetyPrediction]:
        from openai import OpenAI

        config = self.config
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL") or "http://localhost:11434/v1"
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            if any(host in base_url for host in LOCAL_HOSTS):
                api_key = "local"
            else:
                raise ValueError(f"`{config.api_key_env}` is not set and {base_url} is not a local endpoint")
        client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"{config.model} at {base_url}, output style {config.output_style}")

        def classify(sample: SafetyInput) -> SafetyPrediction:
            text = sample.require_text()
            if not text.strip():
                return empty_text_prediction()
            messages = []
            if config.system_prompt:
                messages.append({"role": "system", "content": config.system_prompt})
            messages.append({"role": "user", "content": config.prompt_template.format(text=text)})
            response = client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                extra_body=config.extra_body,
            )
            output = response.choices[0].message.content or ""
            label, category = parse_guard_verdict(output, config.output_style)
            return SafetyPrediction(
                label=label, unsafe_score=float(label.value == "unsafe"), category=category, raw_output=output
            )

        return classify

    def parse_input(self, input_sample: SafetySample) -> SafetyInput:
        return SafetyInput.from_sample(input_sample)

    def parse_output(self, output: SafetyPrediction) -> SafetyOutput:
        return SafetyOutput(prediction=output)

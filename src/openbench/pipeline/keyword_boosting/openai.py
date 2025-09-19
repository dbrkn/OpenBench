from pathlib import Path
from typing import Callable

from pydantic import Field

from ...engine import OpenAIApi, OpenAIApiResponse
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import BoostingOutput


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class OpenAIBoostingPipelineConfig(PipelineConfig):
    model_version: str = Field(
        default="whisper-1",
        description="The version of the OpenAI model to use",
    )
    boosting: bool = Field(
        default=True,
        description="Whether to use keyword prompting",
    )


@register_pipeline
class OpenAIBoostingPipeline(Pipeline):
    _config_class = OpenAIBoostingPipelineConfig
    pipeline_type = PipelineType.BOOSTING_TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], OpenAIApiResponse]:
        openai_api = OpenAIApi(model=self.config.model_version)

        def transcribe(audio_path: Path) -> OpenAIApiResponse:
            response = openai_api.transcribe(
                audio_path,
                prompt=self.current_keywords_prompt,
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def __call__(self, sample) -> BoostingOutput:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        self.current_keywords_prompt = None
        if self.config.boosting:
            keywords = sample.extra_info.get('dictionary', [])
            if keywords:
                # Format keywords as comma-separated prompt for OpenAI
                self.current_keywords_prompt = ", ".join(keywords)

        # Call parent implementation
        return super().__call__(sample)

    def parse_input(self, input_sample) -> Path:
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: OpenAIApiResponse) -> BoostingOutput:
        return BoostingOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=output.speakers,
                start=output.start,
                end=output.end,
            ),
            transcription_output=None,
        )

from pathlib import Path
from typing import Callable, Optional

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
    keywords_path: Optional[Path] = Field(
        default=None,
        description="The path to the keywords file",
    )


@register_pipeline
class OpenAIBoostingPipeline(Pipeline):
    _config_class = OpenAIBoostingPipelineConfig
    pipeline_type = PipelineType.BOOSTING_TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], OpenAIApiResponse]:
        openai_api = OpenAIApi(model=self.config.model_version)

        self.keywords_prompt = None
        if self.config.keywords_path is not None:
            # Read keywords and format them as a prompt for OpenAI
            keywords = []
            with open(self.config.keywords_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # Split by comma and strip whitespace from each keyword
                        line_keywords = [kw.strip() for kw in line.strip().split(',')]
                        keywords.extend(line_keywords)
            if keywords:
                # Format keywords as comma-separated prompt for OpenAI
                self.keywords_prompt = ", ".join(keywords)

        def transcribe(audio_path: Path) -> OpenAIApiResponse:
            response = openai_api.transcribe(
                audio_path,
                prompt=(
                    self.keywords_prompt
                    if self.config.keywords_path is not None
                    else None
                ),
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

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

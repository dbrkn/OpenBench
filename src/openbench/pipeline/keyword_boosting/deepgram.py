from pathlib import Path
from typing import Callable

from deepgram import PrerecordedOptions
from pydantic import Field

from ...engine import DeepgramApi, DeepgramApiResponse
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import BoostingOutput


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class DeepgramBoostingPipelineConfig(PipelineConfig):
    model_version: str = Field(
        default="base",
        description="The version of the Deepgram model to use",
    )
    boosting: bool = Field(
        default=True,
        description="Whether to use keywords boosting",
    )


@register_pipeline
class DeepgramBoostingPipeline(Pipeline):
    _config_class = DeepgramBoostingPipelineConfig
    pipeline_type = PipelineType.BOOSTING_TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        deepgram_api = DeepgramApi(
            options=PrerecordedOptions(
                model=self.config.model_version, smart_format=True)
        )

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            response = deepgram_api.transcribe(
                audio_path, keyterm=self.current_keywords
            )
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

    def __call__(self, sample) -> BoostingOutput:
        """Override to extract keywords from sample before processing."""
        # Extract keywords from sample's extra_info if flag is enabled
        self.current_keywords = None
        if self.config.boosting:
            keywords = sample.extra_info.get('dictionary', [])
            if keywords:
                # Add + between keywords for Deepgram URL
                self.current_keywords = "+".join(keywords)

        # Call parent implementation
        return super().__call__(sample)

    def parse_input(self, input_sample) -> Path:
        return input_sample.save_audio(TEMP_AUDIO_DIR)

    def parse_output(self, output: DeepgramApiResponse) -> BoostingOutput:
        return BoostingOutput(
            prediction=Transcript.from_words_info(
                words=output.words,
                speaker=output.speakers,
                start=output.start,
                end=output.end,
            ),
            transcription_output=None,
        )

from pathlib import Path
from typing import Callable

from deepgram import PrerecordedOptions
from pydantic import Field

from ...engine import DeepgramApi, DeepgramApiResponse
from ...pipeline import Pipeline, PipelineConfig, register_pipeline
from ...pipeline_prediction import Transcript
from ...types import PipelineType
from .common import BoostingOutput

from typing import Optional


TEMP_AUDIO_DIR = Path("temp_audio_dir")


class DeepgramBoostingPipelineConfig(PipelineConfig):
    model_version: str = Field(
        default="base",
        description="The version of the Deepgram model to use",
    )
    keywords_path: Optional[Path] = Field(
        default=None,
        description="The path to the keywords file",
    )


@register_pipeline
class DeepgramBoostingPipeline(Pipeline):
    _config_class = DeepgramBoostingPipelineConfig
    pipeline_type = PipelineType.BOOSTING_TRANSCRIPTION

    def build_pipeline(self) -> Callable[[Path], DeepgramApiResponse]:
        deepgram_api = DeepgramApi(
            options=PrerecordedOptions(
                model=self.config.model_version, smart_format=True, diarize=False, detect_language=True
            )
        )
        self.encoded_keywords = None
        if self.config.keywords_path is not None:
            with open(self.config.keywords_path, 'r') as f:
                keywords = []
                for line in f:
                    if line.strip():
                        # Split by comma and strip whitespace from each keyword
                        line_keywords = [kw.strip() for kw in line.strip().split(',')]
                        keywords.extend(line_keywords)
                
                # URL encode keywords for Deepgram
                self.encoded_keywords = " ".join("%20".join(kw.split()) for kw in keywords)

        def transcribe(audio_path: Path) -> DeepgramApiResponse:
            response = deepgram_api.transcribe(audio_path,
                                               keyterm=self.encoded_keywords if self.config.keywords_path is not None else None)
            # Remove temporary audio path
            audio_path.unlink(missing_ok=True)
            return response

        return transcribe

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

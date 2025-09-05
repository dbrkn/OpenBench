# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

from ..base import PipelineConfig, PipelineOutput
from ...pipeline_prediction import Transcript
from pydantic import Field
from ..base import (
    PipelineOutput,
)
from ..transcription import TranscriptionOutput


class BoostingOutput(PipelineOutput[Transcript]):
    transcription_output: TranscriptionOutput | None = Field(
        default=None,
        description="The transcription output from the ASR pipeline",
    )

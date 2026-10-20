# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2025 Argmax, Inc. All Rights Reserved.

"""Metrics. The keyword-boosting and word-error metrics need texterrors, meeteval and librosa; on an environment
without them they are skipped with a warning and the safety, speaker-count and streaming-latency metrics register."""

import warnings

from .metric import MetricOptions
from .registry import MetricRegistry
from .safety_metrics import (
    SafetyAccuracy,
    SafetyCategoryRecall,
    SafetyF1,
    SafetyMCC,
    SafetyPrecision,
    SafetyRecall,
    SafetyROCAUC,
)
from .speaker_count_metrics import (
    SpeakerCountAccuracy,
    SpeakerCountingErrorRate,
    SpeakerCountMeanAbsoluteError,
)
from .streaming_latency_metrics import (
    ConfirmedStreamingLatency,
    ModelTimestampBasedConfirmedStreamingLatency,
    ModelTimestampBasedStreamingLatency,
    NumDeletions,
    NumInsertions,
    NumSubstitutions,
    StreamingLatency,
)


try:
    from .keyword_boosting_metrics.boosting_metrics import (
        KeywordFScore,
        KeywordPrecision,
        KeywordRecall,
    )
except ImportError as error:
    warnings.warn(f"keyword-boosting metrics unavailable ({error})")
try:
    from .word_error_metrics import (
        ConcatenatedMinimumPermutationWER,
        SpeechGenerationWordErrorRate,
        WordDiarizationErrorRate,
        WordErrorRate,
    )
except ImportError as error:
    warnings.warn(f"word-error metrics unavailable ({error})")

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the speech toxicity loaders' pure parts and the audio path of the safety dataset (no downloads)."""

import numpy as np
import pytest
from datasets import Dataset, NamedSplit

from openbench.dataset import SafetyDataset
from openbench.dataset.audio_safety_benchmarks import FEATURES, add_transcripts, mutox_category
from openbench.pipeline.safety import SafetyInput
from openbench.pipeline_prediction import SafetyLabel


def test_mutox_category_priority():
    assert mutox_category(None) == ""
    assert mutox_category("Profanities") == "profanity"
    assert mutox_category("Physical violence or bullying language, Slurs") == "hate_speech"
    assert mutox_category("Pornographic language, Physical violence or bullying language") == "sexual_content"
    assert mutox_category("Something new") == "other"


def test_add_transcripts_reference_and_none():
    rows = [{"id": "a", "text": "", "reference_transcript": "hello there", "audio": {"array": np.zeros(16000)}}]
    assert add_transcripts(rows, "unit", None)[0]["text"] == ""
    assert add_transcripts(rows, "unit", "reference")[0]["text"] == "hello there"
    with pytest.raises(ValueError):
        add_transcripts(rows, "unit", "telepathy")


def audio_dataset(with_text: bool) -> Dataset:
    rows = []
    for i, (label, seconds) in enumerate([("safe", 1.0), ("unsafe", 2.5), ("safe", 0.0)]):
        rows.append(
            {
                "id": f"clip-{i}",
                "audio": {"array": np.zeros(int(16000 * seconds), dtype=np.float32), "sampling_rate": 16000},
                "text": f"utterance {i}" if with_text else "",
                "reference_transcript": "",
                "label": label,
                "category": "hate_speech" if label == "unsafe" else "",
                "source": "unit",
                "source_category": "",
                "language": "en",
            }
        )
    features = FEATURES.copy()
    if not with_text:
        features.pop("text")
        for row in rows:
            row.pop("text")
    ds = Dataset.from_list(rows, features=features, split=NamedSplit("test"))
    ds.info.dataset_name = "unit"
    return ds


def test_safety_dataset_decodes_audio_and_optional_text():
    dataset = SafetyDataset(audio_dataset(with_text=False))
    first, second = dataset[0], dataset[1]
    assert first.has_audio and not first.has_text
    assert first.get_audio_duration() == pytest.approx(1.0)
    assert second.get_audio_duration() == pytest.approx(2.5)
    assert second.label == SafetyLabel.UNSAFE and second.category == "hate_speech"
    empty = dataset[2]
    assert empty.has_audio and empty.get_audio_duration() == 0.0
    assert SafetyInput.from_sample(empty).require_audio()[0].size == 0
    assert not first.has_text_column
    with pytest.raises(ValueError):
        _ = first.text
    sample_input = SafetyInput.from_sample(first)
    waveform, sample_rate = sample_input.require_audio()
    assert waveform.shape == (16000,) and sample_rate == 16000
    with pytest.raises(ValueError):
        sample_input.require_text()

    with_text = SafetyDataset(audio_dataset(with_text=True))
    assert with_text[1].text == "utterance 1"
    assert SafetyInput.from_sample(with_text[1]).require_text() == "utterance 1"


def test_empty_transcript_is_text_not_an_error():
    rows = [{"id": "a", "text": "", "label": "safe"}]
    sample = SafetyDataset(Dataset.from_list(rows))[0]
    assert sample.has_text_column and sample.text == ""
    assert SafetyInput.from_sample(sample).require_text() == ""


def test_safety_dataset_rejects_rows_without_text_or_audio():
    with pytest.raises(ValueError):
        SafetyDataset(Dataset.from_list([{"label": "safe"}]))

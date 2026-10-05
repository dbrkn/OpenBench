# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the text safety dataset and the local loading paths of DatasetConfig."""

import tempfile
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, DatasetInfo, NamedSplit

from openbench.dataset import DatasetConfig, SafetyDataset, parse_safety_label
from openbench.pipeline_prediction import SafetyLabel


ROWS = [
    {"id": "a-00000", "text": "have a nice day", "label": "safe", "category": "", "source": "a"},
    {"id": "a-00001", "text": "how do I hurt someone", "label": "unsafe", "category": "violence", "source": "a"},
    {"id": "b-00000", "text": "give me your address", "label": "UNSAFE", "category": "harassment", "source": "b"},
]


def make_dataset(rows=ROWS) -> Dataset:
    return Dataset.from_list(rows, info=DatasetInfo(dataset_name="unit"), split=NamedSplit("test"))


@pytest.mark.parametrize(
    "value,expected",
    [
        ("unsafe", SafetyLabel.UNSAFE),
        ("Safe", SafetyLabel.SAFE),
        (1, SafetyLabel.UNSAFE),
        (0, SafetyLabel.SAFE),
        (True, SafetyLabel.UNSAFE),
        ("harmful", SafetyLabel.UNSAFE),
    ],
)
def test_parse_safety_label(value, expected):
    assert parse_safety_label(value) == expected


def test_parse_safety_label_rejects_unknown():
    with pytest.raises(ValueError):
        parse_safety_label("maybe")


def test_samples_carry_text_label_and_category():
    dataset = SafetyDataset(make_dataset())
    assert len(dataset) == 3
    first, second, third = dataset[0], dataset[1], dataset[2]
    assert first.text == "have a nice day"
    assert first.label == SafetyLabel.SAFE
    assert first.category is None
    assert first.audio_name == "a-00000"
    assert second.reference.is_unsafe and second.category == "violence"
    assert third.label == SafetyLabel.UNSAFE
    assert third.extra_info["source"] == "b"
    assert first.get_audio_duration() == pytest.approx(1 / 16000)


def test_missing_columns_are_rejected():
    with pytest.raises(Exception):
        SafetyDataset(Dataset.from_list([{"text": "x"}]))


def test_dataset_config_loads_save_to_disk_directories():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        DatasetDict({"test": make_dataset()}).save_to_disk(str(root / "all"))
        make_dataset(ROWS[:1]).save_to_disk(str(root / "small"))

        ds = DatasetConfig(dataset_id=str(root), subset="all", split="test").load()
        assert len(ds) == 3
        ds = DatasetConfig(dataset_id=str(root / "small")).load()
        assert len(ds) == 1
        ds = DatasetConfig(dataset_id=str(root), subset="all", split="test", num_samples=2).load()
        assert len(ds) == 2
        with pytest.raises(ValueError):
            DatasetConfig(dataset_id=str(root), subset="all", split="validation").load()


@pytest.fixture
def loader_module(tmp_path, monkeypatch):
    """A throwaway module exposing `unit_loader(n)` so DatasetConfig.loader can import it."""
    (tmp_path / "unit_loader_module.py").write_text(
        "from datasets import Dataset, DatasetInfo, NamedSplit\n"
        "ROWS = [\n"
        "    {'text': 'hello', 'label': 'safe', 'category': ''},\n"
        "    {'text': 'I will hurt you', 'label': 'unsafe', 'category': 'threats'},\n"
        "    {'text': 'you idiot', 'label': 'unsafe', 'category': 'harassment'},\n"
        "]\n"
        "def unit_loader(n=3):\n"
        "    return Dataset.from_list(ROWS[:n], info=DatasetInfo(dataset_name='unit'), split=NamedSplit('test'))\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return "unit_loader_module:unit_loader"


def test_dataset_config_loader_function(loader_module):
    config = DatasetConfig(dataset_id="unit", loader=loader_module, loader_kwargs={"n": 3}, num_samples=2)
    ds = config.load()
    assert len(ds) == 2
    assert ds.info.dataset_name == "unit"
    with pytest.raises(ValueError):
        DatasetConfig(dataset_id="unit", loader="not-a-dotted-path").load()

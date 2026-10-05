# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the text safety pipelines that need no model weights."""

import tempfile

import numpy as np
import pytest

from openbench.dataset import DatasetConfig, SafetySample
from openbench.metric import MetricOptions
from openbench.pipeline import PipelineRegistry
from openbench.pipeline.safety import ConstantSafetyPipeline, parse_guard_verdict
from openbench.pipeline.safety.safety_hf_classifier import resolve_unsafe_label_ids
from openbench.pipeline_prediction import SafetyLabel, SafetyPrediction
from openbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig


@pytest.mark.parametrize(
    "output,style,label,category",
    [
        ("unsafe\nS1,S9", "safe_unsafe", SafetyLabel.UNSAFE, "S1,S9"),
        ("\n\nsafe", "safe_unsafe", SafetyLabel.SAFE, None),
        ("Safety: Controversial\nCategories: Violent", "qwen3guard", SafetyLabel.UNSAFE, "Violent"),
        ("Safety: Safe\nCategories: None", "qwen3guard", SafetyLabel.SAFE, "None"),
        ("Harmful request: yes\nResponse refusal: no", "wildguard", SafetyLabel.UNSAFE, None),
        (
            '{"User Safety": "unsafe", "Safety Categories": "Violence, Threat"}',
            "json",
            SafetyLabel.UNSAFE,
            "Violence, Threat",
        ),
        ('Verdict: {"label": "safe", "rationale": "benign"}', "auto", SafetyLabel.SAFE, None),
        ("Yes", "yes_no", SafetyLabel.UNSAFE, None),
        ("No.", "auto", SafetyLabel.SAFE, None),
        ("Harmful request: no", "auto", SafetyLabel.SAFE, None),
        ("I cannot tell", "auto", SafetyLabel.SAFE, "unparsed"),
    ],
)
def test_parse_guard_verdict(output, style, label, category):
    assert parse_guard_verdict(output, style) == (label, category)


def test_resolve_unsafe_label_ids():
    id2label = {0: "polite", 1: "somewhat polite", 2: "neutral", 3: "impolite"}
    assert resolve_unsafe_label_ids(id2label, ["impolite"], False) == [3]
    assert resolve_unsafe_label_ids(id2label, None, False) == [3]
    assert resolve_unsafe_label_ids({0: "LABEL_0", 1: "LABEL_1"}, None, False) == [1]
    assert resolve_unsafe_label_ids({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, None, True) == [0, 1, 2]
    with pytest.raises(ValueError):
        resolve_unsafe_label_ids(id2label, ["rude"], False)
    with pytest.raises(ValueError):
        resolve_unsafe_label_ids({0: "A", 1: "B", 2: "C"}, None, False)


def test_constant_pipeline_runs_on_a_sample():
    pipeline = ConstantSafetyPipeline.from_dict({"label": "safe"})
    sample = SafetySample(
        audio_name="x",
        waveform=np.zeros(1, dtype=np.float32),
        sample_rate=16000,
        reference=SafetyPrediction(label=SafetyLabel.UNSAFE),
        extra_info={"text": "anything"},
    )
    output = pipeline(sample)
    assert output.prediction.label == SafetyLabel.SAFE
    assert output.prediction.unsafe_score == 0.0
    assert output.prediction_time is not None


def test_alias_overrides_and_pipeline_type():
    pipeline = PipelineRegistry.create_pipeline("constant-unsafe", {"label": "safe"})
    assert pipeline.config.label == SafetyLabel.SAFE
    assert PipelineRegistry.get_pipeline_type("roblox-pii-coreml").value == "safety_classification"


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


def test_benchmark_runner_end_to_end_with_constant_pipeline(loader_module):
    with tempfile.TemporaryDirectory() as tmp:
        config = BenchmarkConfig(
            wandb_config=WandbConfig(project_name="unit", run_name="unit", is_active=False),
            metrics={
                MetricOptions.RECALL: {},
                MetricOptions.PRECISION: {},
                MetricOptions.ROC_AUC: {},
                MetricOptions.CATEGORY_RECALL: {},
            },
            datasets={"three": DatasetConfig(dataset_id="three", loader=loader_module)},
        )
        pipeline = ConstantSafetyPipeline.from_dict({"label": "unsafe", "out_dir": tmp})
        results = BenchmarkRunner(config, [pipeline]).run()
        by_name = {g.metric_name: g for g in results.global_results}
        assert by_name["recall"].global_result == pytest.approx(1.0)
        assert by_name["precision"].global_result == pytest.approx(2 / 3)
        assert by_name["roc_auc"].global_result == pytest.approx(0.5)
        assert by_name["category_recall"].detailed_result["true_positives__threats"] == 1
        assert all(r.audio_duration == 0.0 for r in results.sample_results)

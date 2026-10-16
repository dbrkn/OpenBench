# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Tests for the PII pipelines and the runner that need no model weights."""

import tempfile

import numpy as np
import pytest

from openbench.dataset import DatasetConfig, SafetySample
from openbench.metric import MetricOptions
from openbench.pipeline import PipelineRegistry
from openbench.pipeline.safety import ConstantSafetyPipeline
from openbench.pipeline_prediction import SafetyLabel, SafetyPrediction
from openbench.runner import BenchmarkConfig, BenchmarkRunner, WandbConfig


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
    pipeline = PipelineRegistry.create_pipeline("constant-pii", {"label": "safe"})
    assert pipeline.config.label == SafetyLabel.SAFE
    for alias in ("roblox-pii-v2", "roblox-pii-v2-coreml", "openai-privacy-filter"):
        assert PipelineRegistry.get_pipeline_type(alias).value == "safety_classification"


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

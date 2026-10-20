# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""The full-run chain without model weights: run_full (regex and constant pipelines) -> report_full -> build_viewer."""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from openbench.dataset import DatasetConfig, DatasetRegistry
from openbench.pii_runs import Job, is_complete, run_pipeline_over_datasets
from openbench.pipeline.safety.safety_regex_pii import regex_spans
from openbench.types import PipelineType


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "pii"
sys.path.insert(0, str(SCRIPTS))

import build_viewer  # noqa: E402
import report_full  # noqa: E402


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    (tmp_path / "pii_tiny_loader.py").write_text(
        "from openbench.dataset.pii_benchmarks import finish, found, turn_rows\n"
        "def load():\n"
        "    turns = [\n"
        "        ('agent', 'how can I reach you?', []),\n"
        "        ('user', 'mail jo@mail.com please', [found('email', 5, 16)]),\n"
        "        ('user', 'my name is Ann Lee', [found('name', 11, 18)]),\n"
        "        ('user', 'thanks, that is all', []),\n"
        "    ]\n"
        "    rows = turn_rows(turns, [False, True, True, True], id_prefix='t', source='tiny', language='en')\n"
        "    return finish(rows, 'tiny', None, 0)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    DatasetRegistry.register_alias(
        "pii-tiny-unit",
        DatasetConfig(dataset_id="pii-tiny-unit", split="test", loader="pii_tiny_loader:load"),
        supported_pipeline_types={PipelineType.SAFETY_CLASSIFICATION},
        description="unit",
    )
    return "pii-tiny-unit"


def test_regex_floor_finds_structured_identifiers_only():
    kinds = [kind for _, _, kind, _ in regex_spans("jo@mail.com, +1 415 555 0123, card 4111 1111 1111 1111")]
    assert kinds == ["email", "phone", "card_number"]
    assert regex_spans("I have 3 kids and 2024 was great") == []


def test_full_run_report_and_viewer(tmp_path, tiny_dataset):
    out = tmp_path / "out"
    for pipeline in ("regex-pii", "constant-pii"):
        run_pipeline_over_datasets(pipeline, [tiny_dataset], out)
        assert is_complete(Job(pipeline, tiny_dataset), out)
    manifest = json.loads((out / "regex-pii" / tiny_dataset / "manifest.json").read_text())
    assert manifest["rows"] == 3 and manifest["pii_rows"] == 2 and manifest["errors"] == 0
    table = pq.read_table(out / "regex-pii" / tiny_dataset / "predictions.parquet").to_pylist()
    assert [row["verdict"] for row in sorted(table, key=lambda r: r["id"])] == ["unsafe", "safe", "safe"]
    assert run_pipeline_over_datasets("regex-pii", [tiny_dataset], out) == []  # complete jobs are skipped

    report_full.main(out)
    summary = json.loads((out / "report.json").read_text())[tiny_dataset]
    assert summary["regex-pii"]["recall"] == pytest.approx(0.5)  # the e-mail, not the name
    assert summary["regex-pii"]["types"]["email"] == [1, 1] and summary["regex-pii"]["types"]["name"] == [0, 1]
    assert summary["constant-pii"]["false_mask"] == pytest.approx(1.0)
    assert "regex-pii" in (out / "report.md").read_text()

    subsets = build_viewer.build(out, subset_size=10, seed=0)
    wide = pq.read_table(out / "viewer" / f"{tiny_dataset}.parquet").to_pylist()
    email_row = next(r for r in wide if "jo@mail.com" in r["text"])
    assert email_row["gold_censored"] == "mail [EMAIL] please"
    assert email_row["regex-pii__censored"] == "mail [EMAIL] please"
    assert email_row["regex-pii__outcome"] == "tp"
    assert json.loads(email_row["conversation"]) == [
        ["agent", "how can I reach you?", False],
        ["user", "mail jo@mail.com please", True],
    ]
    assert len(subsets[tiny_dataset]) == 3 and "regex-pii__raw" in subsets[tiny_dataset][0]


def test_wilson_interval_and_censor():
    low, high = report_full.wilson(8, 10)
    assert 0.44 < low < 0.5 and 0.94 < high < 0.97
    assert build_viewer.censor("call 555 0101 now", [(5, 13, "phone")]) == "call [PHONE] now"
    assert report_full.contaminated("gliner-pii-nvidia", "nemotron-pii-sentences-50k")
    assert not report_full.contaminated("pii-tracer", "abcd")

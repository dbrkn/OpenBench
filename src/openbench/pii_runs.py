# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Full PII benchmark runs: every segment of a dataset through a span pipeline in batches, with complete records.

`openbench-cli evaluate` scores one segment at a time and keeps a JSON per
prediction; a full run (hundreds of thousands of segments, many models) needs
batches and one columnar file per job. A job is one (pipeline alias, dataset
alias) pair and writes, under `<out>/<pipeline>/<dataset>/`:

- `predictions.parquet`: one row per segment (id, verdict, score, category,
  spans as JSON [start, end, type] in the segment's text, latency);
- `manifest.json`: the pipeline config, the model's Hub revision, package and
  device versions, the dataset's loader and build key, row counts, timings,
  errors, and `status: complete` once the parquet is in place;
- `log.txt`: the job's log.

The predictions come from the pipelines' own `classify_batch`, the code
`openbench-cli` runs one segment at a time, so both paths give the same verdicts.
"""

import json
import logging
import os
import platform
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


PACKAGES = ("torch", "transformers", "gliner", "gliner2", "datasets", "tokenizers", "huggingface_hub", "numpy")


@dataclass
class Job:
    pipeline: str
    dataset: str

    def directory(self, out: Path) -> Path:
        return out / self.pipeline / self.dataset


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
        info["visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
    else:
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return info


def git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        return (
            subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def model_revision(model_id: str | None) -> str | None:
    if not model_id:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:  # offline, local path, or a model without Hub metadata
        return None


def load_dataset_rows(dataset: str):
    """The built dataset (a `datasets.Dataset`) and its config, from the registered alias."""
    from .dataset import DatasetRegistry

    config = DatasetRegistry.get_alias_config(dataset)
    return config.load(), config


def to_inputs(rows) -> list:
    """SafetyInput per row, the same conversion `openbench-cli` applies through SafetyDataset."""
    from .dataset import SafetyDataset
    from .pipeline.safety.common import SafetyInput

    return [SafetyInput.from_sample(sample) for sample in SafetyDataset(rows)]


def batches(order: list[int], lengths: list[int], batch_size: int, char_budget: int):
    """Length-sorted batches of at most `batch_size` segments and `char_budget` characters (the longest times the
    count), so the long tail of a dataset runs in smaller batches instead of running out of GPU memory."""
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and end - start < batch_size and lengths[order[end]] * (end - start + 1) <= char_budget:
            end += 1
        yield start, order[start:end]
        start = end


def run_job(
    job: Job,
    out: Path,
    pipeline_obj: Any,
    pipeline_config: dict,
    *,
    batch_size: int,
    logger: logging.Logger,
    char_budget: int = 24_000,
) -> dict:
    """Run one (pipeline, dataset) job and write its parquet, manifest and log; returns the manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = job.directory(out)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows, dataset_config = load_dataset_rows(job.dataset)
    inputs = to_inputs(rows)
    ids = [str(row["id"]) for row in rows]
    lengths = [sum(len(t) for _, t, _ in item.window(8)) for item in inputs]
    order = sorted(range(len(inputs)), key=lambda i: lengths[i])
    logger.info(f"{job.pipeline} on {job.dataset}: {len(inputs)} segments, batch {batch_size}")

    predictions: dict[int, Any] = {}
    latency: dict[int, float] = {}
    errors: list[dict] = []
    batch_fn = getattr(pipeline_obj, "classify_batch", None) or (
        lambda batch: [pipeline_obj.pipeline(s) for s in batch]
    )
    for start, chunk in batches(order, lengths, batch_size, char_budget):
        tick = time.perf_counter()
        try:
            outputs = batch_fn([inputs[i] for i in chunk])
        except Exception as error:  # fall back to one segment at a time so a single bad row cannot sink the batch
            logger.warning(f"batch at {start} failed ({error!r}); retrying one by one")
            outputs = []
            for i in chunk:
                try:
                    outputs.append(pipeline_obj.pipeline(inputs[i]))
                except Exception as single:
                    errors.append({"id": ids[i], "error": repr(single), "trace": traceback.format_exc()[-2000:]})
                    outputs.append(None)
        per_item = 1000 * (time.perf_counter() - tick) / max(1, len(chunk))
        for i, output in zip(chunk, outputs):
            predictions[i] = output
            latency[i] = per_item
        if len(predictions) // 2000 != (len(predictions) - len(chunk)) // 2000:
            logger.info(f"  {len(predictions)}/{len(order)} segments")

    table = pa.table(
        {
            "id": ids,
            "verdict": [p.label.value if p else "error" for p in (predictions[i] for i in range(len(ids)))],
            "score": [float(p.unsafe_score or 0.0) if p else None for p in (predictions[i] for i in range(len(ids)))],
            "category": [(p.category or "") if p else "" for p in (predictions[i] for i in range(len(ids)))],
            "spans": [(p.raw_output or "") if p else "" for p in (predictions[i] for i in range(len(ids)))],
            "latency_ms": [latency[i] for i in range(len(ids))],
        }
    )
    partial = directory / "predictions.parquet.partial"
    pq.write_table(table, partial)
    partial.rename(directory / "predictions.parquet")
    finished = time.time()
    manifest = {
        "status": "complete",
        "pipeline": job.pipeline,
        "dataset": job.dataset,
        "pipeline_class": type(pipeline_obj).__name__,
        "pipeline_config": pipeline_config,
        "model_id": pipeline_config.get("model_id"),
        "model_revision": model_revision(pipeline_config.get("model_id")),
        "dataset_loader": dataset_config.loader,
        "dataset_loader_kwargs": dataset_config.loader_kwargs,
        "rows": len(ids),
        "pii_rows": sum(1 for row in rows if row["label"] == "unsafe"),
        "errors": len(errors),
        "batch_size": batch_size,
        "char_budget": char_budget,
        "seconds": round(finished - started, 2),
        "ms_per_segment_mean": round(sum(latency.values()) / max(1, len(latency)), 3),
        "started": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "finished": datetime.fromtimestamp(finished, timezone.utc).isoformat(),
        "openbench_commit": git_commit(),
        "packages": package_versions(),
        "device": device_info(),
    }
    if errors:
        (directory / "errors.json").write_text(json.dumps(errors, indent=1))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    logger.info(f"{job.pipeline} on {job.dataset}: done in {manifest['seconds']} s, {len(errors)} errors")
    return manifest


def is_complete(job: Job, out: Path) -> bool:
    manifest = job.directory(out) / "manifest.json"
    try:
        return json.loads(manifest.read_text()).get("status") == "complete"
    except (OSError, ValueError):
        return False


def job_logger(job: Job, out: Path) -> logging.Logger:
    logger = logging.getLogger(f"pii_runs.{job.pipeline}.{job.dataset}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    job.directory(out).mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.FileHandler(job.directory(out) / "log.txt"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_pipeline_over_datasets(
    pipeline: str, datasets: list[str], out: Path, *, overrides: dict | None = None, batch_size: int | None = None
) -> list[dict]:
    """Load one pipeline once and run it over every dataset whose job is not complete yet."""
    from .pipeline import PipelineRegistry

    pending = [Job(pipeline, dataset) for dataset in datasets if not is_complete(Job(pipeline, dataset), out)]
    if not pending:
        return []
    overrides = dict(overrides or {})
    if batch_size is not None:
        overrides.setdefault("batch_size", batch_size)
    tick = time.time()
    pipeline_obj = PipelineRegistry.create_pipeline(pipeline, overrides)
    load_seconds = round(time.time() - tick, 2)
    config = pipeline_obj.config.model_dump(mode="json")
    effective_batch = int(config.get("batch_size") or batch_size or 16)
    manifests = []
    for job in pending:
        logger = job_logger(job, out)
        logger.info(f"model load {load_seconds} s")
        try:
            manifest = run_job(job, out, pipeline_obj, config, batch_size=effective_batch, logger=logger)
            manifest["model_load_seconds"] = load_seconds
            (job.directory(out) / "manifest.json").write_text(json.dumps(manifest, indent=1))
            manifests.append(manifest)
        except Exception as error:
            logger.error(f"job failed: {error!r}\n{traceback.format_exc()}")
            (job.directory(out) / "manifest.json").write_text(
                json.dumps({"status": "failed", "error": repr(error), "pipeline": pipeline, "dataset": job.dataset})
            )
    return manifests

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Preflight for a full run: each pipeline's batched output must equal its one-segment-at-a-time output.

    python scripts/pii/check_batching.py --dataset abcd-mini --pipelines pii-tracer gliner2-pii-fastino

Padding can leak into attention on some backends (it did on MPS for PII-Tracer), which would silently change
verdicts in a batched run. Exits non-zero when any pipeline differs on any of the checked segments.
"""

import argparse
import json
import sys

from openbench.dataset import DatasetRegistry, SafetyDataset
from openbench.pipeline import PipelineRegistry
from openbench.pipeline.safety.common import SafetyInput


def differences(pipeline, inputs, tolerance: float) -> list[tuple[int, float, float]]:
    single = [pipeline.pipeline(item) for item in inputs]
    batch = pipeline.classify_batch(inputs)
    found = []
    for index, (a, b) in enumerate(zip(single, batch)):
        spans_a = json.loads(a.raw_output or '{"spans": []}')
        spans_b = json.loads(b.raw_output or '{"spans": []}')
        if a.label != b.label or abs((a.unsafe_score or 0) - (b.unsafe_score or 0)) > tolerance or spans_a != spans_b:
            found.append((index, a.unsafe_score, b.unsafe_score))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="abcd-mini")
    parser.add_argument("--pipelines", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=2e-3)
    args = parser.parse_args()
    rows = SafetyDataset(DatasetRegistry.get_alias_config(args.dataset).load())
    inputs = [SafetyInput.from_sample(sample) for _, sample in zip(range(args.samples), rows)]
    failed = False
    for alias in args.pipelines:
        pipeline = PipelineRegistry.create_pipeline(alias, {"batch_size": args.batch_size})
        if not hasattr(pipeline, "classify_batch"):
            print(f"skip {alias}: no batch path")
            continue
        found = differences(pipeline, inputs, args.tolerance)
        print(f"{'OK  ' if not found else 'DIFF'} {alias}: {len(found)}/{len(inputs)} differ {found[:3]}", flush=True)
        failed |= bool(found)
        del pipeline
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

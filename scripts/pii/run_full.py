# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Full PII benchmark: each pipeline loaded once, run over every dataset in batches (see openbench.pii_runs).

    python scripts/pii/run_full.py --out pii_full --pipelines openai-privacy-filter pii-tracer --datasets abcd spokenwoz
    python scripts/pii/run_full.py --out pii_full --worker 0 --num-workers 4     # one of 4 workers (scripts/pii/run_full.sh)
    python scripts/pii/run_full.py --build-only                                  # build every dataset cache, no models

Workers take pipelines round-robin, so a model is loaded by one worker only. A finished job is skipped.
"""

import argparse
import os
import sys
from pathlib import Path

from openbench.pii_runs import run_pipeline_over_datasets


FULL_PIPELINES = [
    "regex-pii",
    "openai-privacy-filter",
    "openai-privacy-filter-owner-context-2",
    "openmed-privacy-filter-multilingual-v2",
    "openmed-privacy-filter-multilingual-v2-owner-context-2",
    "pii-tracer",
    "pii-tracer-owner-context-2",
    "gliner2-pii-fastino",
    "gliner2-pii-fastino-owner-context-2",
    "gliner-pii-knowledgator-edge",
    "gliner-pii-knowledgator-edge-owner-context-2",
    "gliner-pii-knowledgator-small",
    "gliner-pii-knowledgator-base",
    "gliner-pii-knowledgator-large",
    "gliner-pii-urchade-multi",
    "gliner-pii-nvidia",
    "gliner-guard-omni",
    "gliner-guard-uni",
    "constant-pii",
]
FULL_DATASETS = os.getenv("PII_FULL_DATASETS", "").split() or [
    # conversation and speech
    "abcd",
    "pii-trace",
    "dialogpii",
    "dialogpii-asr",
    "spokenwoz",
    "spokenwoz-dev",
    "pii-bench-ru",
    "roblox-pii-chat-giving",
    # real text
    "tab",
    "tab-quasi",
    # documents and domains
    "nemotron-pii-sentences-50k",
    "gretel-finance",
    "gretel-en",
    "spy",
    "privy",
    # multilingual
    "openpii-50k",
    "ai4privacy-400k",
    "ai4privacy-500k",
    "e3jsi-multi",
    # a model's own training data (contamination check)
    "urchade-mistral",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("pii_full"))
    parser.add_argument("--pipelines", nargs="*", default=FULL_PIPELINES)
    parser.add_argument("--datasets", nargs="*", default=FULL_DATASETS)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None, help="Override every span pipeline's batch size.")
    parser.add_argument("--device", default=None, help="Override the pipelines' device (cuda, cpu, mps).")
    parser.add_argument("--build-only", action="store_true", help="Build the dataset caches and exit.")
    args = parser.parse_args()

    if args.build_only:
        from openbench.dataset import DatasetRegistry

        for dataset in args.datasets:
            rows = DatasetRegistry.get_alias_config(dataset).load()
            print(f"{dataset}: {len(rows)} rows, {sum(r == 'unsafe' for r in rows['label'])} with PII", flush=True)
        os._exit(0)  # a streamed dataset's threads can hang interpreter shutdown

    mine = [p for i, p in enumerate(args.pipelines) if i % args.num_workers == args.worker]
    overrides = {"device": args.device} if args.device else {}
    for pipeline in mine:
        print(f"[worker {args.worker}] {pipeline}", flush=True)
        try:
            run_pipeline_over_datasets(
                pipeline,
                args.datasets,
                args.out,
                overrides=overrides if pipeline not in ("constant-pii", "regex-pii") else {},
                batch_size=args.batch_size,
            )
        except Exception as error:  # a model that will not load must not stop the worker's other models
            print(f"[worker {args.worker}] {pipeline} failed to load: {error!r}", file=sys.stderr, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()

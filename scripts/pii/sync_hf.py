# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Keep a private Hub dataset in step with a running full PII benchmark (scripts/pii/run_full.sh).

    python scripts/pii/sync_hf.py ~/pii_full --repo argmaxinc/pii-benchmark-results --every 15

Every `--every` minutes, when new jobs have finished since the last pass, it rebuilds the report from the jobs
done so far, rebuilds the viewer tables of the datasets those jobs belong to, and uploads:

- `runs/`: each job's manifest.json, predictions.parquet, log.txt (and errors.json), the report (report.md,
  report.json, tables/*.csv), the dataset build list, the batching check, and the viewer page with its data;
- one dataset config per benchmark dataset: the wide per-segment table (text, gold spans, gold-censored text,
  and per pipeline its verdict, score, spans, censored text and outcome), for the Hub viewer and SQL console.

It stops after the pass that follows the end of the run (the run log names the viewer files). Needs a write token
(HF_TOKEN or `huggingface-cli login`).
"""

import argparse
import json
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import build_viewer  # noqa: E402
import report_full  # noqa: E402


RUN_FILES = [
    "*/*/manifest.json",
    "*/*/predictions.parquet",
    "*/*/log.txt",
    "*/*/errors.json",
    "report.md",
    "report.json",
    "tables/*.csv",
    "datasets.txt",
    "batching_check.txt",
    "viewer/web/index.html",
    "viewer/web/data/*.json",
]


def finished_jobs(out: Path) -> dict[str, set[str]]:
    by_dataset: dict[str, set[str]] = {}
    for path in out.glob("*/*/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
        except ValueError:
            continue
        if manifest.get("status") == "complete":
            by_dataset.setdefault(manifest["dataset"], set()).add(manifest["pipeline"])
    return by_dataset


def sync(out: Path, repo: str, changed: set[str], api) -> None:
    report_full.main(out)
    build_viewer.build(out, only=changed)
    build_viewer.push(out, repo, only=changed)
    api.upload_folder(
        folder_path=str(out),
        path_in_repo="runs",
        repo_id=repo,
        repo_type="dataset",
        allow_patterns=RUN_FILES,
        commit_message=f"runs: {sum(len(v) for v in finished_jobs(out).values())} jobs complete",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--every", type=float, default=15, help="Minutes between passes.")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    seen: dict[str, set[str]] = {}
    while True:
        ended = (
            "viewer files in" in (args.out / "run.log").read_text(errors="ignore")
            if (args.out / "run.log").exists()
            else False
        )
        jobs = finished_jobs(args.out)
        changed = {dataset for dataset, pipelines in jobs.items() if pipelines != seen.get(dataset)}
        if changed:
            started = time.time()
            try:
                sync(args.out, args.repo, changed, api)
                seen = {dataset: set(pipelines) for dataset, pipelines in jobs.items()}
                print(f"synced {len(changed)} datasets in {time.time() - started:.0f} s", flush=True)
            except Exception as error:  # a failed pass is retried on the next one
                print(f"sync failed: {error!r}", flush=True)
        if ended:
            # One last upload of everything, so the files the run's final step wrote (the full report, the viewer
            # page and its data) reach the Hub even when no job finished since the previous pass
            api.upload_folder(
                folder_path=str(args.out),
                path_in_repo="runs",
                repo_id=args.repo,
                repo_type="dataset",
                allow_patterns=RUN_FILES,
                commit_message="runs: final",
            )
            print("run finished and synced", flush=True)
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()

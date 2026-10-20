# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Per-segment viewer tables from a full PII run (scripts/pii/run_full.py), and optionally a private Hub push.

    python scripts/pii/build_viewer.py pii_full                        # writes pii_full/viewer/
    python scripts/pii/build_viewer.py pii_full --push argmaxinc/pii-benchmark-results

Writes:
- `viewer/<dataset>.parquet`: every segment, one row, wide: id, language, source category, text, gold spans,
  the gold-censored text, and per pipeline `<pipeline>__verdict`, `__score`, `__spans`, `__censored` and
  `__outcome` (tp / fn / fp / tn at the segment level). `--push` uploads each as a config of a private dataset, so
  the Hub viewer and SQL console can filter and sort them.
- `viewer/subset.json`: a seeded uniform sample per dataset (1,500 segments, or all of them) for the viewer page,
  with each pipeline's spans and scores (`__raw`) and the turns before the segment, so the page can re-score any
  subset under any choice of PII types without bias.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from report import masked_ranges  # noqa: E402


def censor(text: str, spans: list[tuple[int, int, str]]) -> str:
    """The text with every span replaced by [TYPE]; overlapping spans merge into the first."""
    pieces, position = [], 0
    for start, end, kind in sorted(spans):
        start, end = max(start, position), max(end, position)
        if end <= start:
            continue
        pieces.append(text[position:start])
        pieces.append(f"[{kind.upper()}]")
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def predicted_spans(prediction: dict, text: str) -> list[tuple[int, int, str]]:
    """The spans a prediction censors (at or above its bar), or the whole flagged text for a segment-level model."""
    from openbench.pipeline.safety.span_common import parse_spans

    parsed = parse_spans(prediction.get("raw_output"))
    if parsed is None:
        return [(a, b, "segment") for a, b in masked_ranges(prediction, text)]
    spans, bar = parsed
    return [(a, b, k) for a, b, k, score in spans if score >= bar]


def outcome(label: bool, flagged: bool) -> str:
    return {(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}[(label, flagged)]


def build(out: Path, subset_size: int = 1500, seed: int = 0, only: set[str] | None = None) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    from openbench.dataset import DatasetRegistry

    jobs: dict[str, dict[str, Path]] = defaultdict(dict)
    for manifest_path in sorted(out.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            jobs[manifest["dataset"]][manifest["pipeline"]] = manifest_path.parent / "predictions.parquet"
    viewer = out / "viewer"
    viewer.mkdir(exist_ok=True)
    subset_path = viewer / "subset.json"
    subsets: dict[str, list[dict]] = json.loads(subset_path.read_text()) if only and subset_path.exists() else {}
    for dataset, by_pipeline in sorted(jobs.items()):
        if only is not None and dataset not in only:
            continue
        rows = DatasetRegistry.get_alias_config(dataset).load()
        predictions = {
            pipeline: {row["id"]: row for row in pq.read_table(path).to_pylist()}
            for pipeline, path in by_pipeline.items()
        }
        pipelines = sorted(predictions)
        wide = []
        for row in rows:
            text = row["text"]
            gold = [(s["start"], s["end"], s["type"]) for s in row.get("pii_spans") or []]
            label = row["label"] == "unsafe"
            record = {
                "id": str(row["id"]),
                "dataset": dataset,
                "language": row.get("language") or "",
                "source_category": row.get("source_category") or "",
                "label": "pii" if label else "clean",
                "text": text,
                "speaker": row.get("speaker") or "",
                "conversation": json.dumps(
                    [[t["speaker"], t["text"], bool(t["target"])] for t in row.get("conversation") or []],
                    ensure_ascii=False,
                ),
                "gold_spans": json.dumps(gold),
                "gold_censored": censor(text, gold) if gold else "",
            }
            for pipeline in pipelines:
                found = predictions[pipeline].get(str(row["id"]))
                if found is None:
                    continue
                prediction = {"predicted_label": found["verdict"], "raw_output": found["spans"]}
                spans = predicted_spans(prediction, text) if found["verdict"] == "unsafe" or found["spans"] else []
                record[f"{pipeline}__verdict"] = found["verdict"]
                record[f"{pipeline}__score"] = found["score"]
                record[f"{pipeline}__spans"] = json.dumps(spans)
                record[f"{pipeline}__raw"] = found["spans"] or ""
                record[f"{pipeline}__censored"] = censor(text, spans)
                record[f"{pipeline}__outcome"] = outcome(label, found["verdict"] == "unsafe")
            wide.append(record)
        import pyarrow as pa

        # The Hub table leaves out the candidate spans below each bar (`__raw`); they stay in the job's parquet
        hub_rows = [{k: v for k, v in record.items() if not k.endswith("__raw")} for record in wide]
        pq.write_table(pa.Table.from_pylist(hub_rows), viewer / f"{dataset}.parquet")

        # A uniform random sample, so the metrics the page recomputes on any subset of it are unbiased
        rng = random.Random(seed)
        picked = wide if len(wide) <= subset_size else rng.sample(wide, subset_size)
        subsets[dataset] = [{k: v for k, v in record.items() if not k.endswith("__censored")} for record in picked]
        print(f"{dataset}: {len(wide)} rows, {len(pipelines)} pipelines, {len(picked)} in the viewer sample")
    subset_path.write_text(json.dumps(subsets, ensure_ascii=False))
    return subsets


def push(out: Path, repo: str, only: set[str] | None = None) -> None:
    from datasets import Dataset

    for path in sorted((out / "viewer").glob("*.parquet")):
        if only is not None and path.stem not in only:
            continue
        Dataset.from_parquet(str(path)).push_to_hub(repo, config_name=path.stem, split="test", private=True)
        print(f"pushed {path.stem} -> {repo} (private)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--subset-size", type=int, default=1500, help="Uniform sample per dataset in subset.json.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", metavar="REPO", help="Upload the wide tables to this private Hub dataset.")
    args = parser.parse_args()
    build(args.out, args.subset_size, args.seed)
    if args.push:
        push(args.out, args.push)


if __name__ == "__main__":
    main()

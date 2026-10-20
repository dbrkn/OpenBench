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
- `viewer/subset.json`: an error-weighted sample per dataset for the interactive viewer page: every segment the
  pipelines disagree on (up to a cap), each pipeline's misses and false masks, and a seeded random sample.
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
    try:
        return [(int(a), int(b), str(k)) for a, b, k in json.loads(prediction["raw_output"] or "")["spans"]]
    except (ValueError, KeyError, TypeError):
        whole = masked_ranges(prediction, text)
        return [(a, b, "segment") for a, b in whole]


def outcome(label: bool, flagged: bool) -> str:
    return {(True, True): "tp", (True, False): "fn", (False, True): "fp", (False, False): "tn"}[(label, flagged)]


def build(out: Path, subset_size: int, seed: int) -> dict[str, list[dict]]:
    import pyarrow.parquet as pq

    from openbench.dataset import DatasetRegistry

    jobs: dict[str, dict[str, Path]] = defaultdict(dict)
    for manifest_path in sorted(out.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "complete":
            jobs[manifest["dataset"]][manifest["pipeline"]] = manifest_path.parent / "predictions.parquet"
    viewer = out / "viewer"
    viewer.mkdir(exist_ok=True)
    subsets: dict[str, list[dict]] = {}
    for dataset, by_pipeline in sorted(jobs.items()):
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
                "context": "\n".join(f"{t['speaker']}: {t['text']}" for t in (row.get("conversation") or [])[:-1]),
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
                record[f"{pipeline}__censored"] = censor(text, spans)
                record[f"{pipeline}__outcome"] = outcome(label, found["verdict"] == "unsafe")
            wide.append(record)
        import pyarrow as pa

        pq.write_table(pa.Table.from_pylist(wide), viewer / f"{dataset}.parquet")

        rng = random.Random(seed)
        chosen: dict[str, set[str]] = defaultdict(set)
        by_id = {record["id"]: record for record in wide}
        disagree = [r["id"] for r in wide if len({r.get(f"{p}__verdict") for p in pipelines} - {None}) > 1]
        for reason, ids in [("disagreement", disagree)] + [
            (f"{kind} {p}", [r["id"] for r in wide if r.get(f"{p}__outcome") == kind])
            for p in pipelines
            if p != "constant-pii"
            for kind in ("fn", "fp")
        ]:
            for record_id in rng.sample(ids, min(len(ids), max(3, subset_size // (2 * len(pipelines) + 2)))):
                chosen[record_id].add(reason)
        for record_id in rng.sample(list(by_id), min(len(by_id), subset_size // 6)):
            chosen[record_id].add("random")
        picked = list(chosen)[:subset_size]
        subsets[dataset] = [
            {
                **{k: v for k, v in by_id[i].items() if not k.endswith(("__censored",))},
                "why": sorted(chosen[i]),
            }
            for i in picked
        ]
        print(f"{dataset}: {len(wide)} rows, {len(pipelines)} pipelines, {len(picked)} in the viewer subset")
    (viewer / "subset.json").write_text(json.dumps(subsets, ensure_ascii=False))
    return subsets


def push(out: Path, repo: str) -> None:
    from datasets import Dataset

    for path in sorted((out / "viewer").glob("*.parquet")):
        Dataset.from_parquet(str(path)).push_to_hub(repo, config_name=path.stem, split="test", private=True)
        print(f"pushed {path.stem} -> {repo} (private)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--subset-size", type=int, default=400, help="Segments per dataset in subset.json.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--push", metavar="REPO", help="Upload the wide tables to this private Hub dataset.")
    args = parser.parse_args()
    build(args.out, args.subset_size, args.seed)
    if args.push:
        push(args.out, args.push)


if __name__ == "__main__":
    main()

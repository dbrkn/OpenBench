# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""The full PII benchmark report, from the parquet outputs of scripts/pii/run_full.py.

    python scripts/pii/report_full.py pii_full

Writes `<out>/report.md`, `<out>/report.json` and `<out>/tables/<dataset>.csv`. Per (dataset, pipeline):

- per segment: recall and false-mask rate with 95 % Wilson intervals, precision, recall at a 1 / 5 / 10 / 20 %
  false-mask rate (the bar set on the clean segments' scores), ROC AUC;
- per character (datasets with gold spans): PII characters censored, clean characters censored;
- per PII type: the share of gold spans at least half censored;
- per language (multilingual sets): recall at a 5 % false-mask rate;
- latency mean / p50 / p95 and error count, from the manifest and the parquet;
- a contamination flag where the model's card names the dataset among its training data.
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent))
from report import MASK_RATES, character_scores, masked_ranges, recall_at_mask_rate  # noqa: E402


# Model -> dataset families its card lists as training data (prefix match on the dataset alias)
TRAINED_ON = {
    "openai-privacy-filter": ["openpii", "ai4privacy"],
    "openmed-privacy-filter-multilingual": ["openpii", "ai4privacy", "nemotron", "gretel", "privy"],
    "gliner-pii-nvidia": ["nemotron"],
    "gliner-pii-urchade-multi": ["urchade-mistral"],
    "pii-tracer": ["pii-trace"],
}


def contaminated(pipeline: str, dataset: str) -> bool:
    return any(
        pipeline.startswith(model) and any(dataset.startswith(family) for family in families)
        for model, families in TRAINED_ON.items()
    )


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total == 0:
        return None
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


def references(dataset: str) -> dict[str, dict]:
    from openbench.dataset import DatasetRegistry

    rows = DatasetRegistry.get_alias_config(dataset).load()
    return {
        str(row["id"]): {
            "label": row["label"] == "unsafe",
            "text": row["text"],
            "spans": row.get("pii_spans") or [],
            "language": row.get("language") or "",
        }
        for row in rows
    }


def type_recall(pairs: list[tuple[dict, dict]]) -> dict[str, tuple[int, int]]:
    """Per gold type: (spans at least half censored, spans)."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for reference, prediction in pairs:
        masked = np.zeros(len(reference["text"]), dtype=bool)
        for start, end in masked_ranges(prediction, reference["text"]):
            masked[max(0, start) : end] = True
        for span in reference["spans"]:
            length = max(1, span["end"] - span["start"])
            covered = masked[span["start"] : span["end"]].sum()
            counts[span["type"]][1] += 1
            counts[span["type"]][0] += int(covered * 2 >= length)
    return {kind: (hit, total) for kind, (hit, total) in counts.items()}


def score_job(refs: dict[str, dict], parquet: Path, manifest: dict) -> dict:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet).to_pylist()
    pairs = [
        (
            refs[row["id"]],
            {"predicted_label": row["verdict"], "unsafe_score": row["score"], "raw_output": row["spans"]},
        )
        for row in table
        if row["id"] in refs
    ]
    labels = np.array([ref["label"] for ref, _ in pairs])
    flagged = np.array([pred["predicted_label"] == "unsafe" for _, pred in pairs])
    scores = np.array([float(pred["unsafe_score"] or 0.0) for _, pred in pairs])
    tp, fp = int((labels & flagged).sum()), int((~labels & flagged).sum())
    positives, negatives = int(labels.sum()), int((~labels).sum())
    latencies = np.array([row["latency_ms"] for row in table if row.get("latency_ms") is not None])
    result = {
        "n": len(pairs),
        "pii": positives,
        "recall": tp / positives if positives else None,
        "recall_ci": wilson(tp, positives),
        "false_mask": fp / negatives if negatives else None,
        "false_mask_ci": wilson(fp, negatives),
        "precision": tp / (tp + fp) if tp + fp else None,
        "errors": int(manifest.get("errors", 0)),
        "ms_mean": float(latencies.mean()) if len(latencies) else None,
        "ms_p50": float(np.percentile(latencies, 50)) if len(latencies) else None,
        "ms_p95": float(np.percentile(latencies, 95)) if len(latencies) else None,
        "device": (manifest.get("device") or {}).get("gpu") or "cpu",
        "model_revision": manifest.get("model_revision"),
    }
    for rate in MASK_RATES:
        result[f"recall@{rate:.0%}"] = recall_at_mask_rate(scores, labels, rate)
    try:
        from sklearn.metrics import roc_auc_score

        result["roc_auc"] = float(roc_auc_score(labels, scores)) if 0 < positives < len(labels) else None
    except ImportError:
        result["roc_auc"] = None
    result["chars"] = character_scores(pairs)
    result["types"] = type_recall(pairs)
    by_language: dict[str, list[int]] = defaultdict(list)
    for index, (ref, _) in enumerate(pairs):
        by_language[ref["language"]].append(index)
    if len(by_language) > 1:
        result["languages"] = {
            language: recall_at_mask_rate(scores[idx], labels[idx], 0.05)
            for language, idx in sorted(by_language.items())
        }
    return result


def fmt(value, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def ci(pair) -> str:
    return f"[{pair[0]:.2f}, {pair[1]:.2f}]" if pair else ""


def main(out: Path) -> None:
    jobs: dict[str, dict[str, tuple[Path, dict]]] = defaultdict(dict)
    for manifest_path in sorted(out.glob("*/*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        parquet = manifest_path.parent / "predictions.parquet"
        if manifest.get("status") == "complete" and parquet.exists():
            jobs[manifest["dataset"]][manifest["pipeline"]] = (parquet, manifest)

    summary: dict[str, dict] = {}
    lines = ["# PII benchmark, full run", ""]
    (out / "tables").mkdir(exist_ok=True)
    for dataset, by_pipeline in sorted(jobs.items()):
        refs = references(dataset)
        scored = {
            pipeline: score_job(refs, parquet, manifest)
            for pipeline, (parquet, manifest) in sorted(by_pipeline.items())
        }
        summary[dataset] = scored
        first = next(iter(scored.values()))
        lines += [
            f"## {dataset} ({first['n']:,} segments, {fmt(first['pii'] / max(1, first['n']), True)} with PII)",
            "",
        ]
        header = [
            "pipeline",
            "recall [95% CI]",
            "false-mask [95% CI]",
            "precision",
            *[f"recall@{rate:.0%}" for rate in MASK_RATES],
            "ROC AUC",
            "PII chars",
            "clean chars",
            "ms p50 / p95",
            "errors",
        ]
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        csv_rows = [",".join(h.replace(",", ";") for h in header)]
        ranked = sorted(scored.items(), key=lambda item: -(item[1].get("recall@5%") or -1))
        for pipeline, result in ranked:
            chars = result["chars"] or (None, None)
            name = pipeline + (" ⚠ trained on" if contaminated(pipeline, dataset) else "")
            cells = [
                name,
                f"{fmt(result['recall'])} {ci(result['recall_ci'])}",
                f"{fmt(result['false_mask'], True)} {ci(result['false_mask_ci'])}",
                fmt(result["precision"]),
                *[fmt(result[f"recall@{rate:.0%}"]) for rate in MASK_RATES],
                fmt(result["roc_auc"]),
                fmt(chars[0], True),
                fmt(chars[1], True),
                f"{result['ms_p50']:.1f} / {result['ms_p95']:.1f}" if result["ms_p50"] is not None else "-",
                str(result["errors"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
            csv_rows.append(",".join(c.replace(",", ";") for c in cells))
        (out / "tables" / f"{dataset}.csv").write_text("\n".join(csv_rows) + "\n")
        type_names = sorted({kind for result in scored.values() for kind in result["types"]})
        if type_names:
            top = sorted(type_names, key=lambda kind: -max(r["types"].get(kind, (0, 0))[1] for r in scored.values()))[
                :12
            ]
            lines += ["", f"Gold spans at least half censored, by type ({dataset}):", ""]
            lines += ["| pipeline | " + " | ".join(top) + " |", "|" + "---|" * (len(top) + 1)]
            for pipeline, result in ranked:
                cells = []
                for kind in top:
                    hit, total = result["types"].get(kind, (0, 0))
                    cells.append(f"{hit / total:.2f}" if total else "-")
                lines.append(f"| {pipeline} | " + " | ".join(cells) + " |")
        languages = sorted({lang for result in scored.values() for lang in (result.get("languages") or {})})
        if languages:
            lines += ["", f"Recall at a 5 % false-mask rate by language ({dataset}):", ""]
            lines += ["| pipeline | " + " | ".join(languages) + " |", "|" + "---|" * (len(languages) + 1)]
            for pipeline, result in ranked:
                values = result.get("languages") or {}
                lines.append(f"| {pipeline} | " + " | ".join(fmt(values.get(lang)) for lang in languages) + " |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    (out / "report.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pii_full"))

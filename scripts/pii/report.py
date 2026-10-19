# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Collect the PII benchmark runs under OUT/<pipeline>/<dataset>/ into one Markdown report (OUT/report.md).

Every run's predictions are joined with the dataset's rows (rebuilt from the
built-dataset cache) by row id, and scored two ways:

- Per segment, the masking decision: recall (PII segments censored), the
  false-mask rate (clean segments censored), and recall at a fixed false-mask
  rate (1, 5, 10, 20 %), the threshold set on the clean segments' scores. A
  censored segment stays recoverable, so recall comes first and the mask rate
  is the price.
- Per character, on datasets whose PII rows all carry gold spans: the share of
  PII characters censored and the share of clean characters censored. A flagged
  segment is censored whole for a sentence classifier (Roblox); Privacy Filter
  censors its own spans (its `raw_output`).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np


MASK_RATES = (0.01, 0.05, 0.10, 0.20)


def latest_results_dir(job_dir: Path) -> Path | None:
    found = sorted(job_dir.glob("outputs/*/*/pii_results/results/sample_results_table.csv"))
    return found[-1].parent if found else None


def load_predictions(results_dir: Path) -> dict[str, dict]:
    with open(results_dir / "sample_results_table.csv", newline="") as handle:
        return {row["audio_name"]: row for row in csv.DictReader(handle)}


def load_references(dataset: str) -> dict[str, dict]:
    from openbench.dataset import DatasetRegistry

    rows = DatasetRegistry.get_alias_config(dataset).load()
    return {
        row["id"]: {"label": row["label"] == "unsafe", "text": row["text"], "spans": row.get("pii_spans") or []}
        for row in rows
    }


def recall_at_mask_rate(scores: np.ndarray, labels: np.ndarray, rate: float) -> float | None:
    """Recall when the bar is set so that `rate` of the clean segments score above it."""
    clean, pii = scores[~labels], scores[labels]
    if not len(clean) or not len(pii):
        return None
    bar = np.quantile(clean, 1 - rate, method="higher")
    return float(np.mean(pii > bar))


def masked_ranges(prediction: dict, text: str) -> list[tuple[int, int]]:
    """Character ranges censored by a prediction: its spans when it emits them, else the whole flagged text."""
    try:
        spans = json.loads(prediction.get("raw_output") or "")["spans"]
        return [(int(start), int(end)) for start, end, _ in spans]
    except (ValueError, KeyError, TypeError):
        return [(0, len(text))] if prediction["predicted_label"] == "unsafe" else []


def character_scores(pairs: list[tuple[dict, dict]]) -> tuple[float, float] | None:
    """(PII characters censored, clean characters censored), or None without gold spans on every PII row."""
    if any(reference["label"] and not reference["spans"] for reference, _ in pairs):
        return None
    pii_total = pii_hit = clean_total = clean_hit = 0
    for reference, prediction in pairs:
        text = reference["text"]
        gold = np.zeros(len(text), dtype=bool)
        for span in reference["spans"]:
            gold[max(0, span["start"]) : span["end"]] = True
        masked = np.zeros(len(text), dtype=bool)
        for start, end in masked_ranges(prediction, text):
            masked[max(0, start) : end] = True
        visible = np.array([not c.isspace() for c in text], dtype=bool)
        pii_total += int((gold & visible).sum())
        pii_hit += int((gold & masked & visible).sum())
        clean_total += int((~gold & visible).sum())
        clean_hit += int((~gold & masked & visible).sum())
    if not pii_total or not clean_total:
        return None
    return pii_hit / pii_total, clean_hit / clean_total


def score(pairs: list[tuple[dict, dict]]) -> dict:
    labels = np.array([reference["label"] for reference, _ in pairs])
    flagged = np.array([prediction["predicted_label"] == "unsafe" for _, prediction in pairs])
    scores = np.array([float(prediction.get("unsafe_score") or 0.0) for _, prediction in pairs])
    times = [float(p["prediction_time"]) for _, p in pairs if p.get("prediction_time")]
    tp = int((labels & flagged).sum())
    result = {
        "n": len(pairs),
        "pii_share": float(labels.mean()) if len(labels) else None,
        "recall": tp / labels.sum() if labels.sum() else None,
        "precision": tp / flagged.sum() if flagged.sum() else None,
        "false_mask": float((flagged & ~labels).sum() / (~labels).sum()) if (~labels).sum() else None,
        "ms": 1000 * sum(times) / len(times) if times else None,
    }
    for rate in MASK_RATES:
        result[f"recall@{rate:.0%}"] = recall_at_mask_rate(scores, labels, rate)
    try:
        from sklearn.metrics import roc_auc_score

        result["roc_auc"] = float(roc_auc_score(labels, scores)) if 0 < labels.sum() < len(labels) else None
    except ImportError:
        result["roc_auc"] = None
    result["chars"] = character_scores(pairs)
    return result


def fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def main(out: Path) -> None:
    jobs: dict[str, dict[str, Path]] = {}
    for job_dir in sorted(p for p in out.glob("*/*") if p.is_dir()):
        results_dir = latest_results_dir(job_dir)
        if results_dir is not None:
            jobs.setdefault(job_dir.name, {})[job_dir.parent.name] = results_dir

    lines = ["# PII benchmark", ""]
    summary = {}
    rate_columns = [f"recall@{rate:.0%} mask" for rate in MASK_RATES]
    for dataset, by_pipeline in sorted(jobs.items()):
        references = load_references(dataset)
        scored = {}
        for pipeline, results_dir in sorted(by_pipeline.items()):
            predictions = load_predictions(results_dir)
            pairs = [(references[i], p) for i, p in predictions.items() if i in references]
            if pairs:
                scored[pipeline] = score(pairs)
        if not scored:
            continue
        first = next(iter(scored.values()))
        lines += [f"## {dataset} ({first['n']} segments, {fmt(first['pii_share'], True)} with PII)", ""]
        header = ["pipeline", "recall", "false-mask rate", "precision", *rate_columns, "ROC AUC"]
        has_chars = any(result["chars"] for result in scored.values())
        if has_chars:
            header += ["PII chars masked", "clean chars masked"]
        header.append("ms/segment")
        lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for pipeline, result in scored.items():
            cells = [
                pipeline,
                fmt(result["recall"]),
                fmt(result["false_mask"], True),
                fmt(result["precision"]),
                *[fmt(result[f"recall@{rate:.0%}"]) for rate in MASK_RATES],
                fmt(result["roc_auc"]),
            ]
            if has_chars:
                chars = result["chars"] or (None, None)
                cells += [fmt(chars[0], True), fmt(chars[1], True)]
            cells.append(f"{result['ms']:.1f}" if result["ms"] is not None else "-")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        summary[dataset] = scored
    report = "\n".join(lines)
    (out / "report.md").write_text(report)
    (out / "report.json").write_text(json.dumps(summary, indent=1))
    print(report)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pii_results"))

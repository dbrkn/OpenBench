# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Collect the PII benchmark runs under OUT/<pipeline>/<dataset>/ into one Markdown report (OUT/report.md).

Beside the runner's metrics it gives the false-mask rate, the share of PII-free
turns that would be masked: FP / negatives = R * pi * (1 - P) / (P * (1 - pi)),
with the PII share pi read off `constant-pii`, whose precision is exactly pi.
"""

import csv
import json
import sys
from pathlib import Path


COLUMNS = ("recall", "precision", "f1", "mcc", "roc_auc")


def latest_results(job_dir: Path) -> tuple[dict, Path] | None:
    found = sorted(job_dir.glob("outputs/*/*/pii_results/results/global_results.json"))
    return (json.loads(found[-1].read_text()), found[-1].parent) if found else None


def mean_ms(results_dir: Path) -> float | None:
    path = results_dir / "sample_results_table.csv"
    if not path.exists():
        return None
    with open(path, newline="") as handle:
        times = [float(row["prediction_time"]) for row in csv.DictReader(handle) if row.get("prediction_time")]
    return 1000 * sum(times) / len(times) if times else None


def false_mask_rate(metrics: dict, share: float | None) -> float | None:
    recall, precision = metrics.get("recall"), metrics.get("precision")
    if share is None or recall is None or not precision or share >= 1:
        return None
    return recall * share * (1 - precision) / (precision * (1 - share))


def main(out: Path) -> None:
    runs: dict[str, dict[str, tuple[dict, Path]]] = {}
    for job_dir in sorted(p for p in out.glob("*/*") if p.is_dir()):
        result = latest_results(job_dir)
        if result is not None:
            runs.setdefault(job_dir.name, {})[job_dir.parent.name] = result
    lines = ["# PII benchmark", ""]
    for dataset, by_pipeline in sorted(runs.items()):
        constant = by_pipeline.get("constant-pii")
        share = constant[0].get("precision") if constant else None
        lines += [f"## {dataset}" + (f" (PII share {share:.1%})" if share is not None else ""), ""]
        lines += [
            "| pipeline | " + " | ".join(COLUMNS) + " | false-mask rate | ms/turn |",
            "|" + "---|" * (len(COLUMNS) + 3),
        ]
        for pipeline, (metrics, results_dir) in sorted(by_pipeline.items()):
            cells = [f"{metrics[c]:.3f}" if isinstance(metrics.get(c), (int, float)) else "-" for c in COLUMNS]
            rate = false_mask_rate(metrics, share)
            ms = mean_ms(results_dir)
            cells += [f"{rate:.3f}" if rate is not None else "-", f"{ms:.1f}" if ms is not None else "-"]
            lines.append(f"| {pipeline} | " + " | ".join(cells) + " |")
        lines.append("")
    report = "\n".join(lines)
    (out / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pii_results"))

# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Compact data files for the Redaction Bench viewer page (scripts/pii/viewer.html), from a full run.

    python scripts/pii/export_viewer.py pii_full          # after report_full.py and build_viewer.py

Writes `<out>/viewer/web/`: `index.html` (a copy of viewer.html), `data/summary.json` (every dataset's size, PII
share and each pipeline's headline metrics from report.json) and `data/<dataset>.json` (the error-weighted subset
from build_viewer.py: text, the speaker's preceding turns, gold spans and each pipeline's verdict, score and
spans). Censored views are drawn in the page from the spans, so the files stay small.
"""

import json
import shutil
import sys
from pathlib import Path


def short(value, digits: int = 4):
    return round(value, digits) if isinstance(value, float) else value


def main(out: Path) -> None:
    viewer = out / "viewer"
    web = viewer / "web"
    (web / "data").mkdir(parents=True, exist_ok=True)
    report = json.loads((out / "report.json").read_text())
    subsets = json.loads((viewer / "subset.json").read_text())

    summary = []
    for dataset, by_pipeline in sorted(report.items()):
        first = next(iter(by_pipeline.values()))
        summary.append(
            {
                "name": dataset,
                "n": first["n"],
                "pii": first["pii"],
                "models": {
                    pipeline: {
                        "recall5": short(r.get("recall@5%")),
                        "recall10": short(r.get("recall@10%")),
                        "recall": short(r.get("recall")),
                        "false_mask": short(r.get("false_mask")),
                        "precision": short(r.get("precision")),
                        "auc": short(r.get("roc_auc")),
                        "pii_chars": short((r.get("chars") or [None, None])[0]),
                        "clean_chars": short((r.get("chars") or [None, None])[1]),
                        "ms": short(r.get("ms_p50"), 2),
                        "device": r.get("device") or "",
                    }
                    for pipeline, r in by_pipeline.items()
                },
            }
        )
    (web / "data" / "summary.json").write_text(json.dumps({"datasets": summary}, separators=(",", ":")))

    for dataset, rows in subsets.items():
        pipelines = sorted({k.split("__")[0] for row in rows for k in row if k.endswith("__verdict")})
        compact = []
        for row in rows:
            predictions = {}
            for pipeline in pipelines:
                if f"{pipeline}__verdict" not in row:
                    continue
                predictions[pipeline] = [
                    1 if row[f"{pipeline}__verdict"] == "unsafe" else 0,
                    short(row.get(f"{pipeline}__score") or 0.0, 3),
                    json.loads(row.get(f"{pipeline}__spans") or "[]"),
                ]
            compact.append(
                {
                    "id": row["id"],
                    "lang": row.get("language") or "",
                    "cat": row.get("source_category") or "",
                    "pii": row["label"] == "pii",
                    "text": row["text"],
                    "ctx": (row.get("context") or "")[-600:],
                    "gold": json.loads(row.get("gold_spans") or "[]"),
                    "why": row.get("why") or [],
                    "p": predictions,
                }
            )
        payload = {"dataset": dataset, "pipelines": pipelines, "rows": compact}
        (web / "data" / f"{dataset}.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        print(f"{dataset}: {len(compact)} segments, {len(pipelines)} pipelines")
    shutil.copy(Path(__file__).with_name("viewer.html"), web / "index.html")
    print(f"viewer files in {web}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pii_full"))

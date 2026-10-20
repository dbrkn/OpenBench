# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Data files for the Redaction Bench viewer page (scripts/pii/viewer.html), from a full run.

    python scripts/pii/export_viewer.py pii_full          # after report_full.py and build_viewer.py

Writes `<out>/viewer/web/`: `index.html` (a copy of viewer.html), `data/summary.json` (each dataset's size, PII
share, gold types and every pipeline's full-run metrics from report.json) and `data/<dataset>.json`: the uniform
sample from build_viewer.py with, per segment, its text, gold spans, the turns before it (each marked as the
speaker's own or someone else's) and each pipeline's verdict, score and spans with their scores. The page
re-scores any subset under any choice of gold and predicted PII types from these.
"""

import json
import re
import shutil
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

MAX_TURNS_BEFORE = 4
MAX_TURN_CHARS = 400


def short(value, digits: int = 4):
    return round(value, digits) if isinstance(value, float) else value


def context_setting(pipeline: str) -> dict:
    """How many earlier turns a pipeline read, and whose (from its alias)."""
    if pipeline.endswith("-owner-context-2"):
        return {"turns": 2, "speakers": "own"}
    if pipeline.endswith("-owner-context"):
        return {"turns": 16, "speakers": "own"}
    if pipeline.endswith("-context") or pipeline.startswith("roblox-pii-v2") and not pipeline.endswith("target-only"):
        return {"turns": 16, "speakers": "all"}
    return {"turns": 0, "speakers": "own"}


def parse(raw: str):
    from openbench.pipeline.safety.span_common import parse_spans

    return parse_spans(raw)


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
        names = sorted({k.split("__")[0] for row in rows for k in row if k.endswith("__verdict")})
        gold_types = sorted({span[2] for row in rows for span in json.loads(row.get("gold_spans") or "[]")})
        gold_index = {kind: i for i, kind in enumerate(gold_types)}
        pipelines, type_index = [], {}
        for name in names:
            kinds, bar, span_model = set(), 0.0, False
            for row in rows:
                parsed = parse(row.get(f"{name}__raw") or "")
                if parsed is not None:
                    span_model = True
                    bar = parsed[1]
                    kinds.update(kind for _, _, kind, _ in parsed[0])
            type_index[name] = {kind: i for i, kind in enumerate(sorted(kinds))}
            pipelines.append(
                {"id": name, "types": sorted(kinds), "bar": bar, "spans": span_model, **context_setting(name)}
            )
        compact = []
        for row in rows:
            conversation = json.loads(row.get("conversation") or "[]")
            target_at = max((i for i, turn in enumerate(conversation) if turn[2]), default=len(conversation))
            before = conversation[max(0, target_at - MAX_TURNS_BEFORE) : target_at]
            own = row.get("speaker") or ""
            predictions = []
            for name in names:
                if f"{name}__verdict" not in row:
                    predictions.append(None)
                    continue
                parsed = parse(row.get(f"{name}__raw") or "")
                spans = (
                    [[s, e, type_index[name][k], round(sc, 3)] for s, e, k, sc in parsed[0]] if parsed is not None else []
                )
                predictions.append([1 if row[f"{name}__verdict"] == "unsafe" else 0, short(row.get(f"{name}__score") or 0.0, 3), spans])
            compact.append(
                {
                    "id": row["id"],
                    "lang": row.get("language") or "",
                    "cat": row.get("source_category") or "",
                    "text": row["text"],
                    "gold": [[s, e, gold_index[k]] for s, e, k in json.loads(row.get("gold_spans") or "[]")],
                    "before": [
                        [1 if (turn[0] == own or turn[0] == "t") else 0, turn[0], turn[1][:MAX_TURN_CHARS]]
                        for turn in before
                    ],
                    "p": predictions,
                }
            )
        payload = {"dataset": dataset, "gold_types": gold_types, "pipelines": pipelines, "rows": compact}
        path = web / "data" / f"{re.sub(r'[^A-Za-z0-9._-]', '_', dataset)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        print(f"{dataset}: {len(compact)} segments, {len(pipelines)} pipelines, {path.stat().st_size / 1e6:.1f} MB")
    shutil.copy(Path(__file__).with_name("viewer.html"), web / "index.html")
    print(f"viewer files in {web}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "pii_full"))

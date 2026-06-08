"""Push per-sample evaluation results from a completed run to HuggingFace.

Usage:
    python scripts/push_results_to_hf.py \
        --results-csv outputs/2026-06-07/13-22-43/results/task_results_table.csv \
        --dataset-id argmaxinc/qwen-tts-medusa-fleurs-coreml-evals \
        --pipeline-name whisperkitpro-parakeet-v3-compressed \
        --language en
"""

import argparse

import pandas as pd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Push per-sample results to HuggingFace dataset")
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="Path to task_results_table.csv from a completed evaluation run",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        required=True,
        help="HuggingFace dataset ID (e.g. argmaxinc/qwen-tts-medusa-fleurs-coreml-evals)",
    )
    parser.add_argument(
        "--pipeline-name",
        type=str,
        required=True,
        help="Pipeline name used for the column prefix (e.g. whisperkitpro-parakeet-v3-compressed)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language filter that was used during the evaluation (e.g. 'en')",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    args = parser.parse_args()

    # Read results CSV
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} task results from {args.results_csv}")

    # Load full dataset from HF
    print(f"Loading dataset {args.dataset_id} (split={args.split})...")
    full_ds = load_dataset(args.dataset_id, split=args.split, verification_mode="no_checks")
    print(f"Dataset has {len(full_ds)} rows")

    # Build filtered index mapping
    if args.language is not None:
        filtered_indices = [i for i in range(len(full_ds)) if full_ds[i]["language"] == args.language]
        print(f"Filtered to {len(filtered_indices)} rows with language='{args.language}'")
    else:
        filtered_indices = list(range(len(full_ds)))

    # Sanitize pipeline name for column naming
    col_prefix = args.pipeline_name.replace("-", "_")

    # Group by metric and add columns
    for metric_name, metric_df in df.groupby("metric_name"):
        col_name = f"{metric_name}_{col_prefix}"
        values = [None] * len(full_ds)

        for _, row in metric_df.iterrows():
            sample_id = int(row["sample_id"])
            if sample_id < len(filtered_indices):
                original_idx = filtered_indices[sample_id]
                values[original_idx] = row["result"]

        # Remove column if it already exists
        if col_name in full_ds.column_names:
            full_ds = full_ds.remove_columns(col_name)
        full_ds = full_ds.add_column(col_name, values)

        populated = sum(1 for v in values if v is not None)
        print(f"  Added column '{col_name}' ({populated}/{len(values)} values populated)")

    # Push to hub
    print(f"Pushing to {args.dataset_id}...")
    full_ds.push_to_hub(args.dataset_id, split=args.split)
    print(f"Done! Results at https://huggingface.co/datasets/{args.dataset_id}")


if __name__ == "__main__":
    main()

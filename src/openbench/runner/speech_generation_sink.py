# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2026 Argmax, Inc. All Rights Reserved.

"""Incremental, append-only results sink for speech-generation benchmarks.

Buffers per-sample rows and, every `flush_every` samples, writes a *new*
parquet shard to a Hugging Face dataset repo (``data/chunk-NNNNN.parquet``).
Each flush only uploads a new file — existing shards are never rewritten — so
results accumulate safely and the HF dataset viewer auto-concatenates them.

The schema mirrors the source seedTTS-eval dataset (``text``, ``language``,
``sample_idx``, ``audio``) and adds ``reference_audio``, ``generated_audio``
(both playable Audio columns, embedded in the parquet) plus per-sample ``SIM``
and ``WER``.
"""

import re
import tempfile
from pathlib import Path

from argmaxtools.utils import get_logger
from huggingface_hub import HfApi


logger = get_logger(__name__)

# Where shards live inside the repo. The HF datasets viewer auto-loads every
# parquet file under `data/`, concatenating them into a single table.
_DATA_DIR = "data"
_CHUNK_RE = re.compile(r"chunk-(\d+)\.parquet$")


class SpeechGenerationResultSink:
    """Accumulates per-sample rows and flushes parquet shards to an HF dataset."""

    def __init__(self, repo_id: str, flush_every: int = 100, private: bool = True) -> None:
        self.repo_id = repo_id
        self.flush_every = max(1, int(flush_every))
        self._buffer: list[dict] = []
        self._api = HfApi()

        # Ensure the repo exists (idempotent) and resume the chunk counter past
        # any shards already present, so re-runs append instead of overwriting.
        self._api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=private)
        self._chunk_index = self._next_chunk_index()
        logger.info(
            f"Speech-generation results sink → hf://datasets/{repo_id} "
            f"(flush every {self.flush_every}, starting at chunk {self._chunk_index})"
        )

    def _next_chunk_index(self) -> int:
        try:
            files = self._api.list_repo_files(self.repo_id, repo_type="dataset")
        except Exception as e:  # noqa: BLE001 - fresh/empty repo or transient API error
            logger.warning(f"Could not list existing shards in {self.repo_id}: {e}")
            return 0
        indices = [int(m.group(1)) for f in files if (m := _CHUNK_RE.search(f))]
        return max(indices) + 1 if indices else 0

    def add(self, row: dict) -> None:
        """Buffer one per-sample row, flushing automatically at the threshold."""
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as a new parquet shard and upload it."""
        if not self._buffer:
            return

        # Imported lazily so importing the runner never pulls in the (heavy)
        # datasets Audio stack unless a sink is actually used.
        from datasets import Audio, Dataset, Features, Value

        features = Features(
            {
                "sample_idx": Value("string"),
                "text": Value("string"),
                "language": Value("string"),
                "reference_audio": Audio(),
                "generated_audio": Audio(),
                "SIM": Value("float32"),
                "WER": Value("float32"),
            }
        )
        rows, self._buffer = self._buffer, []
        dataset = Dataset.from_list(rows, features=features)

        chunk_name = f"chunk-{self._chunk_index:05d}.parquet"
        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / chunk_name
            # to_parquet embeds the Audio columns as bytes, so each shard is
            # self-contained (no external file references).
            dataset.to_parquet(str(local_path))
            self._api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=f"{_DATA_DIR}/{chunk_name}",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Add {len(rows)} samples (chunk {self._chunk_index})",
            )
        logger.info(f"Uploaded {len(rows)} samples → {self.repo_id}:{_DATA_DIR}/{chunk_name}")
        self._chunk_index += 1

    def close(self) -> None:
        """Flush any remaining buffered rows."""
        self.flush()

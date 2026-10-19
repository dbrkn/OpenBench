#!/usr/bin/env bash
# Run every (pipeline, dataset) pair of the PII benchmark as its own openbench-cli process, $JOBS at a time,
# spread round-robin over the visible GPUs. Results land in $OUT/<pipeline>/<dataset>/ with one log per job.
#
#   scripts/pii/run.sh                 # full datasets
#   MINI=1 scripts/pii/run.sh          # the seeded 2,000-row -mini variants (smoke test)
#   PIPELINES="openai-privacy-filter" DATASETS="spokenwoz abcd" scripts/pii/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"
PII_ENV="$(cd "${PII_ENV:-.venv-pii}" && pwd)"
OUT="$(mkdir -p "${OUT:-pii_results}" && cd "${OUT:-pii_results}" && pwd)"
JOBS="${JOBS:-4}"
PIPELINES="${PIPELINES:-roblox-pii-v2-target-only roblox-pii-v2-owner-context-2 roblox-pii-v2-owner-context roblox-pii-v2 openai-privacy-filter openai-privacy-filter-owner-context-2 openai-privacy-filter-owner-context openai-privacy-filter-context constant-pii}"
DATASETS="${DATASETS:-roblox-pii-chat-giving nemotron-pii nemotron-pii-sentences openpii dialogpii dialogpii-asr pii-trace spokenwoz abcd}"
METRICS="-m recall -m precision -m f1 -m accuracy -m mcc -m roc_auc -m category_recall"
export WANDB_MODE="${WANDB_MODE:-disabled}" TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore
GPUS="$("$PII_ENV/bin/python" -c 'import torch; print(max(1, torch.cuda.device_count()))')"

# Build the dataset caches once, before the jobs race for them. The hard exit skips interpreter shutdown, where a
# streamed dataset's background threads can hang the process after the build is saved.
for dataset in $DATASETS; do
  name="$dataset"; [ "${MINI:-0}" = 1 ] && name="$dataset-mini"
  echo "building $name"
  "$PII_ENV/bin/python" -W ignore -c "from openbench.dataset import DatasetRegistry as R; import os, sys; print(len(R.get_alias_config('$name').load()), 'rows'); sys.stdout.flush(); os._exit(0)" \
    2>/dev/null | tail -1 || echo "warning: could not pre-build $name"
done

jobs_file="$(mktemp)"
index=0
# Roblox's own benchmark is read with every head (its unsafe rows carry any of the three labels).
for pipeline in $PIPELINES roblox-pii-v2-any-head; do
  for dataset in $DATASETS roblox-pii-chat; do
    case "$pipeline:$dataset" in
      roblox-pii-v2-any-head:roblox-pii-chat | constant-pii:roblox-pii-chat) ;;
      roblox-pii-v2-any-head:* | *:roblox-pii-chat) continue ;;
    esac
    name="$dataset"; [ "${MINI:-0}" = 1 ] && name="$dataset-mini"
    echo "$((index % GPUS)) $pipeline $name" >> "$jobs_file"
    index=$((index + 1))
  done
done
echo "$(wc -l < "$jobs_file") jobs on $GPUS GPU(s), $JOBS at a time -> $OUT"

run_job() {
  local gpu="$1" pipeline="$2" dataset="$3" dir="$OUT/$2/$3"
  mkdir -p "$dir"
  if grep -q "Evaluation completed successfully" "$dir/log.txt" 2>/dev/null; then echo "skip $pipeline $dataset"; return; fi
  (cd "$dir" && CUDA_VISIBLE_DEVICES="$gpu" "$PII_ENV/bin/openbench-cli" evaluate -p "$pipeline" -d "$dataset" $METRICS) \
    > "$dir/log.txt" 2>&1 && echo "done $pipeline $dataset" || echo "FAILED $pipeline $dataset (see $dir/log.txt)"
}
export -f run_job
export OUT PII_ENV METRICS ROOT
xargs -P "$JOBS" -L 1 bash -c 'run_job "$@"' _ < "$jobs_file"
rm -f "$jobs_file"
"$PII_ENV/bin/python" scripts/pii/report.py "$OUT"

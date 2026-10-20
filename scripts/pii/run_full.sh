#!/usr/bin/env bash
# The full PII benchmark on a GPU machine: build the datasets once, then JOBS_PER_GPU workers per visible GPU, each
# taking pipelines round-robin and running them over every dataset (python scripts/pii/run_full.py). Then the
# report and the viewer files.
#
#   scripts/pii/run_full.sh                        # everything into ./pii_full
#   OUT=pii_full JOBS_PER_GPU=2 scripts/pii/run_full.sh
#   PIPELINES="pii-tracer openai-privacy-filter" PII_FULL_DATASETS="abcd spokenwoz" scripts/pii/run_full.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PII_ENV="$(cd "${PII_ENV:-.venv-pii}" && pwd)"
OUT="$(mkdir -p "${OUT:-pii_full}" && cd "${OUT:-pii_full}" && pwd)"
JOBS_PER_GPU="${JOBS_PER_GPU:-2}"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS="$("$PII_ENV/bin/python" -c 'import torch; print(max(1, torch.cuda.device_count()))')"
WORKERS=$((GPUS * JOBS_PER_GPU))
PIPELINE_ARGS=()
[ -n "${PIPELINES:-}" ] && PIPELINE_ARGS=(--pipelines $PIPELINES)
[ -n "${BATCH_SIZE:-}" ] && PIPELINE_ARGS+=(--batch-size "$BATCH_SIZE")

echo "building datasets"
"$PII_ENV/bin/python" scripts/pii/run_full.py --build-only | tee "$OUT/datasets.txt"

echo "checking that batched and single outputs agree"
"$PII_ENV/bin/python" scripts/pii/check_batching.py --dataset abcd-mini --pipelines \
  openai-privacy-filter-owner-context-2 pii-tracer-owner-context-2 gliner2-pii-fastino gliner-pii-knowledgator-edge \
  | tee "$OUT/batching_check.txt"

echo "$WORKERS workers on $GPUS GPU(s) -> $OUT"
for worker in $(seq 0 $((WORKERS - 1))); do
  CUDA_VISIBLE_DEVICES=$((worker % GPUS)) "$PII_ENV/bin/python" scripts/pii/run_full.py --out "$OUT" \
    --worker "$worker" --num-workers "$WORKERS" "${PIPELINE_ARGS[@]}" > "$OUT/worker_$worker.log" 2>&1 &
done
wait
grep -h "failed" "$OUT"/worker_*.log || true
"$PII_ENV/bin/python" scripts/pii/report_full.py "$OUT"

#!/usr/bin/env bash
# Create the text-only PII environment at $PII_ENV (default ./.venv-pii) and install this checkout into it.
set -euo pipefail
cd "$(dirname "$0")/../.."
PII_ENV="${PII_ENV:-.venv-pii}"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 "$PII_ENV"
# TORCH_INDEX picks the PyTorch wheel build, e.g. https://download.pytorch.org/whl/cu126 for a 550-series driver
[ -n "${TORCH_INDEX:-}" ] && VIRTUAL_ENV="$PII_ENV" uv pip install "torch==2.8.*" --index-url "$TORCH_INDEX"
VIRTUAL_ENV="$PII_ENV" uv pip install -r scripts/pii/requirements.txt
VIRTUAL_ENV="$PII_ENV" uv pip install --no-deps -e .
"$PII_ENV/bin/python" -W ignore -c "import torch, transformers; from transformers import OpenAIPrivacyFilterForTokenClassification; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPUs; transformers', transformers.__version__)"
"$PII_ENV/bin/python" -m pytest -q -W ignore tests/dataset/test_pii_benchmarks.py tests/pipeline/test_pii_pipelines.py tests/pipeline/test_safety_pipelines.py tests/dataset/test_safety_dataset.py tests/metric/safety

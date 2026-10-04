#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-1.5}"
if (( $# )); then shift; fi
case "$VERSION" in
  1|1.5|1.6|1.7) ;;
  *) echo "usage: sample_efficiency.sh 1|1.5|1.6|1.7 [training arguments]" >&2; exit 2 ;;
esac
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
MODEL_NAME="GR00T-N${VERSION}-3B"
if [[ "$VERSION" == "1" ]]; then MODEL_NAME="GR00T-N1-2B"; fi
MODEL_PATH="${BASE_MODEL_PATH:-nvidia/${MODEL_NAME}}"
if [[ -z "${BASE_MODEL_PATH:-}" && -d "/workspace/models/${MODEL_NAME}" ]]; then
  MODEL_PATH="/workspace/models/${MODEL_NAME}"
fi
export TOKENIZERS_PARALLELISM=false
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$PROJECT_ROOT/.venv/bin/python" -m gr00t.experiment.sample_efficiency \
  --base-model-path "$MODEL_PATH" --model-version "$VERSION" \
  --dataset-root "${DATASET_ROOT:-/root/libero_spatial_post}" \
  --eval-python "${EVAL_PYTHON:-$PROJECT_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python}" \
  --output-dir "${OUTPUT_DIR:-/workspace/minimal-groot-outputs/n${VERSION/./}-spatial-subsets-seed42}" \
  "$@"

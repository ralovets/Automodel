#!/usr/bin/env bash
# Launch experiment/base.yaml inside the training image. Override any setting via environment:
#   IMAGE, NPROC, DATA_DIR, OUTPUT_DIR, OMP_NUM_THREADS, WANDB_* (API key, base URL, entity, mode).
# Trailing arguments are recipe overrides, e.g. --model.config.text_config.use_gr false
set -euo pipefail
name=${1:?Usage: run.sh RUN_NAME [recipe overrides...]}
shift
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
IMAGE=${IMAGE:-automodel:flash}
NPROC=${NPROC:-2}
DATA_DIR=${DATA_DIR:-$HOME/storage/datasets/cl32k-python-english-50b-split-v2-hf-uncompressed}
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/workspace/checkpoints/custom1b}
mkdir -p "$OUTPUT_DIR"
docker run --rm --gpus all --ipc=host --network host --name "custom1b-$name" \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e RUN_NAME="$name" -e DATA_DIR=/data -e OUTPUT_DIR=/outputs \
  -e WANDB_API_KEY -e WANDB_BASE_URL -e WANDB_ENTITY -e WANDB_MODE \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONPATH=/opt/Automodel -e PYTHONDONTWRITEBYTECODE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$repo:/opt/Automodel" -v "$DATA_DIR:/data:ro" -v "$OUTPUT_DIR:/outputs" \
  -w /opt/Automodel "$IMAGE" \
  automodel experiment/base.yaml --nproc-per-node "$NPROC" "$@" 2>&1 | tee "$OUTPUT_DIR/$name.log"

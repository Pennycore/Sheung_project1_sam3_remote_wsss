#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/potsdam_sam3_2080ti.json}
LABELS_CSV=${2:-data/potsdam_image_level_labels.csv}
OUTPUT_DIR=${3:-runs/potsdam_sam3_2080ti}

mkdir -p "${OUTPUT_DIR}/logs"

CUDA_VISIBLE_DEVICES=0 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config "${CONFIG}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards 2 \
  --shard-index 0 \
  --skip-existing \
  > "${OUTPUT_DIR}/logs/gpu0.log" 2>&1 &

PID0=$!

CUDA_VISIBLE_DEVICES=1 python -m sam3_remote_wsss.generate_pseudo_labels \
  --config "${CONFIG}" \
  --labels-csv "${LABELS_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-shards 2 \
  --shard-index 1 \
  --skip-existing \
  > "${OUTPUT_DIR}/logs/gpu1.log" 2>&1 &

PID1=$!

wait "${PID0}"
wait "${PID1}"

echo "Done. Outputs: ${OUTPUT_DIR}"


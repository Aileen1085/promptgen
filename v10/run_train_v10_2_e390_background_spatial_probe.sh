#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
V10_2_INIT="${V10_2_INIT:-$ROOT/output/v10_2_e380_fp_ema_e420/20260920_122704/best.pth}"
OUT_DIR="${OUT_DIR:-$ROOT/output/v10_2_e390_background_spatial_probe_20260924}"
TRAIN_ENTRYPOINT="$ROOT/finetune_multisource_sam2_v10_2_background_spatial.py"

exec env \
  PYTHON="$PYTHON" \
  GPU="$GPU" \
  V10_2_INIT="$V10_2_INIT" \
  OUT_DIR="$OUT_DIR" \
  TRAIN_ENTRYPOINT="$TRAIN_ENTRYPOINT" \
  TRAIN_MAX_ROI_FRAMES="${TRAIN_MAX_ROI_FRAMES:-512}" \
  SAM2_FRAME_BATCH_SIZE="${SAM2_FRAME_BATCH_SIZE:-32}" \
  TRAIN_MEMORY_GROUP_SIZE="${TRAIN_MEMORY_GROUP_SIZE:-4}" \
  TRAIN_PREFETCH_BATCHES="${TRAIN_PREFETCH_BATCHES:-2}" \
  SAM2_FEATURE_CACHE_GPU_MAX_FRAMES="${SAM2_FEATURE_CACHE_GPU_MAX_FRAMES:-256}" \
  bash "$ROOT/run_train_v10_2_multidataset.sh" \
  --epochs 2 \
  --validate-every 2 \
  --no-validate-before-train \
  --train-cases-per-epoch 64 \
  --dataset-foreground-mode scribble \
  --val-foreground-mode scribble \
  --val-background-mode scribble \
  --prompt-lr 1e-4 \
  --adapter-lr 1e-8 \
  --memory-adapter-lr 1e-8 \
  --freeze-prompt-3d-adapter \
  --freeze-prompt-memory-adapter \
  --v10-2-physical-distance-hidden-dim 16 \
  --v10-2-physical-distance-clip-mm 80 \
  --v10-2-background-hard-negative-weight 0.10 \
  --v10-2-background-hard-negative-ratio 0.05 \
  --v10-2-background-hard-negative-min 256 \
  --v10-2-background-hard-negative-max 65536 \
  --validation-seed 2027 \
  --mask-threshold 0.60 \
  --no-anchor-coarse-crop \
  --no-patch-scaling \
  "$@"

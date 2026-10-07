#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export PYTHON="${PYTHON:-python}"
export GPU="${GPU_IDS:?Set verified GPU pair}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export OUT_DIR="${OUT_DIR:-output/sam2_v9_2_ct13_frozen_hardsampling_e360_20_$(date +%Y%m%d_%H%M%S)}"
export INIT_CHECKPOINT="output/sam2_v9_2_ct13_semantic_e380_400_20260928/20260928_200027/epoch360_dice0.7817.pth"
export FRAME_BATCH="${FRAME_BATCH:-4}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TASK_TMPDIR="${PROJECT_ROOT}/.tmp/v92_frozen_hardsampling"
[[ ! -e "${OUT_DIR}" ]] || { echo 'Output already exists' >&2; exit 2; }
[[ -f "${INIT_CHECKPOINT}" ]] || { echo 'Missing E360' >&2; exit 2; }
exec "${PYTHON}" -u v9_2_frozen_hard_sources.py --launch \
  --epochs 20 --validate-every 5 --train-cases-per-epoch 200 \
  --max-frames 192 --validation-window-frames 192 \
  --freeze-decoder --precision-semantic-align-lr 0 "$@"

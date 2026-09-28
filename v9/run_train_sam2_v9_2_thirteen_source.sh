#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export PYTHON="${PYTHON:-python}"
export GPU="${GPU_IDS:?Set two verified free physical GPU IDs}"
export OUT_DIR="${OUT_DIR:-output/sam2_v9_2_ct13_semantic_e380_400_$(date +%Y%m%d_%H%M%S)}"
export INIT_CHECKPOINT="${INIT_CHECKPOINT:-output/sam2_v9_2_eight_source_semantic_aligned_e320_fb4_400/20260923_221902/best.pth}"
case "${OUT_DIR}" in output/sam2_v9_2_ct13_*) ;; *) echo 'Refusing non-CT13 output path' >&2; exit 2 ;; esac
if [[ -e "${OUT_DIR}" ]]; then echo 'Refusing existing output directory' >&2; exit 2; fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then echo 'Missing v9.2 best checkpoint' >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="${GPU}"
export FRAME_BATCH="${FRAME_BATCH:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SAMPROMPT_RUN_STAMP="${SAMPROMPT_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
export TASK_TMPDIR="${PROJECT_ROOT}/.tmp/sam2_v9_2_ct13"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
# Fixed reference: configs/ct13_v10_2_validation_topcow2_classweight_v2_20260928.json
exec "${PYTHON}" -u "${PROJECT_ROOT}/sam2_v9_2_ct13_entry.py" --launch \
  --max-frames 192 --validation-window-frames 192 --validate-before-train \
  "$@"

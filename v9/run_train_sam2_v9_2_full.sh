#!/usr/bin/env bash
# Opt-in two-GPU SAM2 v9.2 image-encoder + mask-decoder fine-tuning.
# Phase A: ENCODER_SCOPE=late. After fixed-validation improvement, start a
# separate phase-B output with ENCODER_SCOPE=all and phase-A best as INIT_CHECKPOINT.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-python}"
TOTALSEG_ROOT="${TOTALSEG_ROOT:-${PROJECT_ROOT}/../../totalsegmentor}"
GPU_IDS="${GPU_IDS:?Set two verified physical GPU IDs, e.g. GPU_IDS=1,5}"
OUT_DIR="${OUT_DIR:?Set a NEW relative output directory under output/}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:?Set the existing v9.2 best checkpoint}"
ENCODER_SCOPE="${ENCODER_SCOPE:-all}"
MAX_FRAMES="${MAX_FRAMES:-48}"
FRAME_BATCH="${FRAME_BATCH:-1}"
TASK_TMPDIR="${TASK_TMPDIR:-${PROJECT_ROOT}/.tmp/sam2_v9_2_full}"

case "${OUT_DIR}" in
  output/*) ;;
  *) echo "OUT_DIR must be a fresh relative path under output/" >&2; exit 2 ;;
esac
if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to reuse an existing output path: ${OUT_DIR}" >&2
  exit 2
fi
if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "INIT_CHECKPOINT does not exist: ${INIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ "${ENCODER_SCOPE}" != late && "${ENCODER_SCOPE}" != all ]]; then
  echo "ENCODER_SCOPE must be late or all" >&2
  exit 2
fi
IFS=, read -r -a GPU_PARTS <<< "${GPU_IDS}"
if [[ ${#GPU_PARTS[@]} -ne 2 || "${GPU_PARTS[0]}" == "${GPU_PARTS[1]}" ]]; then
  echo "GPU_IDS must contain two distinct physical GPU IDs" >&2
  exit 2
fi

mkdir -p "${TASK_TMPDIR}"
export TMPDIR="${TASK_TMPDIR}" TMP="${TASK_TMPDIR}" TEMP="${TASK_TMPDIR}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export V9_SHARED_CASE_CACHE_DIR="${V9_SHARED_CASE_CACHE_DIR:-${PROJECT_ROOT}/v10/.cache/totalseg_sam2_promptgen_v10_prompt_roi_full_v1}"
export V9_SHARED_CACHE_WRITE_POLICY="${V9_SHARED_CACHE_WRITE_POLICY:-write}"
export SAMPROMPT_RUN_STAMP="${SAMPROMPT_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

exec "${PYTHON}" -u -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${PROJECT_ROOT}/v9_family_shared_cache_entry.py" \
  finetune_multisource_sam2_v9_2_full \
  --gpu "${GPU_IDS}" --device cuda \
  --data-root "${TOTALSEG_ROOT}" --labels-root "${TOTALSEG_ROOT}" \
  --totalseg-meta-csv "${TOTALSEG_ROOT}/meta.csv" \
  --model-cfg "${PROJECT_ROOT}/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml" \
  --checkpoint "${PROJECT_ROOT}/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --label-map "${PROJECT_ROOT}/configs/totalseg_label_map.txt" \
  --class-presence-manifest "${PROJECT_ROOT}/.cache/totalseg_nonempty_classes_v1.json" \
  --prompt-cache-dir "${PROJECT_ROOT}/.cache/prompt_roi" \
  --prompt-slice-cache-dir "${PROJECT_ROOT}/.cache/prompt_planes" \
  --v9-1-sat-base-model "${PROJECT_ROOT}/models/biolord-2023-c" \
  --v9-1-sat-checkpoint "${PROJECT_ROOT}/models/sat-medical-text/text_encoder.pth" \
  --v9-1-sat-modality ct --v9-1-morphology-num-experts 4 \
  --v9-1-morphology-top-k 2 --v9-1-morphology-route-hidden-dim 64 \
  --v9-2-soft-routing-epochs 20 --v9-2-topk-routing-epoch 80 \
  --v9-2-residual-strength-init 0.05 \
  --v9-2-counterfactual-interval 4 --v9-2-counterfactual-weight 0.05 \
  --v9-2-router-loss-weight 0.02 --v9-2-tubular-cldice-weight 0.20 \
  --v9-2-sheet-boundary-weight 0.10 \
  --v9-2-ct-channel-mode adaptive_soft_lung_bone \
  --v9-2-ct-bounds-cache-dir "${PROJECT_ROOT}/.cache/v9_2_ct_hu_bounds" \
  --multidataset-split-json "${PROJECT_ROOT}/configs/v9_2_extended_split_4case_v2.json" \
  --multidataset-validation-json "${PROJECT_ROOT}/configs/v9_2_extended_validation_4case_v2.json" \
  --totalseg-sampling-ratio 0.22 --amos-sampling-ratio 0.165 \
  --magic-sampling-ratio 0.165 --msd-task07-sampling-ratio 0.10 \
  --msd-task10-sampling-ratio 0.075 --msd-task08-sampling-ratio 0.10 \
  --parse2022-sampling-ratio 0.05 --topcow2024-cta-sampling-ratio 0.125 \
  --v9-2-difficulty-sampling-cap 1.25 \
  --v9-2-adaptive-scribble --v9-2-scribble-min-radius 1 \
  --v9-2-scribble-max-radius 20 --v9-2-scribble-width-fraction 0.15 \
  --v9-2-scribble-min-plane-ratio 0.02 \
  --precision-prompt-scope-policy prompt_aligned \
  --v9-2-recalibrate-routing --v9-2-recalibration-policy soft \
  --precision-small-target-threshold 0.01 --precision-small-target-pos-cap 3.0 \
  --precision-tversky-weight 0.10 --precision-tversky-alpha 0.70 --precision-tversky-beta 0.30 \
  --precision-hard-negative-weight 0.20 --precision-hard-negative-ratio 3.0 \
  --precision-hard-negative-min 256 --precision-hard-negative-max 65536 \
  --precision-ring-negative-weight 0.10 --precision-ring-radius-pixels 16 \
  --precision-shared-lr-scale 0.10 --precision-new-path-lr 5e-7 \
  --precision-adapter-lr-scale 0.25 --precision-rare-class-max-cases 5 \
  --prompt-generator-checkpoint "${INIT_CHECKPOINT}" --load-decoder-from-prompt-checkpoint \
  --out-dir "${OUT_DIR}" --epochs 40 --validate-every 10 --train-cases-per-epoch 200 \
  --batch-size 1 --sam2-case-batch-size 1 --max-frames "${MAX_FRAMES}" \
  --sam2-frame-batch-size "${FRAME_BATCH}" \
  --model-input-size 1024 --cache-image-size 512 --prompt-work-size 192 \
  --prompt-lr 1e-6 --adapter-lr 2e-6 --decoder-lr 5e-7 \
  --min-lr 5e-8 --weight-decay 1e-5 --grad-clip-norm 1.0 \
  --seg-pos-weight-max 10 --amp --amp-dtype bfloat16 --no-freeze-decoder \
  --prompt-semantic-dim 128 --precision-semantic-align-lr 0 \
  --boundary-loss-weight 0.4 --boundary-dice-loss-weight 0.2 \
  --prompt-consistency-loss-weight 0.10 --v9-1-cache-write-policy write \
  --validation-complete-prompt-roi --validation-roi-auto-expand \
  --validation-roi-expand-step-ratio 0.50 --validation-roi-max-expand-ratio 1.0 \
  --validation-roi-boundary-band-frames 2 --validation-window-frames 192 \
  --val-clear-cache-between-tasks --val-samples-per-class 0 \
  --v9-train-foreground-modes scribble box \
  --v9-train-background-modes scribble point none \
  --v9-eval-foreground-mode scribble --v9-eval-background-mode scribble \
  --no-v9-eval-random-prompt-modes --no-v9-decoder-feedback \
  --full-ft-encoder-scope "${ENCODER_SCOPE}" \
  --full-ft-encoder-lr 2e-7 --full-ft-encoder-early-lr 5e-8 \
  "$@"

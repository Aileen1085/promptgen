#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/../conda-envs/al/bin/python}"
GPU="${GPU:-2}"
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-GPU-06fff85b-357a-f4ed-8267-4229321acede}"
OUT="v10/output/v10_2_sixteen_source_bestA_frozen_40_20261012"
[[ -f configs/v10_2_sixteen_source_train_20261012.json ]] || { echo 'Native373-case audit not complete'; exit 2; }
[[ ! -e "$OUT" ]] || { echo 'Independent output already exists; do not duplicate training'; exit 2; }
USED="$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits)"
TEMP="$(nvidia-smi -i "$GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits)"
UUID="$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)"
[[ "$UUID" == "$EXPECTED_GPU_UUID" && "$USED" -lt 64 && "$TEMP" -lt 75 ]] || { echo 'Target GPU not idle/healthy or UUID changed'; exit 2; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$PYTHON_BIN" -u v10/v10_2_sixteen_source_train.py --gpu "$GPU" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" --output "$OUT"

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-$ROOT/output/v10_2_ct13_e490_lowpg_lr_control_10_20261004}"
export GPU="${GPU:-1}"
# Same repaired-gradient implementation, fresh E490 weights and 40-epoch LR horizon.
# Only PromptGen/3D/memory adapter base rates change relative to inputgrad control.
exec bash "$ROOT/run_train_v10_2_ct13_e490_full_decoder_40.sh" \
 --epochs 10 --lr-schedule-epochs 40 --validate-every 5 \
 --prompt-lr 2.5e-7 --adapter-lr 5e-7 --memory-adapter-lr 5e-7 "$@"

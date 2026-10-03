#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-$ROOT/output/v10_2_ct13_e490_inputgrad_control_10_20261003}"
export GPU="${GPU:-1}"
# Preserve the original 40-epoch LR trajectory, but stop after epoch 10.
# All other settings and the model-only E490 start come from the audited launcher.
exec bash "$ROOT/run_train_v10_2_ct13_e490_full_decoder_40.sh" \
 --epochs 10 --lr-schedule-epochs 40 --validate-every 5 "$@"

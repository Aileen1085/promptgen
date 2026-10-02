#!/usr/bin/env bash
# Fresh CT13 SAM2 encoder+decoder fine-tune; reuses the audited full launcher.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-python}"
exec "${PYTHON}" "${PROJECT_ROOT}/tools/launch_ct13_full_from_v92.py" "$@"

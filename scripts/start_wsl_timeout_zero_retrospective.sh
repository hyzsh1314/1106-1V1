#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_TIMEOUT_ZERO_ROOT:-${PROJECT_ROOT}/runs/timeout_zero_retrospective}"
UNIT_NAME="${WVR_TIMEOUT_ZERO_UNIT:-wvr-timeout-zero-retrospective}"
mkdir -p "${ROOT_DIR}"
if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
  echo "retrospective already completed"
  exit 0
fi
if systemctl --user is-active --quiet "${UNIT_NAME}.service"; then
  echo "retrospective already running"
  exit 0
fi
systemctl --user reset-failed "${UNIT_NAME}.service" 2>/dev/null || true
systemd-run --user --unit="${UNIT_NAME}" --collect \
  --setenv="WVR_TIMEOUT_ZERO_ROOT=${ROOT_DIR}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/supervise_wsl_timeout_zero_retrospective.sh"
echo "started unit=${UNIT_NAME}.service root_dir=${ROOT_DIR}"

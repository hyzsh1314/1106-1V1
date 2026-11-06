#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_R1_5M_ROOT:-${PROJECT_ROOT}/runs/r1_attack_pbrs_5m_validation}"
UNIT_NAME="${WVR_R1_5M_UNIT:-wvr-r1-5m-validation}"
mkdir -p "${ROOT_DIR}"
if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
  echo "R1 5m validation already completed"
  exit 0
fi
if systemctl --user is-active --quiet "${UNIT_NAME}.service"; then
  echo "R1 5m validation already running unit=${UNIT_NAME}.service"
  exit 0
fi
systemctl --user reset-failed "${UNIT_NAME}.service" 2>/dev/null || true
systemd-run --user --unit="${UNIT_NAME}" --collect \
  --setenv="WVR_R1_5M_ROOT=${ROOT_DIR}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/supervise_wsl_r1_5m_validation.sh"
echo "started unit=${UNIT_NAME}.service root_dir=${ROOT_DIR}"


#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_REWARD_ROOT:-${PROJECT_ROOT}/runs/reward_v4_ablation}"
UNIT_NAME="${WVR_REWARD_UNIT:-wvr-reward-ablation-v4}"
mkdir -p "${ROOT_DIR}"
if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
  echo "reward ablation already completed"
  exit 0
fi
if systemctl --user is-active --quiet "${UNIT_NAME}.service"; then
  echo "reward ablation already running unit=${UNIT_NAME}.service"
  exit 0
fi
systemctl --user reset-failed "${UNIT_NAME}.service" 2>/dev/null || true
systemd-run --user --unit="${UNIT_NAME}" --collect \
  --setenv="WVR_REWARD_ROOT=${ROOT_DIR}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/supervise_wsl_reward_ablation_v4.sh"
echo "started unit=${UNIT_NAME}.service root_dir=${ROOT_DIR}"


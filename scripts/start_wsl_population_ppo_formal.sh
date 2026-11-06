#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${WVR_RUN_DIR:-${PROJECT_ROOT}/runs/ppo_population_1v1_baseline_seed85_100m}"
UNIT_NAME="${WVR_SYSTEMD_UNIT:-wvr-ppo-population-baseline}"
mkdir -p "${RUN_DIR}"

if [[ -f "${RUN_DIR}/COMPLETED.json" ]]; then
  echo "training already completed"
  exit 0
fi
if systemctl --user is-active --quiet "${UNIT_NAME}.service"; then
  echo "training service already running unit=${UNIT_NAME}.service"
  exit 0
fi

systemctl --user reset-failed "${UNIT_NAME}.service" 2>/dev/null || true
systemd-run --user \
  --unit="${UNIT_NAME}" \
  --collect \
  --setenv="WVR_RUN_DIR=${RUN_DIR}" \
  /usr/bin/bash "${PROJECT_ROOT}/scripts/supervise_wsl_population_ppo_formal.sh"
echo "started formal baseline service unit=${UNIT_NAME}.service run_dir=${RUN_DIR}"

#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_TIMEOUT_ZERO_1M_ROOT:-${PROJECT_ROOT}/runs/timeout_zero_1m_paired}"
mkdir -p "${ROOT_DIR}"
exec 9>"${ROOT_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "another timeout-zero 1M supervisor holds the lock"
  exit 0
fi
attempt=0
while [[ ! -f "${ROOT_DIR}/COMPLETED.json" ]]; do
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] attempt=${attempt}" \
    | tee -a "${ROOT_DIR}/supervisor.log"
  bash "${PROJECT_ROOT}/scripts/run_wsl_timeout_zero_1m.sh" \
    2>&1 | tee -a "${ROOT_DIR}/run.log"
  status=${PIPESTATUS[0]}
  if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] exited=${status}; retry in 60s" \
    | tee -a "${ROOT_DIR}/supervisor.log"
  sleep 60
done

#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_ANCHOR_1M_ROOT:-${PROJECT_ROOT}/runs/anchor_curriculum_1m_paired}"
OUTPUT_DIR="${ROOT_DIR}/checkpoint-selection"
mkdir -p "${OUTPUT_DIR}"
exec 9>"${OUTPUT_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "another checkpoint-selection supervisor holds the lock"
  exit 0
fi
attempt=0
while [[ ! -f "${OUTPUT_DIR}/COMPLETED.json" ]]; do
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] attempt=${attempt}" \
    | tee -a "${OUTPUT_DIR}/supervisor.log"
  bash "${PROJECT_ROOT}/scripts/run_wsl_anchor_checkpoint_selection.sh" \
    2>&1 | tee -a "${OUTPUT_DIR}/run.log"
  status=${PIPESTATUS[0]}
  if [[ -f "${OUTPUT_DIR}/COMPLETED.json" ]]; then
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] exited=${status}; retry in 30s" \
    | tee -a "${OUTPUT_DIR}/supervisor.log"
  sleep 30
done

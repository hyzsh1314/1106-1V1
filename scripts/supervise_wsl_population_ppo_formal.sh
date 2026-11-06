#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${WVR_RUN_DIR:-${PROJECT_ROOT}/runs/ppo_population_1v1_baseline_seed85_100m}"
mkdir -p "${RUN_DIR}"

exec 9>"${RUN_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "another formal-training supervisor already holds the lock"
  exit 0
fi

echo "$$" > "${RUN_DIR}/supervisor.pid"
cleanup() {
  rm -f "${RUN_DIR}/supervisor.pid"
}
trap cleanup EXIT INT TERM

attempt=0
while [[ ! -f "${RUN_DIR}/COMPLETED.json" ]]; do
  attempt=$((attempt + 1))
  timestamp="$(date --iso-8601=seconds)"
  echo "[${timestamp}] supervisor attempt=${attempt}" | tee -a "${RUN_DIR}/supervisor.log"
  bash "${PROJECT_ROOT}/scripts/run_wsl_population_ppo_formal.sh" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
  status=${PIPESTATUS[0]}
  if [[ -f "${RUN_DIR}/COMPLETED.json" ]]; then
    echo "[$(date --iso-8601=seconds)] training completed" | tee -a "${RUN_DIR}/supervisor.log"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] training exited status=${status}; restarting from latest checkpoint in 60 seconds" \
    | tee -a "${RUN_DIR}/supervisor.log"
  sleep 60
done

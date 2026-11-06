#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_REWARD_ROOT:-${PROJECT_ROOT}/runs/reward_v4_ablation}"
mkdir -p "${ROOT_DIR}"
exec 9>"${ROOT_DIR}/supervisor.lock"
if ! flock -n 9; then
  echo "another reward-ablation supervisor already holds the lock"
  exit 0
fi
echo "$$" > "${ROOT_DIR}/supervisor.pid"
trap 'rm -f "${ROOT_DIR}/supervisor.pid"' EXIT INT TERM

attempt=0
while [[ ! -f "${ROOT_DIR}/COMPLETED.json" ]]; do
  attempt=$((attempt + 1))
  echo "[$(date --iso-8601=seconds)] supervisor attempt=${attempt}" \
    | tee -a "${ROOT_DIR}/supervisor.log"
  bash "${PROJECT_ROOT}/scripts/run_wsl_reward_ablation_v4.sh" \
    2>&1 | tee -a "${ROOT_DIR}/suite.log"
  status=${PIPESTATUS[0]}
  if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
    echo "[$(date --iso-8601=seconds)] reward ablation completed" \
      | tee -a "${ROOT_DIR}/supervisor.log"
    exit 0
  fi
  echo "[$(date --iso-8601=seconds)] suite exited status=${status}; retrying in 60 seconds" \
    | tee -a "${ROOT_DIR}/supervisor.log"
  sleep 60
done


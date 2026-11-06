#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${WVR_RUN_DIR:-${PROJECT_ROOT}/runs/ppo_population_1v1_baseline_seed85_100m}"
UNIT_NAME="${WVR_SYSTEMD_UNIT:-wvr-ppo-population-baseline}"
echo "run_dir=${RUN_DIR}"
if systemctl --user is-active --quiet "${UNIT_NAME}.service"; then
  echo "service=RUNNING unit=${UNIT_NAME}.service"
else
  echo "service=STOPPED unit=${UNIT_NAME}.service"
fi
if [[ -f "${RUN_DIR}/supervisor.pid" ]]; then
  supervisor_pid="$(cat "${RUN_DIR}/supervisor.pid")"
  if kill -0 "${supervisor_pid}" 2>/dev/null; then
    echo "supervisor=RUNNING pid=${supervisor_pid}"
  else
    echo "supervisor=STOPPED stale_pid=${supervisor_pid}"
  fi
else
  echo "supervisor=MISSING"
fi
if [[ -f "${RUN_DIR}/progress.json" ]]; then
  /home/hongyun0405/.venvs/bvr1022/bin/python -m json.tool \
    "${RUN_DIR}/progress.json"
else
  echo "progress=WAITING_FOR_FIRST_UPDATE"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader
fi

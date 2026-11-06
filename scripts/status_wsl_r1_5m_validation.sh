#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_R1_5M_ROOT:-${PROJECT_ROOT}/runs/r1_attack_pbrs_5m_validation}"
UNIT_NAME="${WVR_R1_5M_UNIT:-wvr-r1-5m-validation}"
PYTHON_BIN="${WVR_VENV:-${HOME}/.venvs/bvr1022}/bin/python"
echo "root_dir=${ROOT_DIR}"
systemctl --user is-active "${UNIT_NAME}.service" || true
for seed in 85 185 285; do
  run_dir="${ROOT_DIR}/seed${seed}_5m"
  if [[ -f "${run_dir}/COMPLETED.json" ]]; then
    state=TRAINED
  elif [[ -f "${run_dir}/progress.json" ]]; then
    state=RUNNING
  else
    state=WAITING
  fi
  printf 'seed=%-3s state=%-7s' "${seed}" "${state}"
  if [[ -f "${run_dir}/progress.json" ]]; then
    "${PYTHON_BIN}" - "${run_dir}/progress.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(" update={}/{} steps={} percent={:.2f} throughput={} eta_s={}".format(
    p.get("update"), p.get("target_updates"), p.get("total_steps"),
    float(p.get("percent", 0)), p.get("session_throughput_steps_s"),
    p.get("eta_s", "n/a")))
PY
  else
    echo
  fi
done
if [[ -f "${ROOT_DIR}/COMPLETED.json" ]]; then
  cat "${ROOT_DIR}/COMPLETED.json"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
fi


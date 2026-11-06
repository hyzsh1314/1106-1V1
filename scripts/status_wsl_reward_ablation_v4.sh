#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_REWARD_ROOT:-${PROJECT_ROOT}/runs/reward_v4_ablation}"
UNIT_NAME="${WVR_REWARD_UNIT:-wvr-reward-ablation-v4}"
if [[ -x "${WVR_VENV:-${HOME}/.venvs/bvr1022}/bin/python" ]]; then
  PYTHON_BIN="${WVR_VENV:-${HOME}/.venvs/bvr1022}/bin/python"
else
  PYTHON_BIN=python3
fi
echo "root_dir=${ROOT_DIR}"
systemctl --user is-active "${UNIT_NAME}.service" || true
for variant in R0_terminal R1_attack_pbrs R2_threat_pbrs R3_constraints; do
  for seed in 85 185 285; do
    run_dir="${ROOT_DIR}/${variant}_seed${seed}_1m"
    if [[ -f "${run_dir}/COMPLETED.json" ]]; then
      state=TRAINED
    elif [[ -f "${run_dir}/progress.json" ]]; then
      state=RUNNING
    else
      state=WAITING
    fi
    printf '%-25s seed=%-3s state=%s' "${variant}" "${seed}" "${state}"
    if [[ -f "${run_dir}/progress.json" ]]; then
      "${PYTHON_BIN}" - "${run_dir}/progress.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
print(" update={}/{} steps={} reward={}".format(
    p.get("update"), p.get("target_updates"), p.get("total_steps"),
    p.get("reward", "n/a")))
PY
    else
      echo
    fi
  done
done
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
fi


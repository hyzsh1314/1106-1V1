#!/usr/bin/env bash
set -u

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_name="${WVR_V12_SYSTEMD_UNIT:-wvr-drvoi-v12-reachability}"
data_dir="${WVR_V12_DATA_DIR:-${project_root}/runs/drvoi_rhppo_v12/reachability-a1-seed85}"
model_dir="${WVR_V12_MODEL_DIR:-${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85}"
b0_dir="${WVR_V12_B0_DIR:-${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85}"

systemctl --user --no-pager --full status "${unit_name}.service" 2>/dev/null || true
for path in \
  "${data_dir}/progress.json" \
  "${data_dir}/COMPLETED.json" \
  "${model_dir}/progress.json" \
  "${model_dir}/COMPLETED.json" \
  "${b0_dir}/progress.json" \
  "${b0_dir}/COMPLETED.json"; do
  if [[ -f "${path}" ]]; then
    echo "===== ${path} ====="
    cat "${path}"
  fi
done
echo "===== supervisor tail ====="
tail -n 20 "${data_dir}/supervisor.log" 2>/dev/null || true
echo "===== collector tail ====="
tail -n 20 "${data_dir}/collector.log" 2>/dev/null || true
echo "===== gpu ====="
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv,noheader 2>/dev/null || true

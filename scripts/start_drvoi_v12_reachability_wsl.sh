#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unit_name="${WVR_V12_SYSTEMD_UNIT:-wvr-drvoi-v12-reachability}"
data_dir="${WVR_V12_DATA_DIR:-${project_root}/runs/drvoi_rhppo_v12/reachability-a1-seed85}"
mkdir -p "${data_dir}"

if [[ -f "${data_dir}/COMPLETED.json" ]] && \
   [[ -f "${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85/COMPLETED.json" ]] && \
   [[ -f "${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85/COMPLETED.json" ]] && \
   [[ -s "${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85/eval-hold.json" ]] && \
   [[ -s "${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85/eval-pursuit.json" ]] && \
   [[ -s "${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85/eval-evasive.json" ]]; then
  echo "V12 reachability A1/A2 and MRAT B0 already completed"
  exit 0
fi
if systemctl --user is-active --quiet "${unit_name}.service"; then
  echo "service already active unit=${unit_name}.service"
  exit 0
fi
systemctl --user reset-failed "${unit_name}.service" 2>/dev/null || true
systemd-run --user \
  --unit="${unit_name}" \
  --collect \
  /usr/bin/bash "${project_root}/scripts/supervise_drvoi_v12_reachability_wsl.sh"
echo "started unit=${unit_name}.service data_dir=${data_dir}"


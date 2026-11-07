#!/usr/bin/env bash
set -u -o pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_dir="${WVR_V12_B8_DIR:-${project_root}/runs/drvoi_rhppo_v12/b8-sprt-latched-diagnostic-seed85}"
mkdir -p "${run_dir}"
supervisor_log="${run_dir}/supervisor.log"
exec 9>"${run_dir}/supervisor.lock"
if ! flock -n 9; then
  printf '%s supervisor already holds lock\n' "$(date --iso-8601=seconds)" >> "${supervisor_log}"
  exit 0
fi
max_restarts=5
attempt=0
while (( attempt <= max_restarts )); do
  printf '%s start attempt=%d\n' "$(date --iso-8601=seconds)" "${attempt}" >> "${supervisor_log}"
  /usr/bin/bash "${project_root}/scripts/run_drvoi_v12_b8_sprt_latched_diagnostic_wsl.sh"
  status=$?
  if (( status == 0 )); then
    printf '%s completed\n' "$(date --iso-8601=seconds)" >> "${supervisor_log}"
    exit 0
  fi
  attempt=$((attempt + 1))
  printf '%s failed status=%d next_attempt=%d\n' "$(date --iso-8601=seconds)" "${status}" "${attempt}" >> "${supervisor_log}"
  if (( attempt > max_restarts )); then exit "${status}"; fi
  sleep 5
done

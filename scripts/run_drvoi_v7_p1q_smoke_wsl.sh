#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
root="runs/drvoi_rhppo_v7/p1-response-smoke"

set +e
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_q_v7.py \
    --onpolicy "$root/data/onpolicy-seed6185.pt" \
    --delta 3 --epochs 2 --batch-size 512 --seed 10185 \
    --output "$root/q-validation.json" > "$root/q-validation.log" 2>&1
status=$?
set -e
if [[ $status -ne 0 && $status -ne 2 ]]; then
    exit "$status"
fi
test -s "$root/q-validation.json"
test -s "$root/q-validation.pt"
touch "$root/Q_SMOKE_COMPLETED"

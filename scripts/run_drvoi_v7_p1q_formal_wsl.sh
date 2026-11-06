#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
root="runs/drvoi_rhppo_v7/p1-response-formal"

test -f "$root/COMPLETED"
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_q_v7.py \
    --onpolicy "$root/data/onpolicy-seed6185.pt" \
               "$root/data/onpolicy-seed7185.pt" \
               "$root/data/onpolicy-seed8185.pt" \
    --delta 3 --epochs 30 --batch-size 1024 --seed 10185 \
    --output "$root/q-validation.json" > "$root/q-validation.log" 2>&1

touch "$root/Q_COMPLETED"

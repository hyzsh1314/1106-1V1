#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo/p1-wvr-response"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_response_p1.py \
    --rhppo-checkpoint runs/drvoi_rhppo/p0-rhppo-smoke-batched/latest.pt \
    --num-envs 96 --collection-steps 384 --epochs 20 --batch-size 2048 \
    --seed 6185 --output "$out/validation.json" \
    > "$out/validation.log" 2>&1

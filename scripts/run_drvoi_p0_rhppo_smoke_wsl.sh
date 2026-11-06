#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo/p0-rhppo-smoke-batched"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" -m benchmarks.build_drvoi_p0_manifest \
    --output "$out/manifest.json" > "$out/manifest.log" 2>&1

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_rhppo.py \
    --num-envs 96 --rollout-steps 256 --updates 3 \
    --epochs 2 --sequence-length 64 --burn-in 20 \
    --chunks-per-minibatch 16 --seed 85 \
    --output "$out/train.json" --checkpoint "$out/latest.pt" \
    > "$out/train.log" 2>&1

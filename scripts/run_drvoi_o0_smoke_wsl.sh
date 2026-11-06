#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo/p2-o0-voi-only-smoke"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_drvoi_rhppo.py \
    --rhppo-checkpoint runs/drvoi_rhppo/p0-rhppo-smoke-batched/latest.pt \
    --response-checkpoint runs/drvoi_rhppo/p1-wvr-response/validation.pt \
    --num-envs 96 --rollout-steps 64 --updates 5 --ppo-epochs 1 \
    --sequence-length 64 --burn-in 20 --chunks-per-minibatch 16 \
    --q-warmup-updates 2 --q-gradient-updates 4 --q-batch-size 1024 \
    --intrinsic-beta 0.10 --seed 85 \
    --output "$out/train.json" --checkpoint "$out/latest.pt" \
    > "$out/train.log" 2>&1

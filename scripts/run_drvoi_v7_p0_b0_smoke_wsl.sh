#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo_v7/p0-b0-smoke/seed85"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" -m benchmarks.build_drvoi_v7_manifest \
    --output "runs/drvoi_rhppo_v7/manifest.json" \
    > "runs/drvoi_rhppo_v7/manifest.log" 2>&1

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_b0_rhppo_v7.py \
    --num-envs 96 --rollout-steps 256 --updates 3 \
    --epochs 2 --sequence-length 64 --burn-in 20 \
    --chunks-per-minibatch 16 --seed 85 --output-dir "$out" \
    > "$out/train.log" 2>&1

for opponent in hold pursuit evasive; do
    PYTHONPATH=. "$python_bin" benchmarks/evaluate_torch_wvr_b0_rhppo_v7.py \
        --checkpoint "$out/latest.pt" --opponent "$opponent" \
        --episodes 12 --num-envs 96 --seed 10085 \
        --output "$out/eval-$opponent.json" \
        > "$out/eval-$opponent.log" 2>&1
done

touch "$out/COMPLETED"

#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
root="runs/drvoi_rhppo_v7/b0-1m"
mkdir -p "$root"

for seed in 85 185 285; do
    out="$root/seed$seed"
    mkdir -p "$out"
    PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_b0_rhppo_v7.py \
        --num-envs 96 --rollout-steps 256 --updates 41 \
        --epochs 2 --sequence-length 64 --burn-in 20 \
        --chunks-per-minibatch 16 --seed "$seed" --output-dir "$out" \
        --resume > "$out/train.log" 2>&1
    for opponent in hold pursuit evasive; do
        PYTHONPATH=. "$python_bin" benchmarks/evaluate_torch_wvr_b0_rhppo_v7.py \
            --checkpoint "$out/latest.pt" --opponent "$opponent" \
            --episodes 192 --num-envs 96 --seed 10085 \
            --output "$out/eval-$opponent.json" \
            > "$out/eval-$opponent.log" 2>&1
    done
    touch "$out/COMPLETED"
done

touch "$root/COMPLETED"

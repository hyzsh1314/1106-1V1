#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
seed="${1:-85}"
out="runs/torch6dof_validation/v4-ppo-fixed-mix-1m/seed${seed}"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_ppo.py \
    --num-envs 96 --rollout-steps 256 --updates 41 \
    --epochs 4 --minibatch-size 2048 --curriculum formal \
    --training-opponents fixed-mix --reward-variant competitive \
    --seed "$seed" --output "$out/train.json" \
    --checkpoint "$out/latest.pt" > "$out/train.log" 2>&1

for opponent in hold pursuit evasive; do
    PYTHONPATH=. "$python_bin" \
        benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --checkpoint "$out/latest.pt" --opponent "$opponent" \
        --episodes 192 --num-envs 96 --seed 10085 \
        --output "$out/eval-$opponent.json" \
        > "$out/eval-$opponent.log" 2>&1
done

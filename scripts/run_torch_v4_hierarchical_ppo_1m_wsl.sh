#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
seed="${1:-85}"
base="runs/torch6dof_validation/v4-reward-ablation-3seed-1m/competitive/seed${seed}/latest.pt"
intercept="runs/torch6dof_validation/v4-bc-los-intercept-3seed/seed${seed}/bc/latest.pt"
out="runs/torch6dof_validation/v4-hierarchical-ppo-1m/seed${seed}"

test -f "$base"
test -f "$intercept"
mkdir -p "$out"

# 96 * 256 * 41 = 1,007,616 environment decisions.  A five-decision
# option horizon corresponds to one simulated second at the 5 Hz policy rate.
PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_hierarchical_ppo.py \
    --base-checkpoint "$base" \
    --intercept-checkpoint "$intercept" \
    --num-envs 96 --rollout-steps 256 --updates 41 \
    --epochs 4 --minibatch-size 2048 --option-horizon 5 \
    --training-opponents staged --seed "$seed" \
    --output "$out/train.json" --checkpoint "$out/latest.pt" \
    > "$out/train.log" 2>&1

for opponent in hold pursuit evasive; do
    PYTHONPATH=. "$python_bin" \
        benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --checkpoint "$out/latest.pt" --opponent "$opponent" \
        --episodes 192 --num-envs 96 --seed 10085 \
        --output "$out/eval-$opponent.json" \
        > "$out/eval-$opponent.log" 2>&1
done

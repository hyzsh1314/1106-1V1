#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
bc_dir="runs/torch6dof_validation/v4-bc-los-intercept-3seed/seed85/bc"
out="runs/torch6dof_validation/v4-bc-los-intercept-3seed/seed85/ppo-aux-0p5m"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_ppo.py \
    --num-envs 96 --rollout-steps 256 --updates 20 \
    --curriculum formal --training-opponents fixed-mix \
    --reward-variant competitive \
    --teacher-aux-coef 0.02 --teacher-aux-final-coef 0.005 \
    --seed 85 --resume "$bc_dir/latest.pt" \
    --output "$out/train.json" --checkpoint "$out/latest.pt" \
    > "$out/train.log" 2>&1

for opponent in hold pursuit evasive; do
    PYTHONPATH=. "$python_bin" benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --checkpoint "$out/latest.pt" --opponent "$opponent" \
        --episodes 192 --num-envs 96 --seed 10085 \
        --output "$out/eval-$opponent.json" \
        > "$out/eval-$opponent.log" 2>&1
done

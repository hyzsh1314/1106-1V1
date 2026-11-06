#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
seed="${1:-85}"
anchor_coef="${2:-0.20}"
base="runs/torch6dof_validation/v4-reward-ablation-3seed-1m/competitive/seed85/latest.pt"
intercept="runs/torch6dof_validation/v4-bc-los-intercept-3seed/seed85/bc/latest.pt"
coef_tag="${anchor_coef/./p}"
out="runs/torch6dof_validation/v4-hierarchical-ppo-anchor-smoke-kl${coef_tag}/seed${seed}"
mkdir -p "$out"

PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_hierarchical_ppo.py \
    --base-checkpoint "$base" --intercept-checkpoint "$intercept" \
    --num-envs 96 --rollout-steps 256 --updates 21 \
    --epochs 4 --minibatch-size 2048 --option-horizon 5 \
    --training-opponents fixed-mix --seed "$seed" \
    --gate-entropy-coef 0.01 --gate-entropy-final-coef 0.003 \
    --option-balance-coef 0.01 --option-balance-final-coef 0.001 \
    --gate-teacher-coef 1.00 --gate-teacher-final-coef 0.00 \
    --gate-teacher-cutoff-fraction 0.50 --gate-anchor-kl-coef "$anchor_coef" \
    --fire-controller option-aware \
    --output "$out/train.json" --checkpoint "$out/latest.pt" \
    > "$out/train.log" 2>&1

for opponent in hold pursuit evasive; do
    PYTHONPATH=. "$python_bin" \
        benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --checkpoint "$out/latest.pt" --opponent "$opponent" \
        --hierarchical-gate-mode learned \
        --episodes 96 --num-envs 96 --seed 10085 \
        --output "$out/eval-$opponent.json" \
        > "$out/eval-$opponent.log" 2>&1
done

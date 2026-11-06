#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
checkpoint="${1:-runs/torch6dof_validation/hierarchical-ppo-v2-smoke/latest.pt}"
out="${2:-runs/torch6dof_validation/hierarchical-ppo-v2-static-diagnostic}"
gates="${3:-always-base always-intercept oracle geometry}"
episodes="${4:-96}"
mkdir -p "$out"

for gate in $gates; do
    for opponent in hold pursuit evasive; do
        PYTHONPATH=. "$python_bin" \
            benchmarks/evaluate_torch_wvr_fixed_opponents.py \
            --checkpoint "$checkpoint" --opponent "$opponent" \
            --hierarchical-gate-mode "$gate" \
            --episodes "$episodes" --num-envs 96 --seed 10085 \
            --output "$out/$gate-$opponent.json" \
            > "$out/$gate-$opponent.log" 2>&1
    done
done

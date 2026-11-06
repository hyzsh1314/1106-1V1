#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/torch6dof_validation/v4-scripted-reachability"
mkdir -p "$out"
for opponent in hold pursuit; do
    PYTHONPATH=. "$python_bin" benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --scripted-learner los-intercept --opponent "$opponent" \
        --episodes 192 --num-envs 96 --seed 10085 \
        --output "$out/los-intercept-vs-$opponent.json" \
        > "$out/los-intercept-vs-$opponent.log" 2>&1
done

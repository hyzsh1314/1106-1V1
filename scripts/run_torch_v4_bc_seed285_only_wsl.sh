#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/torch6dof_validation/v4-bc-los-intercept-3seed/seed285/bc"
mkdir -p "$out"

if [[ ! -f "$out/COMPLETED" ]]; then
    PYTHONPATH=. "$python_bin" benchmarks/pretrain_torch_wvr_bc.py \
        --num-envs 96 --collection-steps 256 --iterations 8 --epochs 4 \
        --minibatch-size 4096 --teacher-opponents evasive --seed 285 \
        --output "$out/train.json" --checkpoint "$out/latest.pt" \
        > "$out/train.log" 2>&1
    touch "$out/COMPLETED"
fi

for opponent in hold pursuit evasive; do
    if [[ ! -f "$out/eval-$opponent.json" ]]; then
        PYTHONPATH=. "$python_bin" benchmarks/evaluate_torch_wvr_fixed_opponents.py \
            --checkpoint "$out/latest.pt" --opponent "$opponent" \
            --episodes 192 --num-envs 96 --seed 10085 \
            --output "$out/eval-$opponent.json" \
            > "$out/eval-$opponent.log" 2>&1
    fi
done

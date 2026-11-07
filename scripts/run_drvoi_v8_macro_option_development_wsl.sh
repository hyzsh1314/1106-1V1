#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
b0="runs/drvoi_rhppo_v7/b0-1m/seed85/latest.pt"
config="configs/drvoi_rhppo_v8_macro_option.json"
out="runs/drvoi_rhppo_v8/macro-option/development"
mkdir -p "$out/data"

for seed in 73185 74185; do
    target="$out/data/train-seed${seed}.pt"
    if [[ ! -s "$target" ]]; then
        PYTHONPATH=. "$python_bin" \
            benchmarks/collect_drvoi_v8_macro_option_data.py \
            --b0-checkpoint "$b0" --split train --seed "$seed" \
            --config "$config" --output "$target" \
            > "${target%.pt}.log" 2>&1
    fi
done

validation="$out/data/validation-seed75185.pt"
if [[ ! -s "$validation" ]]; then
    PYTHONPATH=. "$python_bin" \
        benchmarks/collect_drvoi_v8_macro_option_data.py \
        --b0-checkpoint "$b0" --split validation --seed 75185 \
        --config "$config" --output "$validation" \
        > "${validation%.pt}.log" 2>&1
fi

PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_v8_macro_option_value.py \
    --train "$out/data/train-seed73185.pt" \
            "$out/data/train-seed74185.pt" \
    --validation "$validation" --config "$config" --seed 77185 \
    --output "$out/development.json" | tee "$out/development.log"

touch "$out/COMPLETED"

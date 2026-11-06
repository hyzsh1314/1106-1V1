#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo_v7_1/w4-b0-robustness"
authorization="results/drvoi_rhppo_v7_1/w3/confirmation_audit.json"
mkdir -p "$out/data"
rm -f "$out/COMPLETED" "$out/FAILED"

collection_seeds=(10185 11185 12185 13185 14185)
for b0_seed in 185 285; do
    for collection_seed in "${collection_seeds[@]}"; do
        target="$out/data/onpolicy-b0seed$b0_seed-collection$collection_seed.pt"
        if [[ ! -s "$target" ]]; then
            PYTHONPATH=. "$python_bin" benchmarks/collect_drvoi_v7_1_b0_robustness_data.py \
                --authorization "$authorization" \
                --b0-checkpoint "runs/drvoi_rhppo_v7/b0-1m/seed$b0_seed/latest.pt" \
                --expected-b0-seed "$b0_seed" --dataset onpolicy \
                --seed "$collection_seed" --num-envs 96 \
                --train-episodes 120 --validation-episodes 30 \
                --output "$target" > "${target%.pt}.log" 2>&1
        fi
    done
done

set +e
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_v7_1_b0_robustness.py \
    --b0-seed185 \
        "$out/data/onpolicy-b0seed185-collection10185.pt" \
        "$out/data/onpolicy-b0seed185-collection11185.pt" \
        "$out/data/onpolicy-b0seed185-collection12185.pt" \
        "$out/data/onpolicy-b0seed185-collection13185.pt" \
        "$out/data/onpolicy-b0seed185-collection14185.pt" \
    --b0-seed285 \
        "$out/data/onpolicy-b0seed285-collection10185.pt" \
        "$out/data/onpolicy-b0seed285-collection11185.pt" \
        "$out/data/onpolicy-b0seed285-collection12185.pt" \
        "$out/data/onpolicy-b0seed285-collection13185.pt" \
        "$out/data/onpolicy-b0seed285-collection14185.pt" \
    --epochs 30 --patience 6 --batch-size 4096 --hidden-size 256 --members 5 \
    --seed 39185 --output "$out/robustness.json" > "$out/robustness.log" 2>&1
status=$?
set -e
if [[ $status -eq 0 ]]; then
    touch "$out/COMPLETED"
else
    touch "$out/FAILED"
fi
exit "$status"

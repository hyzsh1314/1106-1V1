#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
b0="runs/drvoi_rhppo_v7/b0-1m/seed85/latest.pt"
dev="runs/drvoi_rhppo_v7_1/w2-history-response-development"
out="runs/drvoi_rhppo_v7_1/w3-confirmation"
authorization="results/drvoi_rhppo_v7_1/w2/development_audit.json"
mkdir -p "$out/data"
rm -f "$out/COMPLETED" "$out/FAILED"

seeds=(10185 11185 12185 13185 14185)
for seed in "${seeds[@]}"; do
    for dataset in onpolicy reactive; do
        target="$out/data/$dataset-seed$seed.pt"
        if [[ ! -s "$target" ]]; then
            PYTHONPATH=. "$python_bin" benchmarks/collect_drvoi_v7_1_confirmation_data.py \
                --authorization "$authorization" \
                --b0-checkpoint "$b0" --dataset "$dataset" --seed "$seed" \
                --num-envs 96 --calibration-episodes 30 --test-episodes 30 \
                --output "$target" > "${target%.pt}.log" 2>&1
        fi
    done
done

set +e
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_history_response_v7_1_confirmation.py \
    --development-onpolicy \
        "$dev/data/onpolicy-seed10185.pt" "$dev/data/onpolicy-seed11185.pt" \
        "$dev/data/onpolicy-seed12185.pt" "$dev/data/onpolicy-seed13185.pt" \
        "$dev/data/onpolicy-seed14185.pt" \
    --development-reactive \
        "$dev/data/reactive-seed10185.pt" "$dev/data/reactive-seed11185.pt" \
        "$dev/data/reactive-seed12185.pt" "$dev/data/reactive-seed13185.pt" \
        "$dev/data/reactive-seed14185.pt" \
    --confirmation-onpolicy \
        "$out/data/onpolicy-seed10185.pt" "$out/data/onpolicy-seed11185.pt" \
        "$out/data/onpolicy-seed12185.pt" "$out/data/onpolicy-seed13185.pt" \
        "$out/data/onpolicy-seed14185.pt" \
    --confirmation-reactive \
        "$out/data/reactive-seed10185.pt" "$out/data/reactive-seed11185.pt" \
        "$out/data/reactive-seed12185.pt" "$out/data/reactive-seed13185.pt" \
        "$out/data/reactive-seed14185.pt" \
    --epochs 30 --patience 6 --batch-size 4096 --hidden-size 256 --members 5 \
    --seed 29185 --output "$out/confirmation.json" \
    > "$out/confirmation.log" 2>&1
status=$?
set -e
if [[ $status -eq 0 ]]; then
    touch "$out/COMPLETED"
else
    touch "$out/FAILED"
fi
exit "$status"

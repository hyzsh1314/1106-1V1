#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
b0="runs/drvoi_rhppo_v7/b0-1m/seed85/latest.pt"
out="runs/drvoi_rhppo_v7_1/w1-history-response-smoke"
mkdir -p "$out/data"

for dataset in onpolicy reactive; do
    PYTHONPATH=. "$python_bin" benchmarks/collect_drvoi_v7_1_development_data.py \
        --b0-checkpoint "$b0" --dataset "$dataset" --seed 10185 \
        --num-envs 96 --train-episodes 6 --validation-episodes 4 \
        --output "$out/data/$dataset-seed10185.pt" \
        > "$out/data/$dataset-seed10185.log" 2>&1
done

set +e
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_history_response_v7_1_dev.py \
    --onpolicy "$out/data/onpolicy-seed10185.pt" \
    --reactive "$out/data/reactive-seed10185.pt" \
    --epochs 15 --patience 4 --batch-size 2048 --hidden-size 128 --members 3 \
    --seed 19185 --output "$out/validation.json" \
    > "$out/validation.log" 2>&1
status=$?
set -e
if [[ $status -ne 0 && $status -ne 2 ]]; then
    exit "$status"
fi
test -s "$out/validation.json"
test -s "$out/validation.pt"
touch "$out/COMPLETED"

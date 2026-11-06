#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
b0="runs/drvoi_rhppo_v7/b0-1m/seed85/latest.pt"
out="runs/drvoi_rhppo_v7/p1-response-smoke"
mkdir -p "$out/data"

PYTHONPATH=. "$python_bin" -m benchmarks.build_drvoi_v7_manifest \
    --output "$out/manifest.json" > "$out/manifest.log" 2>&1

for dataset in cover onpolicy reactive; do
    PYTHONPATH=. "$python_bin" benchmarks/collect_drvoi_v7_response_data.py \
        --b0-checkpoint "$b0" --dataset "$dataset" --seed 6185 \
        --num-envs 96 --train-episodes 6 --validation-episodes 3 \
        --calibration-episodes 3 --test-episodes 3 \
        --output "$out/data/$dataset-seed6185.pt" \
        > "$out/data/$dataset-seed6185.log" 2>&1
done

set +e
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_response_v7.py \
    --onpolicy "$out/data/onpolicy-seed6185.pt" \
    --cover "$out/data/cover-seed6185.pt" \
    --reactive "$out/data/reactive-seed6185.pt" \
    --epochs 2 --patience 2 --batch-size 2048 --hidden-size 64 \
    --seed 9185 --output "$out/validation.json" \
    > "$out/validation.log" 2>&1
status=$?
set -e
# A statistical gate failure is expected for the deliberately tiny smoke set;
# only crashes, missing outputs, or invalid artifacts fail this phase.
if [[ $status -ne 0 && $status -ne 2 ]]; then
    exit "$status"
fi
test -s "$out/validation.json"
test -s "$out/validation.pt"
touch "$out/COMPLETED"

#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
out="runs/drvoi_rhppo_v7/p1-response-formal"
mkdir -p "$out/data"

PYTHONPATH=. "$python_bin" -m benchmarks.build_drvoi_v7_manifest \
    --output "$out/manifest.json" > "$out/manifest.log" 2>&1

collection_seeds=(6185 7185 8185)
b0_seeds=(85 185 285)
for index in 0 1 2; do
    collection_seed="${collection_seeds[$index]}"
    b0_seed="${b0_seeds[$index]}"
    for dataset in cover onpolicy reactive; do
        target="$out/data/$dataset-seed$collection_seed.pt"
        if [[ ! -s "$target" ]]; then
            PYTHONPATH=. "$python_bin" benchmarks/collect_drvoi_v7_response_data.py \
                --b0-checkpoint "runs/drvoi_rhppo_v7/b0-1m/seed$b0_seed/latest.pt" \
                --dataset "$dataset" --seed "$collection_seed" --num-envs 96 \
                --output "$target" > "${target%.pt}.log" 2>&1
        fi
    done
done

PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_response_v7.py \
    --onpolicy "$out/data/onpolicy-seed6185.pt" \
               "$out/data/onpolicy-seed7185.pt" \
               "$out/data/onpolicy-seed8185.pt" \
    --cover "$out/data/cover-seed6185.pt" \
            "$out/data/cover-seed7185.pt" \
            "$out/data/cover-seed8185.pt" \
    --reactive "$out/data/reactive-seed6185.pt" \
               "$out/data/reactive-seed7185.pt" \
               "$out/data/reactive-seed8185.pt" \
    --epochs 30 --patience 6 --batch-size 4096 --hidden-size 256 \
    --seed 9185 --output "$out/validation.json" \
    > "$out/validation.log" 2>&1

touch "$out/COMPLETED"

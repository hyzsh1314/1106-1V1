#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
output_root="$project_root/runs/torch6dof_validation/v4-reward-ablation-3seed-1m"
seeds=(85 185 285)
variants=(competitive engagement attack-quality)
opponents=(hold pursuit evasive)

cd "$project_root"
mkdir -p "$output_root"

for variant in "${variants[@]}"; do
    for seed in "${seeds[@]}"; do
        run_dir="$output_root/$variant/seed$seed"
        mkdir -p "$run_dir"
        if [[ ! -f "$run_dir/COMPLETED" ]]; then
            echo "TRAIN variant=$variant seed=$seed"
            PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_ppo.py \
                --num-envs 96 \
                --rollout-steps 256 \
                --updates 41 \
                --seed "$seed" \
                --training-opponents staged \
                --reward-variant "$variant" \
                --output "$run_dir/train.json" \
                --checkpoint "$run_dir/latest.pt" \
                > "$run_dir/train.log" 2>&1
            touch "$run_dir/COMPLETED"
        fi
        for opponent in "${opponents[@]}"; do
            eval_file="$run_dir/eval-$opponent.json"
            if [[ ! -f "$eval_file" ]]; then
                echo "EVAL variant=$variant seed=$seed opponent=$opponent"
                PYTHONPATH=. "$python_bin" \
                    benchmarks/evaluate_torch_wvr_fixed_opponents.py \
                    --checkpoint "$run_dir/latest.pt" \
                    --opponent "$opponent" \
                    --episodes 192 \
                    --num-envs 96 \
                    --seed 10085 \
                    --output "$eval_file" \
                    > "$run_dir/eval-$opponent.log" 2>&1
            fi
        done
    done
done

touch "$output_root/COMPLETED"
echo "ALL_COMPLETED output=$output_root"

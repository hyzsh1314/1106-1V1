#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
output_root="$project_root/runs/torch6dof_validation/v4-bc-los-intercept-3seed"
seeds=(85 185 285)
opponents=(hold pursuit evasive)

cd "$project_root"
mkdir -p "$output_root"

for seed in "${seeds[@]}"; do
    seed_root="$output_root/seed$seed"
    bc_dir="$seed_root/bc"
    ppo_dir="$seed_root/ppo-0p5m"
    mkdir -p "$bc_dir" "$ppo_dir"

    if [[ ! -f "$bc_dir/COMPLETED" ]]; then
        echo "BC seed=$seed"
        PYTHONPATH=. "$python_bin" benchmarks/pretrain_torch_wvr_bc.py \
            --num-envs 96 --collection-steps 256 --iterations 8 \
            --epochs 4 --minibatch-size 4096 --seed "$seed" \
            --output "$bc_dir/train.json" \
            --checkpoint "$bc_dir/latest.pt" \
            > "$bc_dir/train.log" 2>&1
        touch "$bc_dir/COMPLETED"
    fi

    for opponent in "${opponents[@]}"; do
        eval_file="$bc_dir/eval-$opponent.json"
        if [[ ! -f "$eval_file" ]]; then
            echo "EVAL stage=bc seed=$seed opponent=$opponent"
            PYTHONPATH=. "$python_bin" \
                benchmarks/evaluate_torch_wvr_fixed_opponents.py \
                --checkpoint "$bc_dir/latest.pt" --opponent "$opponent" \
                --episodes 192 --num-envs 96 --seed 10085 \
                --output "$eval_file" > "$bc_dir/eval-$opponent.log" 2>&1
        fi
    done

    if [[ ! -f "$ppo_dir/COMPLETED" ]]; then
        echo "PPO seed=$seed"
        PYTHONPATH=. "$python_bin" benchmarks/train_torch_wvr_ppo.py \
            --num-envs 96 --rollout-steps 256 --updates 20 \
            --curriculum formal --training-opponents fixed-mix \
            --reward-variant competitive --seed "$seed" \
            --resume "$bc_dir/latest.pt" \
            --output "$ppo_dir/train.json" \
            --checkpoint "$ppo_dir/latest.pt" \
            > "$ppo_dir/train.log" 2>&1
        touch "$ppo_dir/COMPLETED"
    fi

    for opponent in "${opponents[@]}"; do
        eval_file="$ppo_dir/eval-$opponent.json"
        if [[ ! -f "$eval_file" ]]; then
            echo "EVAL stage=ppo seed=$seed opponent=$opponent"
            PYTHONPATH=. "$python_bin" \
                benchmarks/evaluate_torch_wvr_fixed_opponents.py \
                --checkpoint "$ppo_dir/latest.pt" --opponent "$opponent" \
                --episodes 192 --num-envs 96 --seed 10085 \
                --output "$eval_file" > "$ppo_dir/eval-$opponent.log" 2>&1
        fi
    done
    touch "$seed_root/COMPLETED"
done

touch "$output_root/COMPLETED"
echo "ALL_COMPLETED output=$output_root"

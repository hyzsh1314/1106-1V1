#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
PYTHON="/home/hongyun0405/.venvs/bvr1022/bin/python"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/torch6dof_validation/v4-evasive-intercept-curriculum-3seed-1m"
seeds=(85 185 285)
opponents=(hold pursuit evasive)

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

for seed in "${seeds[@]}"; do
  seed_root="${OUTPUT_ROOT}/seed${seed}"
  close_dir="${seed_root}/stage1-close-evasive"
  formal_dir="${seed_root}/stage2-formal-mix"
  mkdir -p "${close_dir}" "${formal_dir}"

  if [[ ! -f "${close_dir}/COMPLETED" ]]; then
    echo "TRAIN stage=close-evasive seed=${seed}"
    PYTHONPATH=. "${PYTHON}" benchmarks/train_torch_wvr_ppo.py \
      --num-envs 96 --rollout-steps 256 --updates 21 \
      --curriculum close --training-opponents evasive \
      --reward-variant competitive --seed "${seed}" \
      --output "${close_dir}/train.json" \
      --checkpoint "${close_dir}/latest.pt" \
      > "${close_dir}/train.log" 2>&1
    touch "${close_dir}/COMPLETED"
  fi

  if [[ ! -f "${formal_dir}/COMPLETED" ]]; then
    echo "TRAIN stage=formal-mix seed=${seed}"
    PYTHONPATH=. "${PYTHON}" benchmarks/train_torch_wvr_ppo.py \
      --num-envs 96 --rollout-steps 256 --updates 20 \
      --curriculum formal --training-opponents fixed-mix \
      --reward-variant competitive --seed "${seed}" \
      --resume "${close_dir}/latest.pt" \
      --output "${formal_dir}/train.json" \
      --checkpoint "${formal_dir}/latest.pt" \
      > "${formal_dir}/train.log" 2>&1
    touch "${formal_dir}/COMPLETED"
  fi

  for opponent in "${opponents[@]}"; do
    eval_json="${formal_dir}/eval-${opponent}.json"
    if [[ ! -f "${eval_json}" ]]; then
      echo "EVAL seed=${seed} opponent=${opponent}"
      PYTHONPATH=. "${PYTHON}" benchmarks/evaluate_torch_wvr_fixed_opponents.py \
        --checkpoint "${formal_dir}/latest.pt" \
        --opponent "${opponent}" --episodes 192 --num-envs 96 \
        --seed 10085 --output "${eval_json}" \
        > "${formal_dir}/eval-${opponent}.log" 2>&1
    fi
  done
  touch "${seed_root}/COMPLETED"
done

touch "${OUTPUT_ROOT}/COMPLETED"
echo "ALL_COMPLETED output=${OUTPUT_ROOT}"

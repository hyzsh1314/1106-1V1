#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WVR_PYTHON_BIN:-/home/hongyun0405/.venvs/bvr1022/bin/python}"
data_dir="${WVR_V12_A6_DATA_DIR:-${project_root}/runs/drvoi_rhppo_v12/return-teacher-a6-onpolicy-h60-20k-seed85}"
model_dir="${WVR_V12_A7_MODEL_DIR:-${project_root}/runs/drvoi_rhppo_v12/return-model-a7-onpolicy-behavior-h60-20k-seed85}"
v9_checkpoint="${project_root}/runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt"
response_checkpoint="${project_root}/runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt"
option_q_checkpoint="${project_root}/runs/drvoi_rhppo_v8/macro-option/development/cross-fitted-option-q.pt"
behavior_checkpoint="${project_root}/runs/drvoi_rhppo_v12/mrat-b5-behavior-gated-smoke-seed85/latest.pt"
reach_checkpoint="${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85/reachability.pt"
router_score_checkpoint="${project_root}/runs/drvoi_rhppo_v12/return-model-a5-behavior-h60-20k-seed85/reachability.pt"
mkdir -p "${data_dir}" "${model_dir}"
cd "${project_root}"

for required in "${v9_checkpoint}" "${response_checkpoint}" \
  "${option_q_checkpoint}" "${behavior_checkpoint}" \
  "${reach_checkpoint}" "${router_score_checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    printf 'missing prerequisite: %s\n' "${required}" >&2
    exit 2
  fi
done

if [[ ! -f "${data_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/collect_drvoi_v12_reachability_data.py \
    --checkpoint "${v9_checkpoint}" \
    --response-checkpoint "${response_checkpoint}" \
    --option-q-checkpoint "${option_q_checkpoint}" \
    --behavior-checkpoint "${behavior_checkpoint}" \
    --reachability-checkpoint "${reach_checkpoint}" \
    --router-score-checkpoint "${router_score_checkpoint}" \
    --router-intercept-margin 0.0 \
    --output-dir "${data_dir}" \
    --target-states 20000 --num-envs 96 --horizon 60 \
    --teacher-gamma 0.99 --boundary-stride 6 \
    --chunk-collections 8 --samples-per-member 4 \
    --seed 85 --device cuda --compile-dynamics --resume \
    2>&1 | tee -a "${data_dir}/collector.log"
fi

PYTHONPATH=. "${python_bin}" benchmarks/validate_drvoi_v12_reachability_data.py \
  --data-dir "${data_dir}" --output "${data_dir}/validation.json" \
  2>&1 | tee -a "${data_dir}/validation.log"

if [[ ! -f "${model_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/train_drvoi_v12_reachability.py \
    --data-dir "${data_dir}" --output-dir "${model_dir}" \
    --epochs 50 --batch-size 2048 --learning-rate 3e-4 \
    --hidden-size 128 --members 3 --softmin-temperature 0.25 \
    --censor-horizon 60 --score-target discounted_return_behavior \
    --seed 85 --device cuda 2>&1 | tee -a "${model_dir}/train.log"
fi

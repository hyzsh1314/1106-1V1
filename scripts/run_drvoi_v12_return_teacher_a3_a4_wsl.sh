#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WVR_PYTHON_BIN:-/home/hongyun0405/.venvs/bvr1022/bin/python}"
data_dir="${WVR_V12_RETURN_DATA_DIR:-${project_root}/runs/drvoi_rhppo_v12/return-teacher-a3-h60-20k-seed85}"
model_dir="${WVR_V12_RETURN_MODEL_DIR:-${project_root}/runs/drvoi_rhppo_v12/return-model-a4-h60-20k-seed85}"
mkdir -p "${data_dir}" "${model_dir}"
cd "${project_root}"

if [[ ! -f "${data_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/collect_drvoi_v12_reachability_data.py \
    --output-dir "${data_dir}" \
    --target-states 20000 \
    --num-envs 96 \
    --horizon 60 \
    --teacher-gamma 0.99 \
    --boundary-stride 6 \
    --chunk-collections 8 \
    --samples-per-member 4 \
    --seed 85 \
    --device cuda \
    --compile-dynamics \
    --resume 2>&1 | tee -a "${data_dir}/collector.log"
fi

PYTHONPATH=. "${python_bin}" benchmarks/validate_drvoi_v12_reachability_data.py \
  --data-dir "${data_dir}" --output "${data_dir}/validation.json" \
  2>&1 | tee -a "${data_dir}/validation.log"

if [[ ! -f "${model_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/train_drvoi_v12_reachability.py \
    --data-dir "${data_dir}" \
    --output-dir "${model_dir}" \
    --epochs 50 \
    --batch-size 2048 \
    --learning-rate 3e-4 \
    --hidden-size 128 \
    --members 3 \
    --softmin-temperature 0.25 \
    --censor-horizon 60 \
    --score-target discounted_return \
    --seed 85 \
    --device cuda 2>&1 | tee -a "${model_dir}/train.log"
fi


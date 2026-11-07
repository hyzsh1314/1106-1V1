#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WVR_PYTHON_BIN:-/home/hongyun0405/.venvs/bvr1022/bin/python}"
data_dir="${WVR_V12_DATA_DIR:-${project_root}/runs/drvoi_rhppo_v12/reachability-a1-seed85}"
model_dir="${WVR_V12_MODEL_DIR:-${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85}"
mkdir -p "${data_dir}" "${model_dir}"
cd "${project_root}"

if [[ ! -f "${data_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/collect_drvoi_v12_reachability_data.py \
    --output-dir "${data_dir}" \
    --target-states 50000 \
    --num-envs 96 \
    --horizon 30 \
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
    --censor-horizon 30 \
    --seed 85 \
    --device cuda 2>&1 | tee -a "${model_dir}/train.log"
fi

b0_dir="${WVR_V12_B0_DIR:-${project_root}/runs/drvoi_rhppo_v12/mrat-b0-seed85}"
mkdir -p "${b0_dir}"
if [[ ! -f "${b0_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/train_torch_wvr_mrat_rhppo.py \
    --v9-checkpoint runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt \
    --response-checkpoint runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt \
    --option-q-checkpoint runs/drvoi_rhppo_v8/macro-option/development/cross-fitted-option-q.pt \
    --reachability-checkpoint "${model_dir}/reachability.pt" \
    --output-dir "${b0_dir}" \
    --num-envs 96 \
    --rollout-steps 256 \
    --updates 4 \
    --epochs 2 \
    --minibatch-size 1024 \
    --bad-threshold -0.10 \
    --safe-threshold 0.10 \
    --intercept-advantage 0.05 \
    --force-steps 12 \
    --release-steps 6 \
    --seed 85 \
    --compile-dynamics \
    --resume 2>&1 | tee -a "${b0_dir}/train.log"
fi

for opponent in hold pursuit evasive; do
  evaluation="${b0_dir}/eval-${opponent}.json"
  if [[ ! -s "${evaluation}" ]]; then
    PYTHONPATH=. "${python_bin}" benchmarks/evaluate_torch_wvr_mrat_rhppo.py \
      --checkpoint "${b0_dir}/latest.pt" \
      --v9-checkpoint runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt \
      --response-checkpoint runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt \
      --option-q-checkpoint runs/drvoi_rhppo_v8/macro-option/development/cross-fitted-option-q.pt \
      --reachability-checkpoint "${model_dir}/reachability.pt" \
      --opponent "${opponent}" \
      --episodes 192 \
      --num-envs 96 \
      --seed 10085 \
      --output "${evaluation}" 2>&1 | tee -a "${b0_dir}/eval-${opponent}.log"
  fi
done

analysis_dir="${project_root}/docs/analysis/drvoi-rhppo-v12-b0"
PYTHONPATH=. "${python_bin}" benchmarks/analyze_drvoi_v12_b0.py \
  --run-dir "${b0_dir}" --output-dir "${analysis_dir}"


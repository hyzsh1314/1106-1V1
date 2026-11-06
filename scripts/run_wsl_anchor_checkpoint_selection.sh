#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_ANCHOR_1M_ROOT:-${PROJECT_ROOT}/runs/anchor_curriculum_1m_paired}"
BASELINE_ROOT="${WVR_TIMEOUT_ZERO_1M_ROOT:-${PROJECT_ROOT}/runs/timeout_zero_1m_paired}"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

seeds=(85 185 285)
updates=(20 40 41)
opponents=(pursuit_fire hold evasive_fire standoff_fire)
for seed in "${seeds[@]}"; do
  run_dir="${ROOT_DIR}/candidate-runs/seed${seed}_1m"
  for update in "${updates[@]}"; do
    if [[ "${update}" == "41" ]]; then
      checkpoint="${run_dir}/latest.pt"
    else
      checkpoint="${run_dir}/checkpoints/checkpoint_$(printf '%06d' "${update}").pt"
    fi
    policy_index=0
    for opponent in "${opponents[@]}"; do
      output_dir="${ROOT_DIR}/validation/seed${seed}/update${update}/${opponent}"
      if [[ ! -f "${output_dir}/summary.json" ]]; then
        python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
          --checkpoint "${checkpoint}" --episodes 64 --num-envs 64 \
          --seed "$((1200000 + seed * 100 + update * 10 + policy_index))" \
          --device cuda --start-method forkserver \
          --opponent-policy "${opponent}" --output-dir "${output_dir}"
      fi
      policy_index=$((policy_index + 1))
    done
  done
done

python -u -m benchmarks.select_anchor_curriculum_checkpoint \
  --root-dir "${ROOT_DIR}"

for seed in "${seeds[@]}"; do
  checkpoint_file="$(python -c "import json; print(json.load(open('${ROOT_DIR}/checkpoint-selection/selected.json'))['${seed}']['checkpoint_file'])")"
  checkpoint="${ROOT_DIR}/candidate-runs/seed${seed}_1m/${checkpoint_file}"
  policy_index=0
  for opponent in "${opponents[@]}"; do
    output_dir="${ROOT_DIR}/heldout/seed${seed}/${opponent}"
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${checkpoint}" --episodes 192 --num-envs 96 \
        --seed "$((1300000 + seed * 100 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${output_dir}"
    fi
    policy_index=$((policy_index + 1))
  done
done

python -u -m benchmarks.analyze_anchor_checkpoint_selection \
  --root-dir "${ROOT_DIR}" --baseline-root "${BASELINE_ROOT}"

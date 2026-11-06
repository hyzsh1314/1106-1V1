#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
RUN_DIR="${PROJECT_ROOT}/runs/threat_obs_v2_evasive_curriculum_seed85_1p5m"
OUTPUT_ROOT="${RUN_DIR}/heldout"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

opponents=(pursuit_fire hold evasive_fire standoff_fire)
policy_index=0
for opponent in "${opponents[@]}"; do
  output_dir="${OUTPUT_ROOT}/${opponent}"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
      --checkpoint "${RUN_DIR}/latest.pt" --episodes 192 --num-envs 96 \
      --seed "$((1512500 + policy_index))" \
      --device cuda --start-method forkserver \
      --opponent-policy "${opponent}" --output-dir "${output_dir}"
  fi
  policy_index=$((policy_index + 1))
done

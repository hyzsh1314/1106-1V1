#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
CHECKPOINT="${PROJECT_ROOT}/runs/altitude_safety_1m_paired/candidate-runs/seed85_1m/checkpoints/checkpoint_000020.pt"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/altitude_safety_1m_paired/update20-seed85-screen"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

opponents=(pursuit_fire hold evasive_fire standoff_fire)
policy_index=0
for opponent in "${opponents[@]}"; do
  output_dir="${OUTPUT_ROOT}/${opponent}"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
      --checkpoint "${CHECKPOINT}" --episodes 96 --num-envs 48 \
      --seed "$((1012500 + policy_index))" \
      --device cuda --start-method forkserver \
      --opponent-policy "${opponent}" --output-dir "${output_dir}"
  fi
  policy_index=$((policy_index + 1))
done

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_TIMEOUT_ZERO_ROOT:-${PROJECT_ROOT}/runs/timeout_zero_retrospective}"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

declare -a pairs=(
  "pursuit_fire evasive_fire"
  "standoff_fire evasive_fire"
  "evasive_fire evasive_fire"
)
pair_index=0
for pair in "${pairs[@]}"; do
  read -r policy_a policy_b <<< "${pair}"
  output_dir="${ROOT_DIR}/scripted-crossplay/${policy_a}_vs_${policy_b}"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    python -u -m benchmarks.evaluate_scripted_crossplay_1v1 \
      --policy-a "${policy_a}" --policy-b "${policy_b}" \
      --episodes 300 --num-envs 96 \
      --seed "$((991000 + pair_index * 10000))" \
      --start-method forkserver --output-dir "${output_dir}"
  fi
  pair_index=$((pair_index + 1))
done

SOURCE_ROOT="${PROJECT_ROOT}/runs/r1_attack_pbrs_5m_validation"
seeds=(85 185 285)
# Six longitudinal points are enough to expose regression while keeping the
# diagnostic sweep shorter than a new training run.
updates=(40 80 120 160 200 204)
for seed in "${seeds[@]}"; do
  for update in "${updates[@]}"; do
    checkpoint="${SOURCE_ROOT}/seed${seed}_5m/checkpoints/checkpoint_$(printf '%06d' "${update}").pt"
    output_dir="${ROOT_DIR}/seed${seed}/checkpoint_$(printf '%06d' "${update}")/evasive_fire"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing checkpoint: ${checkpoint}" >&2
      exit 2
    fi
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${checkpoint}" --episodes 96 --num-envs 48 \
        --seed "$((992000 + seed * 1000 + update))" \
        --device cuda --start-method forkserver \
        --opponent-policy evasive_fire --output-dir "${output_dir}"
    fi
  done
done

python -u -m benchmarks.analyze_timeout_zero_retrospective \
  --root-dir "${ROOT_DIR}"

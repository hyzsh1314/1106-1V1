#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_ALTITUDE_SAFETY_1M_ROOT:-${PROJECT_ROOT}/runs/altitude_safety_1m_paired}"
BASELINE_ROOT="${WVR_TIMEOUT_ZERO_1M_ROOT:-${PROJECT_ROOT}/runs/timeout_zero_1m_paired}"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

seeds=(85 185 285)
opponents=(pursuit_fire hold evasive_fire standoff_fire)
for seed in "${seeds[@]}"; do
  run_dir="${ROOT_DIR}/candidate-runs/seed${seed}_1m"
  mkdir -p "${run_dir}"
  if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
    resume_args=()
    if [[ -f "${run_dir}/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/latest.pt")
    fi
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant R1_attack_pbrs --seed "${seed}" \
      --num-envs 96 --rollout-steps 256 --updates 41 \
      --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
      --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
      --entropy-coef 0.001 --max-grad-norm 2.0 \
      --population-capacity 16 --snapshot-interval 2 \
      --max-opponents-per-update 8 --scripted-anchor-fraction 0 \
      --altitude-safety-mask --altitude-descend-mask-m 4000 \
      --altitude-force-climb-m 3250 \
      --checkpoint-interval 5 --archive-interval 20 \
      --run-dir "${run_dir}" "${resume_args[@]}" \
      2>&1 | tee -a "${run_dir}/train.log"
  fi

  policy_index=0
  for opponent in "${opponents[@]}"; do
    output_dir="${ROOT_DIR}/candidate/seed${seed}/${opponent}"
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${run_dir}/latest.pt" --episodes 192 --num-envs 96 \
        --seed "$((994000 + seed * 100 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${output_dir}"
    fi
    heldout_dir="${ROOT_DIR}/heldout/seed${seed}/${opponent}"
    if [[ ! -f "${heldout_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${run_dir}/latest.pt" --episodes 192 --num-envs 96 \
        --seed "$((1494000 + seed * 100 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${heldout_dir}"
    fi
    policy_index=$((policy_index + 1))
  done
done

python -u -m benchmarks.analyze_altitude_safety_1m \
  --root-dir "${ROOT_DIR}" --baseline-root "${BASELINE_ROOT}"

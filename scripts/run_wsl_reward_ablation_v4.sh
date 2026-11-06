#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_REWARD_ROOT:-${PROJECT_ROOT}/runs/reward_v4_ablation}"
if [[ -n "${WVR_VENV:-}" ]]; then
  VENV_DIR="${WVR_VENV}"
elif [[ -x "${HOME}/.venvs/bvr1022/bin/python" ]]; then
  VENV_DIR="${HOME}/.venvs/bvr1022"
else
  VENV_DIR="${HOME}/.venvs/wvr-env"
fi

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

variants=(R0_terminal R1_attack_pbrs R2_threat_pbrs R3_constraints)
seeds=(85 185 285)
for variant in "${variants[@]}"; do
  for seed in "${seeds[@]}"; do
    run_dir="${ROOT_DIR}/${variant}_seed${seed}_1m"
    mkdir -p "${run_dir}"
    if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
      resume_args=()
      if [[ -f "${run_dir}/latest.pt" ]]; then
        resume_args=(--resume "${run_dir}/latest.pt")
      fi
      python -u -m benchmarks.train_ppo_population_1v1 \
        --device cuda --amp --start-method forkserver \
        --reward-variant "${variant}" --seed "${seed}" \
        --num-envs 96 --rollout-steps 256 --updates 41 \
        --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
        --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
        --entropy-coef 0.001 --max-grad-norm 2.0 \
        --population-capacity 16 --snapshot-interval 2 \
        --max-opponents-per-update 8 --checkpoint-interval 5 \
        --archive-interval 41 --run-dir "${run_dir}" \
        "${resume_args[@]}" 2>&1 | tee -a "${run_dir}/train.log"
    fi

    for opponent in pursuit_fire hold; do
      eval_dir="${run_dir}/evaluation-${opponent}"
      if [[ ! -f "${eval_dir}/summary.json" ]]; then
        python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
          --checkpoint "${run_dir}/latest.pt" \
          --episodes 192 --num-envs 96 --seed "$((950000 + seed))" \
          --device cuda --start-method forkserver \
          --opponent-policy "${opponent}" --output-dir "${eval_dir}" \
          2>&1 | tee -a "${run_dir}/evaluation-${opponent}.log"
      fi
    done
  done
done

python -u -m benchmarks.analyze_reward_ablation_v4 --root-dir "${ROOT_DIR}"


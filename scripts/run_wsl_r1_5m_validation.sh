#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="${WVR_R1_5M_ROOT:-${PROJECT_ROOT}/runs/r1_attack_pbrs_5m_validation}"
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

if [[ ! -f "${ROOT_DIR}/initial-state-audit/summary.json" ]]; then
  python -u -m benchmarks.audit_initial_engagement_1v1 \
    --samples 300 --seed 730085 \
    --output-dir "${ROOT_DIR}/initial-state-audit"
fi

seeds=(85 185 285)
for seed in "${seeds[@]}"; do
  run_dir="${ROOT_DIR}/seed${seed}_5m"
  mkdir -p "${run_dir}"
  if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
    resume_args=()
    if [[ -f "${run_dir}/latest.pt" ]]; then
      resume_args=(--resume "${run_dir}/latest.pt")
    fi
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant R1_attack_pbrs --seed "${seed}" \
      --num-envs 96 --rollout-steps 256 --updates 204 \
      --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
      --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
      --entropy-coef 0.001 --max-grad-norm 2.0 \
      --population-capacity 16 --snapshot-interval 2 \
      --max-opponents-per-update 8 --checkpoint-interval 5 \
      --archive-interval 20 --run-dir "${run_dir}" \
      "${resume_args[@]}" 2>&1 | tee -a "${run_dir}/train.log"
  fi

  # Lightweight longitudinal evaluation: 96 episodes every 491,520 steps.
  for update in $(seq 20 20 200); do
    checkpoint="${run_dir}/checkpoints/checkpoint_$(printf '%06d' "${update}").pt"
    eval_dir="${run_dir}/periodic-evaluation/checkpoint_$(printf '%06d' "${update}")"
    if [[ ! -f "${eval_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${checkpoint}" --episodes 96 --num-envs 48 \
        --seed "$((960000 + seed * 100 + update))" \
        --device cuda --start-method forkserver \
        --opponent-policy pursuit_fire --output-dir "${eval_dir}"
    fi
  done

  policy_index=0
  for opponent in pursuit_fire hold evasive_fire standoff_fire; do
    eval_dir="${run_dir}/final-evaluation/${opponent}"
    if [[ ! -f "${eval_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${run_dir}/latest.pt" \
        --episodes 300 --num-envs 96 \
        --seed "$((970000 + seed * 10 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${eval_dir}"
    fi
    policy_index=$((policy_index + 1))
  done
done

for learner_seed in "${seeds[@]}"; do
  for opponent_seed in "${seeds[@]}"; do
    output_dir="${ROOT_DIR}/crossplay/seed${learner_seed}_vs_seed${opponent_seed}"
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_crossplay_1v1 \
        --learner-checkpoint "${ROOT_DIR}/seed${learner_seed}_5m/latest.pt" \
        --opponent-checkpoint "${ROOT_DIR}/seed${opponent_seed}_5m/latest.pt" \
        --episodes 192 --num-envs 96 \
        --seed "$((980000 + learner_seed * 1000 + opponent_seed))" \
        --device cuda --start-method forkserver --output-dir "${output_dir}"
    fi
  done
done

python -u -m benchmarks.analyze_r1_5m_validation --root-dir "${ROOT_DIR}"


#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/threat_obs_v2_multiseed_confirmation"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Together with the already completed seed=85 experiment, these two seeds give
# three independent training replicates.  Evaluation seeds are deliberately
# identical across training replicates to reduce evaluation-noise differences.
seeds=(17 127)
opponents=(pursuit_fire hold evasive_fire standoff_fire)

for seed in "${seeds[@]}"; do
  seed_root="${OUTPUT_ROOT}/seed${seed}"
  stage1="${seed_root}/stage1_population_500k"
  stage2="${seed_root}/stage2_population_1m"
  stage3="${seed_root}/stage3_evasive_focused_1p5m"
  mkdir -p "${stage1}" "${stage2}" "${stage3}"

  if [[ ! -f "${stage1}/COMPLETED.json" ]]; then
    resume_args=()
    if [[ -f "${stage1}/latest.pt" ]]; then
      resume_args=(--resume "${stage1}/latest.pt")
    fi
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant R1_attack_pbrs --seed "${seed}" \
      --num-envs 96 --rollout-steps 256 --updates 21 \
      --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
      --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
      --entropy-coef 0.001 --max-grad-norm 2.0 \
      --population-capacity 16 --snapshot-interval 2 \
      --max-opponents-per-update 8 --scripted-anchor-fraction 0 \
      --altitude-safety-mask --altitude-descend-mask-m 4000 \
      --altitude-force-climb-m 3250 --threat-observation-v2 \
      --checkpoint-interval 5 --archive-interval 20 \
      --run-dir "${stage1}" "${resume_args[@]}" \
      2>&1 | tee -a "${stage1}/train.log"
  fi

  if [[ ! -f "${stage2}/COMPLETED.json" ]]; then
    resume_checkpoint="${stage1}/latest.pt"
    if [[ -f "${stage2}/latest.pt" ]]; then
      resume_checkpoint="${stage2}/latest.pt"
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
      --altitude-force-climb-m 3250 --threat-observation-v2 \
      --checkpoint-interval 5 --archive-interval 20 \
      --run-dir "${stage2}" --resume "${resume_checkpoint}" \
      2>&1 | tee -a "${stage2}/train.log"
  fi

  if [[ ! -f "${stage3}/COMPLETED.json" ]]; then
    resume_checkpoint="${stage2}/latest.pt"
    if [[ -f "${stage3}/latest.pt" ]]; then
      resume_checkpoint="${stage3}/latest.pt"
    fi
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant R1_attack_pbrs --seed "${seed}" \
      --num-envs 96 --rollout-steps 256 --updates 61 \
      --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
      --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
      --entropy-coef 0.001 --max-grad-norm 2.0 \
      --population-capacity 16 --snapshot-interval 2 \
      --max-opponents-per-update 8 --scripted-anchor-fraction 1 \
      --scripted-anchor-policies evasive_fire \
      --altitude-safety-mask --altitude-descend-mask-m 4000 \
      --altitude-force-climb-m 3250 --threat-observation-v2 \
      --allow-curriculum-override-on-resume \
      --checkpoint-interval 5 --archive-interval 20 \
      --run-dir "${stage3}" --resume "${resume_checkpoint}" \
      2>&1 | tee -a "${stage3}/train.log"
  fi

  policy_index=0
  for opponent in "${opponents[@]}"; do
    output_dir="${stage3}/heldout/${opponent}"
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${stage3}/latest.pt" --episodes 192 --num-envs 96 \
        --seed "$((2512500 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${output_dir}"
    fi
    policy_index=$((policy_index + 1))
  done
done

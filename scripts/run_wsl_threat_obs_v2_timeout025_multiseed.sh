#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
OUTPUT_ROOT="${PROJECT_ROOT}/runs/threat_obs_v2_timeout025_ablation"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

seeds=(17 85 127)
opponents=(pursuit_fire hold evasive_fire standoff_fire)

for seed in "${seeds[@]}"; do
  source_checkpoint="${PROJECT_ROOT}/runs/threat_obs_v2_wide_heading_ablation/seed${seed}_span90_1p5m/latest.pt"
  run_dir="${OUTPUT_ROOT}/seed${seed}_span90_timeout025_2m"
  mkdir -p "${run_dir}"
  if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
    resume_checkpoint="${source_checkpoint}"
    if [[ -f "${run_dir}/latest.pt" ]]; then
      resume_checkpoint="${run_dir}/latest.pt"
    fi
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant R1_timeout025 --seed "${seed}" \
      --num-envs 96 --rollout-steps 256 --updates 81 \
      --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
      --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
      --entropy-coef 0.001 --max-grad-norm 2.0 \
      --population-capacity 16 --snapshot-interval 2 \
      --max-opponents-per-update 8 \
      --scripted-anchor-fraction 1.0 \
      --scripted-anchor-policies evasive_fire \
      --altitude-safety-mask --altitude-descend-mask-m 4000 \
      --altitude-force-climb-m 3250 --threat-observation-v2 \
      --heading-command-span-deg 90 \
      --allow-reward-override-on-resume \
      --checkpoint-interval 5 --archive-interval 20 \
      --run-dir "${run_dir}" --resume "${resume_checkpoint}" \
      2>&1 | tee -a "${run_dir}/train.log"
  fi

  policy_index=0
  for opponent in "${opponents[@]}"; do
    output_dir="${run_dir}/heldout/${opponent}"
    if [[ ! -f "${output_dir}/summary.json" ]]; then
      python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
        --checkpoint "${run_dir}/latest.pt" --episodes 192 --num-envs 96 \
        --seed "$((2512500 + policy_index))" \
        --device cuda --start-method forkserver \
        --opponent-policy "${opponent}" --output-dir "${output_dir}"
    fi
    policy_index=$((policy_index + 1))
  done
done

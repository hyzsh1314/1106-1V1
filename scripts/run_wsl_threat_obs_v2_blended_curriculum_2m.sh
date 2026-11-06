#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
SOURCE_RUN="${PROJECT_ROOT}/runs/threat_obs_v2_evasive_curriculum_seed85_1p5m"
RUN_DIR="${PROJECT_ROOT}/runs/threat_obs_v2_blended_curriculum_seed85_2m"
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ ! -f "${RUN_DIR}/COMPLETED.json" ]]; then
  resume_checkpoint="${SOURCE_RUN}/latest.pt"
  if [[ -f "${RUN_DIR}/latest.pt" ]]; then
    resume_checkpoint="${RUN_DIR}/latest.pt"
  fi
  python -u -m benchmarks.train_ppo_population_1v1 \
    --device cuda --amp --start-method forkserver \
    --reward-variant R1_attack_pbrs --seed 85 \
    --num-envs 96 --rollout-steps 256 --updates 81 \
    --epochs 4 --minibatch-size 2048 --learning-rate 0.0003 \
    --gamma 0.99 --gae-lambda 0.95 --clip-ratio 0.2 \
    --entropy-coef 0.001 --max-grad-norm 2.0 \
    --population-capacity 16 --snapshot-interval 2 \
    --max-opponents-per-update 8 --scripted-anchor-fraction 0.25 \
    --scripted-anchor-policies evasive_fire \
    --altitude-safety-mask --altitude-descend-mask-m 4000 \
    --altitude-force-climb-m 3250 --threat-observation-v2 \
    --allow-curriculum-override-on-resume \
    --checkpoint-interval 5 --archive-interval 20 \
    --run-dir "${RUN_DIR}" --resume "${resume_checkpoint}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
fi

opponents=(pursuit_fire hold evasive_fire standoff_fire)
policy_index=0
for opponent in "${opponents[@]}"; do
  output_dir="${RUN_DIR}/eval/${opponent}"
  if [[ ! -f "${output_dir}/summary.json" ]]; then
    python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
      --checkpoint "${RUN_DIR}/latest.pt" --episodes 96 --num-envs 48 \
      --seed "$((1012500 + policy_index))" \
      --device cuda --start-method forkserver \
      --opponent-policy "${opponent}" --output-dir "${output_dir}"
  fi
  policy_index=$((policy_index + 1))
done

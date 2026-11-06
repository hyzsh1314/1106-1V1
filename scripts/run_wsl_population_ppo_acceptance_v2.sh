#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${WVR_VENV:-${HOME}/.venvs/bvr1022}"
RUN_DIR="${WVR_RUN_DIR:-${PROJECT_ROOT}/runs/ppo_population_1v1_v2_acceptance_seed85_1m}"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/latest.pt" && ! -f "${RUN_DIR}/COMPLETED.json" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/latest.pt")
fi

python -u -m benchmarks.train_ppo_population_1v1 \
  --device cuda \
  --amp \
  --start-method forkserver \
  --seed 85 \
  --num-envs 96 \
  --rollout-steps 256 \
  --updates 41 \
  --epochs 4 \
  --minibatch-size 2048 \
  --learning-rate 0.0003 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --clip-ratio 0.2 \
  --entropy-coef 0.001 \
  --max-grad-norm 2.0 \
  --population-capacity 16 \
  --snapshot-interval 2 \
  --max-opponents-per-update 8 \
  --checkpoint-interval 5 \
  --archive-interval 41 \
  --run-dir "${RUN_DIR}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${RUN_DIR}/train.log"

python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
  --checkpoint "${RUN_DIR}/latest.pt" \
  --episodes 1000 \
  --num-envs 96 \
  --seed 950085 \
  --device cuda \
  --start-method forkserver \
  --output-dir "${RUN_DIR}/acceptance-evaluation" \
  2>&1 | tee "${RUN_DIR}/evaluation.log"

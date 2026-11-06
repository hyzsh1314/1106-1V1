#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${WVR_VENV:-}" ]]; then
  VENV_DIR="${WVR_VENV}"
elif [[ -x "${HOME}/.venvs/bvr1022/bin/python" ]]; then
  VENV_DIR="${HOME}/.venvs/bvr1022"
else
  VENV_DIR="${HOME}/.venvs/wvr-env"
fi

RUN_DIR="${WVR_RUN_DIR:-${PROJECT_ROOT}/runs/ppo_population_1v1_baseline_seed85_100m}"
mkdir -p "${RUN_DIR}"
if [[ -f "${RUN_DIR}/COMPLETED.json" ]]; then
  echo "formal baseline is already complete: ${RUN_DIR}"
  exit 0
fi

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RESUME_ARGS=()
if [[ -f "${RUN_DIR}/latest.pt" ]]; then
  RESUME_ARGS=(--resume "${RUN_DIR}/latest.pt")
fi

exec python -u -m benchmarks.train_ppo_population_1v1 \
  --device cuda \
  --amp \
  --start-method forkserver \
  --seed "${WVR_SEED:-85}" \
  --num-envs "${WVR_NUM_ENVS:-96}" \
  --envs-per-worker "${WVR_ENVS_PER_WORKER:-2}" \
  --rollout-steps "${WVR_ROLLOUT_STEPS:-256}" \
  --updates "${WVR_UPDATES:-4070}" \
  --epochs "${WVR_EPOCHS:-4}" \
  --minibatch-size "${WVR_MINIBATCH_SIZE:-2048}" \
  --learning-rate "${WVR_LEARNING_RATE:-0.0003}" \
  --gamma "${WVR_GAMMA:-0.99}" \
  --gae-lambda "${WVR_GAE_LAMBDA:-0.95}" \
  --clip-ratio "${WVR_CLIP_RATIO:-0.2}" \
  --entropy-coef "${WVR_ENTROPY_COEF:-0.001}" \
  --max-grad-norm "${WVR_MAX_GRAD_NORM:-2.0}" \
  --population-capacity "${WVR_POPULATION_CAPACITY:-64}" \
  --snapshot-interval "${WVR_SNAPSHOT_INTERVAL:-20}" \
  --max-opponents-per-update "${WVR_MAX_OPPONENTS:-8}" \
  --checkpoint-interval "${WVR_CHECKPOINT_INTERVAL:-5}" \
  --archive-interval "${WVR_ARCHIVE_INTERVAL:-250}" \
  --run-dir "${RUN_DIR}" \
  "${RESUME_ARGS[@]}"

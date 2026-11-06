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

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

exec python -m benchmarks.train_ppo_population_1v1 \
  --device "${WVR_DEVICE:-cuda}" \
  --amp \
  --start-method forkserver \
  --num-envs "${WVR_NUM_ENVS:-8}" \
  --rollout-steps "${WVR_ROLLOUT_STEPS:-256}" \
  --updates "${WVR_UPDATES:-4}" \
  --minibatch-size "${WVR_MINIBATCH_SIZE:-2048}" \
  --population-capacity "${WVR_POPULATION_CAPACITY:-16}"

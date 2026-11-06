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

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "error: ${VENV_DIR} does not exist; run scripts/setup_wsl_gpu.sh first" >&2
  exit 1
fi
source "${VENV_DIR}/bin/activate"
cd "${PROJECT_ROOT}"

export WVR_WORKER_THREADS="${WVR_WORKER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

CPU_COUNT="$(nproc)"
DEFAULT_ENVS=$((CPU_COUNT > 18 ? 16 : CPU_COUNT - 2))
if (( DEFAULT_ENVS < 2 )); then DEFAULT_ENVS=2; fi

NUM_ENVS="${WVR_NUM_ENVS:-${DEFAULT_ENVS}}"
ENVS_PER_WORKER="${WVR_ENVS_PER_WORKER:-2}"
ROLLOUT_STEPS="${WVR_ROLLOUT_STEPS:-512}"
UPDATES="${WVR_UPDATES:-4}"
MINIBATCH_SIZE="${WVR_MINIBATCH_SIZE:-4096}"
DEVICE="${WVR_DEVICE:-cuda}"

if [[ "${DEVICE}" == "cuda" ]]; then
  AMP_ARG=--amp
else
  AMP_ARG=--no-amp
fi

echo "venv=${VENV_DIR} device=${DEVICE} cpu_count=${CPU_COUNT} num_envs=${NUM_ENVS} envs_per_worker=${ENVS_PER_WORKER} rollout_steps=${ROLLOUT_STEPS} minibatch=${MINIBATCH_SIZE}"
exec python -m benchmarks.train_ppo_1v1_shoot_smoke \
  --device "${DEVICE}" \
  "${AMP_ARG}" \
  --start-method forkserver \
  --num-envs "${NUM_ENVS}" \
  --envs-per-worker "${ENVS_PER_WORKER}" \
  --rollout-steps "${ROLLOUT_STEPS}" \
  --updates "${UPDATES}" \
  --minibatch-size "${MINIBATCH_SIZE}"

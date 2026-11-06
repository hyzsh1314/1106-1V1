#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_INDEX_URL="${WVR_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

if [[ -n "${WVR_VENV:-}" ]]; then
  VENV_DIR="${WVR_VENV}"
elif [[ -x "${HOME}/.venvs/bvr1022/bin/python" ]]; then
  VENV_DIR="${HOME}/.venvs/bvr1022"
else
  VENV_DIR="${HOME}/.venvs/wvr-env"
fi

if ! grep -qi microsoft /proc/version; then
  echo "error: this installer is intended for WSL2" >&2
  exit 1
fi

NVIDIA_SMI="$(command -v nvidia-smi || true)"
if [[ -z "${NVIDIA_SMI}" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  NVIDIA_SMI=/usr/lib/wsl/lib/nvidia-smi
fi
if [[ -z "${NVIDIA_SMI}" ]]; then
  echo "error: WSL cannot see nvidia-smi; update WSL and the Windows NVIDIA driver" >&2
  exit 1
fi

echo "project=${PROJECT_ROOT}"
echo "venv=${VENV_DIR}"
"${NVIDIA_SMI}" --query-gpu=name,driver_version,memory.total --format=csv,noheader

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install the CUDA wheel before the common requirements.  The subsequent
# torch>= constraint is then already satisfied and cannot silently select the
# CPU-only wheel.
if python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  echo "Reusing CUDA-enabled PyTorch from ${VENV_DIR}"
else
  python -m pip install torch --index-url "${TORCH_INDEX_URL}"
fi
python -m pip install -r "${PROJECT_ROOT}/requirements-environment.txt"

cd "${PROJECT_ROOT}"
python -m benchmarks.diagnose_wsl_runtime --require-cuda --environment-smoke

echo
echo "WSL GPU environment is ready. Activate it with:"
echo "  source '${VENV_DIR}/bin/activate'"
echo "Run the vectorised PPO smoke test with:"
echo "  bash '${PROJECT_ROOT}/scripts/run_wsl_gpu_ppo.sh'"

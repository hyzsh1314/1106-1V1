#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${WVR_VENV:-${HOME}/.venvs/bvr1022}/bin/activate"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for variant in R0_terminal R1_attack_pbrs R2_threat_pbrs R3_constraints; do
  run_dir="runs/reward_v4_training_smoke/${variant}"
  if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
    python -u -m benchmarks.train_ppo_population_1v1 \
      --device cuda --amp --start-method forkserver \
      --reward-variant "${variant}" --seed 4085 \
      --num-envs 16 --rollout-steps 64 --updates 2 --epochs 2 \
      --minibatch-size 512 --population-capacity 4 \
      --snapshot-interval 1 --max-opponents-per-update 4 \
      --checkpoint-interval 1 --archive-interval 2 --run-dir "${run_dir}"
  fi
done

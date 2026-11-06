#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${WVR_VENV:-${HOME}/.venvs/bvr1022}/bin/activate"
cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN_DIR="runs/r1_5m_pipeline_smoke/train"
if [[ ! -f "${RUN_DIR}/COMPLETED.json" ]]; then
  python -u -m benchmarks.train_ppo_population_1v1 \
    --device cuda --amp --start-method forkserver \
    --reward-variant R1_attack_pbrs --seed 5085 \
    --num-envs 16 --rollout-steps 64 --updates 2 --epochs 2 \
    --minibatch-size 512 --population-capacity 4 \
    --snapshot-interval 1 --max-opponents-per-update 4 \
    --checkpoint-interval 1 --archive-interval 1 --run-dir "${RUN_DIR}"
fi
for opponent in pursuit_fire hold evasive_fire standoff_fire; do
  output="runs/r1_5m_pipeline_smoke/evaluation-${opponent}"
  if [[ ! -f "${output}/summary.json" ]]; then
    python -u -m benchmarks.evaluate_ppo_1v1_acceptance \
      --checkpoint "${RUN_DIR}/latest.pt" --episodes 8 --num-envs 4 \
      --seed 995085 --device cuda --start-method forkserver \
      --opponent-policy "${opponent}" --output-dir "${output}"
  fi
done
python -u -m benchmarks.evaluate_ppo_crossplay_1v1 \
  --learner-checkpoint "${RUN_DIR}/latest.pt" \
  --opponent-checkpoint "${RUN_DIR}/latest.pt" \
  --episodes 8 --num-envs 4 --seed 996085 --device cuda \
  --start-method forkserver \
  --output-dir runs/r1_5m_pipeline_smoke/crossplay-self

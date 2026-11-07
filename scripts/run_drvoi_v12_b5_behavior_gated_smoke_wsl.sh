#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WVR_PYTHON_BIN:-/home/hongyun0405/.venvs/bvr1022/bin/python}"
run_dir="${WVR_V12_B5_DIR:-${project_root}/runs/drvoi_rhppo_v12/mrat-b5-behavior-gated-smoke-seed85}"
analysis_dir="${WVR_V12_B5_ANALYSIS_DIR:-${project_root}/docs/analysis/drvoi-rhppo-v12-b5-behavior-gated-smoke}"
reach_checkpoint="${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85/reachability.pt"
return_checkpoint="${project_root}/runs/drvoi_rhppo_v12/return-model-a5-behavior-h60-20k-seed85/reachability.pt"
v9_checkpoint="${project_root}/runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt"
response_checkpoint="${project_root}/runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt"
option_q_checkpoint="${project_root}/runs/drvoi_rhppo_v8/macro-option/development/cross-fitted-option-q.pt"
mkdir -p "${run_dir}" "${analysis_dir}"
cd "${project_root}"

for required in "${reach_checkpoint}" "${return_checkpoint}" \
  "${v9_checkpoint}" "${response_checkpoint}" "${option_q_checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    printf 'missing prerequisite: %s\n' "${required}" >&2
    exit 2
  fi
done

if [[ ! -f "${run_dir}/COMPLETED.json" ]]; then
  PYTHONPATH=. "${python_bin}" benchmarks/train_torch_wvr_mrat_rhppo.py \
    --v9-checkpoint "${v9_checkpoint}" \
    --response-checkpoint "${response_checkpoint}" \
    --option-q-checkpoint "${option_q_checkpoint}" \
    --reachability-checkpoint "${reach_checkpoint}" \
    --router-score-checkpoint "${return_checkpoint}" \
    --router-intercept-margin 0.0 \
    --output-dir "${run_dir}" \
    --num-envs 96 --rollout-steps 256 --updates 4 \
    --epochs 2 --minibatch-size 1024 \
    --bad-threshold -0.10 --safe-threshold 0.10 \
    --intercept-advantage 0.05 --force-steps 12 --release-steps 6 \
    --seed 85 --stage-label b5-behavior-gated-smoke-seed85 \
    --compile-dynamics --resume 2>&1 | tee -a "${run_dir}/train.log"
fi

for opponent in hold pursuit evasive; do
  evaluation="${run_dir}/eval-${opponent}.json"
  episode_records="${run_dir}/eval-${opponent}-episodes.jsonl"
  segment_records="${run_dir}/eval-${opponent}-segments.jsonl"
  if [[ ! -s "${evaluation}" || ! -s "${episode_records}" || ! -s "${segment_records}" ]]; then
    PYTHONPATH=. "${python_bin}" benchmarks/evaluate_torch_wvr_mrat_rhppo.py \
      --checkpoint "${run_dir}/latest.pt" \
      --v9-checkpoint "${v9_checkpoint}" \
      --response-checkpoint "${response_checkpoint}" \
      --option-q-checkpoint "${option_q_checkpoint}" \
      --reachability-checkpoint "${reach_checkpoint}" \
      --router-score-checkpoint "${return_checkpoint}" \
      --opponent "${opponent}" --episodes 192 --num-envs 96 --seed 10085 \
      --output "${evaluation}" --episode-records "${episode_records}" \
      --segment-records "${segment_records}" \
      2>&1 | tee -a "${run_dir}/eval-${opponent}.log"
  fi
done

PYTHONPATH=. "${python_bin}" benchmarks/analyze_drvoi_v12_b0.py \
  --run-dir "${run_dir}" --output-dir "${analysis_dir}" \
  --method-label "V12 B5 behavior-gated smoke" \
  --stage-label "B5-BEHAVIOR-GATED-SMOKE" \
  --extra-method "V12-Frozen B1 (seed85)=runs/drvoi_rhppo_v12/mrat-b1-frozen-seed85" \
  --extra-method "V12 B2 cost-regularized (seed85)=runs/drvoi_rhppo_v12/mrat-b2-cost-regularized-frozen-seed85" \
  --extra-method "V12 B3 shield-only diagnostic=runs/drvoi_rhppo_v12/b3-shield-only-diagnostic-seed85" \
  --extra-method "V12 B4 return-gated smoke=runs/drvoi_rhppo_v12/mrat-b4-return-gated-smoke-seed85" \
  2>&1 | tee -a "${run_dir}/analysis.log"

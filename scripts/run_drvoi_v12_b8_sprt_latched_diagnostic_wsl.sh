#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${WVR_PYTHON_BIN:-/home/hongyun0405/.venvs/bvr1022/bin/python}"
run_dir="${WVR_V12_B8_DIR:-${project_root}/runs/drvoi_rhppo_v12/b8-sprt-latched-diagnostic-seed85}"
analysis_dir="${WVR_V12_B8_ANALYSIS_DIR:-${project_root}/docs/analysis/drvoi-rhppo-v12-b8-sprt-latched-diagnostic}"
checkpoint="${project_root}/runs/drvoi_rhppo_v12/mrat-b6-onpolicy-gated-smoke-seed85/latest.pt"
reach_checkpoint="${project_root}/runs/drvoi_rhppo_v12/reachability-model-a2-seed85/reachability.pt"
return_checkpoint="${project_root}/runs/drvoi_rhppo_v12/return-model-a7-onpolicy-behavior-h60-20k-seed85/reachability.pt"
v9_checkpoint="${project_root}/runs/drvoi_rhppo_v9/coach-online/development-1m-seed85/latest.pt"
response_checkpoint="${project_root}/runs/drvoi_rhppo_v7_1/w3-confirmation/confirmation.pt"
option_q_checkpoint="${project_root}/runs/drvoi_rhppo_v8/macro-option/development/cross-fitted-option-q.pt"
mkdir -p "${run_dir}" "${analysis_dir}"
cd "${project_root}"

for required in "${checkpoint}" "${reach_checkpoint}" "${return_checkpoint}" \
  "${v9_checkpoint}" "${response_checkpoint}" "${option_q_checkpoint}"; do
  if [[ ! -s "${required}" ]]; then
    printf 'missing prerequisite: %s\n' "${required}" >&2
    exit 2
  fi
done

for opponent in hold pursuit evasive; do
  evaluation="${run_dir}/eval-${opponent}.json"
  episode_records="${run_dir}/eval-${opponent}-episodes.jsonl"
  segment_records="${run_dir}/eval-${opponent}-segments.jsonl"
  if [[ ! -s "${evaluation}" || ! -s "${episode_records}" || ! -s "${segment_records}" ]]; then
    PYTHONPATH=. "${python_bin}" benchmarks/evaluate_torch_wvr_mrat_rhppo.py \
      --checkpoint "${checkpoint}" \
      --v9-checkpoint "${v9_checkpoint}" \
      --response-checkpoint "${response_checkpoint}" \
      --option-q-checkpoint "${option_q_checkpoint}" \
      --reachability-checkpoint "${reach_checkpoint}" \
      --router-score-checkpoint "${return_checkpoint}" \
      --routing-mode gate_sprt_latched \
      --gate-sprt-evasive-rate 0.67 --gate-sprt-easy-rate 0.325 \
      --gate-sprt-alpha 0.20 --gate-sprt-beta 0.20 \
      --opponent "${opponent}" --episodes 192 --num-envs 96 --seed 10085 \
      --output "${evaluation}" --episode-records "${episode_records}" \
      --segment-records "${segment_records}" \
      2>&1 | tee -a "${run_dir}/eval-${opponent}.log"
  fi
done

PYTHONPATH=. "${python_bin}" benchmarks/analyze_drvoi_v12_b0.py \
  --run-dir "${run_dir}" --output-dir "${analysis_dir}" \
  --method-label "V12 B8 SPRT-latched diagnostic" \
  --stage-label "B8-SPRT-LATCHED-DIAGNOSTIC" --evaluation-only \
  --extra-method "V12 B7 gate-latched diagnostic=runs/drvoi_rhppo_v12/b7-gate-latched-diagnostic-seed85" \
  --extra-method "V12 B6 on-policy-gated smoke=runs/drvoi_rhppo_v12/mrat-b6-onpolicy-gated-smoke-seed85" \
  --extra-method "V12 B5 behavior-gated smoke=runs/drvoi_rhppo_v12/mrat-b5-behavior-gated-smoke-seed85" \
  --extra-method "V12 B4 return-gated smoke=runs/drvoi_rhppo_v12/mrat-b4-return-gated-smoke-seed85" \
  --extra-method "V12-Frozen B1 (seed85)=runs/drvoi_rhppo_v12/mrat-b1-frozen-seed85" \
  2>&1 | tee -a "${run_dir}/analysis.log"

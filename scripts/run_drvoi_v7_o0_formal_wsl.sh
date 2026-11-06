#!/usr/bin/env bash
set -euo pipefail

cd "/mnt/d/Ph.D/毕业/WVR-1102/WVR-Env"
python_bin="/home/hongyun0405/.venvs/bvr1022/bin/python"
root="runs/drvoi_rhppo_v7/p1-response-formal"

test -f "$root/Q_COMPLETED"
PYTHONPATH=. "$python_bin" benchmarks/validate_drvoi_wvr_o0_v7.py \
    --response-checkpoint "$root/validation.pt" \
    --q-checkpoint "$root/q-validation.pt" \
    --q-gate "$root/q_ready_gate.json" \
    --onpolicy "$root/data/onpolicy-seed6185.pt" \
               "$root/data/onpolicy-seed7185.pt" \
               "$root/data/onpolicy-seed8185.pt" \
    --batch-size 64 --stability-contexts 2000 --seed 11185 \
    --output "$root/o0-validation.json" > "$root/o0-validation.log" 2>&1

PYTHONPATH=. "$python_bin" benchmarks/summarize_drvoi_v7_gates.py \
    --response "$root/validation.json" \
    --q-gate "$root/q_ready_gate.json" --o0-gate "$root/o0_gate.json" \
    --output "$root/authorization.json" > "$root/authorization.log" 2>&1

touch "$root/O0_COMPLETED"

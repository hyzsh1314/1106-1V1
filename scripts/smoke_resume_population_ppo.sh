#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(mktemp -d /tmp/wvr-resume-test-XXXXXX)"
echo "test_dir=${TEST_DIR}"

COMMON_ENV=(
  WVR_RUN_DIR="${TEST_DIR}"
  WVR_NUM_ENVS=16
  WVR_UPDATES=20
  WVR_CHECKPOINT_INTERVAL=1
  WVR_ARCHIVE_INTERVAL=20
  WVR_SNAPSHOT_INTERVAL=2
  WVR_POPULATION_CAPACITY=8
  WVR_MAX_OPPONENTS=4
)

timeout --signal=INT 25s env "${COMMON_ENV[@]}" \
  bash "${PROJECT_ROOT}/scripts/run_wsl_population_ppo_formal.sh"
first_status=$?
echo "first_status=${first_status}"
if [[ ! -f "${TEST_DIR}/latest.pt" ]]; then
  echo "error: interrupted run did not create a checkpoint" >&2
  exit 91
fi

env "${COMMON_ENV[@]}" \
  bash "${PROJECT_ROOT}/scripts/run_wsl_population_ppo_formal.sh"
second_status=$?
echo "second_status=${second_status}"
if [[ ${second_status} -ne 0 || ! -f "${TEST_DIR}/COMPLETED.json" ]]; then
  echo "error: resumed run did not complete" >&2
  exit 92
fi

/home/hongyun0405/.venvs/bvr1022/bin/python - "${TEST_DIR}/COMPLETED.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    progress = json.load(stream)
print(
    "resume_test_status={0} update={1} steps={2}".format(
        progress["status"], progress["update"], progress["total_steps"]
    )
)
PY

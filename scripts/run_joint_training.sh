#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-causal_joint_training/config.yaml}"
GENERATE_SPLIT="${GENERATE_SPLIT:-test}"
GENERATE_LIMIT="${GENERATE_LIMIT:-}"

SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVALUATE="${SKIP_EVALUATE:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@"
}

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  run_step "Validate joint-training inputs" \
    "${PYTHON_BIN}" -m causal_joint_training.preflight --config "${CONFIG_PATH}"
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  run_step "Train causal joint model" \
    "${PYTHON_BIN}" -m causal_joint_training.train --config "${CONFIG_PATH}"
fi

if [[ "${SKIP_EVALUATE}" != "1" ]]; then
  run_step "Evaluate causal joint model" \
    "${PYTHON_BIN}" -m causal_joint_training.evaluate --config "${CONFIG_PATH}"
fi

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  generate_cmd=("${PYTHON_BIN}" -m causal_joint_training.generate \
    --config "${CONFIG_PATH}" --split "${GENERATE_SPLIT}")
  if [[ -n "${GENERATE_LIMIT}" ]]; then
    generate_cmd+=(--limit "${GENERATE_LIMIT}")
  fi
  run_step "Generate explanations" "${generate_cmd[@]}"
fi

echo
echo "Causal joint training finished."

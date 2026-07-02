#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-5}"
ATTRIBUTE_LIMIT="${ATTRIBUTE_LIMIT:-}"
SKIP_ATTRIBUTES="${SKIP_ATTRIBUTES:-0}"
SKIP_CAUSAL_ATTRIBUTES="${SKIP_CAUSAL_ATTRIBUTES:-0}"
SKIP_JOINT_TRAINING="${SKIP_JOINT_TRAINING:-0}"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@"
}

if [[ "${SKIP_ATTRIBUTES}" != "1" ]]; then
  attribute_cmd=(bash attribute_pipeline/run_pipeline.sh all --workers "${WORKERS}" --resume)
  if [[ -n "${ATTRIBUTE_LIMIT}" ]]; then
    attribute_cmd+=(--limit "${ATTRIBUTE_LIMIT}")
  fi
  run_step "Extract item attributes" "${attribute_cmd[@]}"
fi

if [[ "${SKIP_CAUSAL_ATTRIBUTES}" != "1" ]]; then
  run_step "Extract causal attributes with LightGCN, intervention, and OMP" \
    bash scripts/run_causal_attributes.sh
fi

if [[ "${SKIP_JOINT_TRAINING}" != "1" ]]; then
  run_step "Train and evaluate causal joint model" \
    bash scripts/run_joint_training.sh
fi

echo
echo "Pipeline finished."

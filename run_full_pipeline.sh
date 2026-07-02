#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKERS="${WORKERS:-5}"
ATTRIBUTE_LIMIT="${ATTRIBUTE_LIMIT:-}"
CAUSAL_MODE="${CAUSAL_MODE:-omp}"
SKIP_ATTRIBUTES="${SKIP_ATTRIBUTES:-0}"
SKIP_CAUSAL_ATTRIBUTES="${SKIP_CAUSAL_ATTRIBUTES:-0}"
SKIP_JOINT_TRAINING="${SKIP_JOINT_TRAINING:-0}"
RUN_DIRECT_JOINT_TRAINING="${RUN_DIRECT_JOINT_TRAINING:-0}"

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
  causal_stage_name="Extract causal attributes with LightGCN, intervention, and OMP"
  if [[ "${CAUSAL_MODE}" == "direct" ]]; then
    causal_stage_name="Extract causal attributes with direct perturbation"
  fi
  run_step "${causal_stage_name}" \
    bash scripts/run_causal_attributes.sh
fi

if [[ "${SKIP_JOINT_TRAINING}" != "1" ]]; then
  if [[ "${CAUSAL_MODE}" == "direct" && "${RUN_DIRECT_JOINT_TRAINING}" != "1" ]]; then
    echo
    echo "==> Skipping joint training"
    echo "Direct mode writes causal labels without OMP. To train with those labels, run:"
    echo "    CONFIG_PATH=causal_joint_training/config_direct.yaml bash scripts/run_joint_training.sh"
  else
    if [[ "${CAUSAL_MODE}" == "direct" ]]; then
      export CONFIG_PATH="${CONFIG_PATH:-causal_joint_training/config_direct.yaml}"
    fi
    run_step "Train and evaluate causal joint model" \
      bash scripts/run_joint_training.sh
  fi
fi

echo
echo "Pipeline finished."

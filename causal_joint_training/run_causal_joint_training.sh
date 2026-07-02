#!/usr/bin/env bash
set -euo pipefail

# This script lives in causal_joint_training/, but commands must run from repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE="${OVERWRITE:---overwrite}"

# Stage toggles. Set any of these to 1 to skip that stage.
SKIP_REGEN="${SKIP_REGEN:-0}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVALUATE="${SKIP_EVALUATE:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"

# Optional test toggle. Tests are useful before a long run, but disabled by default.
RUN_TESTS="${RUN_TESTS:-0}"

# Extra args for each causal_joint_training command.
PREFLIGHT_ARGS="${PREFLIGHT_ARGS:-}"
TRAIN_ARGS="${TRAIN_ARGS:-}"
EVALUATE_ARGS="${EVALUATE_ARGS:-}"
GENERATE_ARGS="${GENERATE_ARGS:-}"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@"
}

if [[ ! -d "causal_joint_training" ]]; then
  echo "Error: causal_joint_training/ was not found under ${REPO_ROOT}." >&2
  exit 1
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  run_step "Run causal_joint_training tests" \
    "${PYTHON_BIN}" -m unittest discover -s causal_joint_training/tests
fi

if [[ "${SKIP_REGEN}" != "1" ]]; then
  run_step "Regenerate training attribute support" \
    "${PYTHON_BIN}" -m extract_causal_attributes.build_training_attribute_support ${OVERWRITE}

  run_step "Regenerate intervention matrices" \
    "${PYTHON_BIN}" extract_causal_attributes/intervention/build_intervention_matrices.py ${OVERWRITE}

  run_step "Regenerate OMP causal labels" \
    "${PYTHON_BIN}" extract_causal_attributes/intervention/omp/run_omp.py ${OVERWRITE}
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  # shellcheck disable=SC2086
  run_step "Validate causal_joint_training inputs" \
    "${PYTHON_BIN}" -m causal_joint_training.preflight ${PREFLIGHT_ARGS}
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  # shellcheck disable=SC2086
  run_step "Train causal_joint_training model" \
    "${PYTHON_BIN}" -m causal_joint_training.train ${TRAIN_ARGS}
fi

if [[ "${SKIP_EVALUATE}" != "1" ]]; then
  # shellcheck disable=SC2086
  run_step "Evaluate causal_joint_training model" \
    "${PYTHON_BIN}" -m causal_joint_training.evaluate ${EVALUATE_ARGS}
fi

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  # shellcheck disable=SC2086
  run_step "Generate validation/test explanations" \
    "${PYTHON_BIN}" -m causal_joint_training.generate ${GENERATE_ARGS}
fi

echo
echo "Done."

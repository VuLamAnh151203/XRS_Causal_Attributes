#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE="${OVERWRITE:---overwrite}"

SKIP_LIGHTGCN="${SKIP_LIGHTGCN:-0}"
SKIP_SUPPORT="${SKIP_SUPPORT:-0}"
SKIP_INTERVENTION="${SKIP_INTERVENTION:-0}"
SKIP_OMP="${SKIP_OMP:-0}"

SUPPORT_LIMIT="${SUPPORT_LIMIT:-}"
INTERVENTION_LIMIT="${INTERVENTION_LIMIT:-}"
OMP_LIMIT="${OMP_LIMIT:-}"

run_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@"
}

overwrite_args=()
if [[ -n "${OVERWRITE}" ]]; then
  overwrite_args+=("${OVERWRITE}")
fi

if [[ "${SKIP_LIGHTGCN}" != "1" ]]; then
  echo
  echo "==> Pretrain LightGCN and export embeddings"
  (
    cd extract_causal_attributes/lightgcn_cf
    "${PYTHON_BIN}" train.py --config config.yaml
  )
fi

if [[ "${SKIP_SUPPORT}" != "1" ]]; then
  support_cmd=("${PYTHON_BIN}" -m extract_causal_attributes.build_training_attribute_support \
    --config configs/amazon/attribute_support_train.yaml)
  if [[ -n "${SUPPORT_LIMIT}" ]]; then
    support_cmd+=(--limit "${SUPPORT_LIMIT}")
  fi
  support_cmd+=("${overwrite_args[@]}")
  run_step "Build training attribute support" "${support_cmd[@]}"
fi

if [[ "${SKIP_INTERVENTION}" != "1" ]]; then
  intervention_cmd=("${PYTHON_BIN}" extract_causal_attributes/intervention/build_intervention_matrices.py \
    --config configs/amazon/intervention_train.yaml)
  if [[ -n "${INTERVENTION_LIMIT}" ]]; then
    intervention_cmd+=(--limit "${INTERVENTION_LIMIT}")
  fi
  intervention_cmd+=("${overwrite_args[@]}")
  run_step "Build intervention matrices" "${intervention_cmd[@]}"
fi

if [[ "${SKIP_OMP}" != "1" ]]; then
  omp_cmd=("${PYTHON_BIN}" extract_causal_attributes/intervention/omp/run_omp.py \
    --config configs/amazon/omp_train.yaml)
  if [[ -n "${OMP_LIMIT}" ]]; then
    omp_cmd+=(--limit "${OMP_LIMIT}")
  fi
  omp_cmd+=("${overwrite_args[@]}")
  run_step "Recover causal attributes with OMP" "${omp_cmd[@]}"
fi

echo
echo "Causal attribute extraction finished."

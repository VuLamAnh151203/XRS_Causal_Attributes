#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE="${OVERWRITE:---overwrite}"

SKIP_LIGHTGCN="${SKIP_LIGHTGCN:-0}"
SKIP_SUPPORT="${SKIP_SUPPORT:-0}"
SKIP_DIRECT_SCORING="${SKIP_DIRECT_SCORING:-0}"
SKIP_DIRECT_SELECTION="${SKIP_DIRECT_SELECTION:-0}"

SUPPORT_LIMIT="${SUPPORT_LIMIT:-}"
DIRECT_TOP_K="${DIRECT_TOP_K:-5}"
DIRECT_MIN_SCORE_DROP="${DIRECT_MIN_SCORE_DROP:-0.0}"
DIRECT_PROPAGATION_MODE="${DIRECT_PROPAGATION_MODE:-local-score}"

LIGHTGCN_CONFIG="${LIGHTGCN_CONFIG:-extract_causal_attributes/lightgcn_cf/config.yaml}"
SUPPORT_CONFIG="${SUPPORT_CONFIG:-configs/amazon/attribute_support_train.yaml}"
SUPPORT_JSONL="${SUPPORT_JSONL:-extract_causal_attributes/artifacts/amazon/trn_attribute_support.jsonl}"
DIRECT_SCORE_OUTPUT="${DIRECT_SCORE_OUTPUT:-extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_attribute_drop_effects.json}"
DIRECT_SELECTION_OUTPUT="${DIRECT_SELECTION_OUTPUT:-extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_causal_attributes.jsonl}"
DIRECT_SUMMARY_OUTPUT="${DIRECT_SUMMARY_OUTPUT:-extract_causal_attributes/direct_perturbation/artifacts/amazon/summary.json}"
DIRECT_COMPAT_DIR="${DIRECT_COMPAT_DIR:-extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_omp_compatible}"
DIRECT_VOCABULARY="${DIRECT_VOCABULARY:-attribute_pipeline/outputs/amazon/vocabulary.json}"
DIRECT_ID_MAPPINGS="${DIRECT_ID_MAPPINGS:-extract_causal_attributes/lightgcn_cf/artifacts/amazon/id_mappings.json}"

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
  run_step "Pretrain LightGCN and export embeddings" \
    "${PYTHON_BIN}" extract_causal_attributes/lightgcn_cf/train.py --config "${LIGHTGCN_CONFIG}"
fi

if [[ "${SKIP_SUPPORT}" != "1" ]]; then
  support_cmd=("${PYTHON_BIN}" -m extract_causal_attributes.build_training_attribute_support \
    --config "${SUPPORT_CONFIG}")
  if [[ -n "${SUPPORT_LIMIT}" ]]; then
    support_cmd+=(--limit "${SUPPORT_LIMIT}")
  fi
  support_cmd+=("${overwrite_args[@]}")
  run_step "Build training attribute support" "${support_cmd[@]}"
fi

if [[ "${SKIP_DIRECT_SCORING}" != "1" ]]; then
  score_cmd=("${PYTHON_BIN}" extract_causal_attributes/lightgcn_cf/score_item_only_attribute_support_jsonl.py \
    --config "${LIGHTGCN_CONFIG}" \
    --support-jsonl "${SUPPORT_JSONL}" \
    --output "${DIRECT_SCORE_OUTPUT}" \
    --propagation-mode "${DIRECT_PROPAGATION_MODE}")
  if [[ -n "${OVERWRITE}" ]]; then
    score_cmd+=(--no-resume)
  fi
  run_step "Score direct attribute perturbations" "${score_cmd[@]}"
fi

if [[ "${SKIP_DIRECT_SELECTION}" != "1" ]]; then
  selection_cmd=("${PYTHON_BIN}" extract_causal_attributes/direct_perturbation/select_direct_causal_attributes.py \
    --scores "${DIRECT_SCORE_OUTPUT}" \
    --output "${DIRECT_SELECTION_OUTPUT}" \
    --summary-output "${DIRECT_SUMMARY_OUTPUT}" \
    --omp-compatible-output-dir "${DIRECT_COMPAT_DIR}" \
    --vocabulary "${DIRECT_VOCABULARY}" \
    --id-mappings "${DIRECT_ID_MAPPINGS}" \
    --top-k "${DIRECT_TOP_K}" \
    --min-score-drop "${DIRECT_MIN_SCORE_DROP}")
  selection_cmd+=("${overwrite_args[@]}")
  run_step "Select direct causal attributes" "${selection_cmd[@]}"
fi

echo
echo "Direct causal attribute extraction finished."

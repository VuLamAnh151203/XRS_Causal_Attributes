#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_PATH="${INPUT_PATH:-${REPO_ROOT}/XRec/data/amazon/item_profile.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/amazon}"

usage() {
  cat <<'EOF'
Usage:
  bash attribute_pipeline/run_pipeline.sh extract [--workers N] [--resume] [--allow-partial]
  bash attribute_pipeline/run_pipeline.sh embed [--batch-size N]
  bash attribute_pipeline/run_pipeline.sh cluster [--threshold FLOAT]
  bash attribute_pipeline/run_pipeline.sh implicit [--top-k N] [--batch-size N]
  bash attribute_pipeline/run_pipeline.sh all [--workers N] [--resume] [--allow-partial]

Environment overrides:
  INPUT_PATH   Source item_profile.json JSONL file
  OUTPUT_DIR   Artifact directory
  PYTHON_BIN   Python executable, for example python3

Extraction options:
  --workers N     Maximum concurrent DeepSeek API requests
  --batch-size N  Alias for --workers N during extraction
  --limit N       Restrict extraction to the first N source items for a pilot
EOF
}

run_python_module() {
  (
    cd -- "${SCRIPT_DIR}"
    "${PYTHON_BIN}" -m item_attribute_pipeline "$@"
  )
}

require_file() {
  local path="$1"
  local previous_stage="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: Missing prerequisite artifact: ${path}" >&2
    echo "Run the '${previous_stage}' stage first." >&2
    exit 2
  fi
}

run_extract() {
  echo "==> Steps 1-3: extract raw attributes and normalize them"
  run_python_module extract-normalize \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_DIR}" \
    "$@"
}

run_embed() {
  require_file "${OUTPUT_DIR}/normalized_item_attributes.jsonl" "extract"
  require_file "${OUTPUT_DIR}/attribute_frequencies.json" "extract"
  echo "==> Step 4: embed normalized attributes"
  run_python_module embed \
    --output "${OUTPUT_DIR}" \
    "$@"
}

run_cluster() {
  require_file "${OUTPUT_DIR}/normalized_item_attributes.jsonl" "extract"
  require_file "${OUTPUT_DIR}/attribute_embeddings.npz" "embed"
  echo "==> Step 5: cluster attributes and build canonical mappings"
  run_python_module cluster \
    --output "${OUTPUT_DIR}" \
    "$@"
}

run_implicit() {
  require_file "${OUTPUT_DIR}/item_attributes.json" "cluster"
  require_file "${OUTPUT_DIR}/vocabulary.json" "cluster"
  require_file "${OUTPUT_DIR}/attribute_embeddings.npz" "embed"
  require_file "${OUTPUT_DIR}/embedding_metadata.json" "embed"
  echo "==> Implicit attributes: rank non-explicit canonical attributes for each item"
  run_python_module implicit \
    --input "${INPUT_PATH}" \
    --output "${OUTPUT_DIR}" \
    "$@"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

stage="$1"
shift

case "${stage}" in
  extract)
    run_extract "$@"
    ;;
  embed)
    run_embed "$@"
    ;;
  cluster)
    run_cluster "$@"
    ;;
  implicit)
    run_implicit "$@"
    ;;
  all)
    run_extract "$@"
    run_embed
    run_cluster
    run_implicit
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "ERROR: Unknown stage '${stage}'." >&2
    usage >&2
    exit 2
    ;;
esac

echo "==> Artifacts: ${OUTPUT_DIR}"

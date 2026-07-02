from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_pipeline.sh"


def bash_path(path: Path) -> str:
    return path.resolve().as_posix()


def make_fake_python(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
echo "$*" >> "${LOG_PATH}"
stage="$3"
output=""
previous=""
for argument in "$@"; do
  if [[ "${previous}" == "--output" ]]; then
    output="${argument}"
  fi
  previous="${argument}"
done
mkdir -p "${output}"
case "${stage}" in
  extract-normalize)
    touch "${output}/raw_item_attributes.jsonl"
    touch "${output}/normalized_item_attributes.jsonl"
    printf '{}\\n' > "${output}/attribute_frequencies.json"
    touch "${output}/issues.jsonl"
    ;;
  embed)
    touch "${output}/attribute_embeddings.npz"
    printf '{}\\n' > "${output}/embedding_metadata.json"
    ;;
  cluster)
    touch "${output}/clusters.json"
    touch "${output}/vocabulary.json"
    touch "${output}/item_attribute_ids.json"
    touch "${output}/item_attributes.json"
    touch "${output}/item_attribute_matrix.npz"
    touch "${output}/matrix_rows.json"
    touch "${output}/matrix_columns.json"
    ;;
  implicit)
    touch "${output}/item_attributes_im_ex.json"
    ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_all_runs_stages_in_order_outside_repository(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake_python.sh"
    output = tmp_path / "artifacts"
    log = tmp_path / "calls.log"
    make_fake_python(fake_python)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": bash_path(fake_python),
            "OUTPUT_DIR": bash_path(output),
            "LOG_PATH": bash_path(log),
        }
    )

    result = subprocess.run(
        ["bash", bash_path(RUNNER), "all", "--batch-size", "10", "--resume"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    assert "extract-normalize" in calls[0]
    assert "--batch-size 10 --resume" in calls[0]
    assert " embed " in f" {calls[1]} "
    assert " cluster " in f" {calls[2]} "
    assert " implicit " in f" {calls[3]} "
    assert "--batch-size" not in calls[1]
    assert "--batch-size" not in calls[2]


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not installed")
def test_embed_reports_missing_extract_artifacts(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["OUTPUT_DIR"] = bash_path(tmp_path / "empty")

    result = subprocess.run(
        ["bash", bash_path(RUNNER), "embed"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Run the 'extract' stage first." in result.stderr

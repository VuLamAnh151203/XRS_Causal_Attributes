# Explainable RS Based on Causal Attributes

This repository packages the full explainable recommendation pipeline around
causal item attributes.

The main flow is:

```text
item profiles
-> extract explicit and implicit item attributes
-> train LightGCN
-> build attribute support for each user-item pair
-> build intervention matrices A and score deltas y_delta
-> recover sparse causal attribute labels with OMP
-> train the causal joint recommendation/explanation model
```

The code is copied from three standalone components:

- `attribute_pipeline/`
- `extract_causal_attributes/`
- `causal_joint_training/`

Step 1 attribute outputs are bundled in `attribute_pipeline/outputs/amazon/`,
so the package can start from the causal-attribute stage. Later generated
causal artifacts, checkpoints, caches, local data, `.env`, and model weights are
still intentionally excluded.

## Repository Layout

```text
.
|-- attribute_pipeline/              # Extract and canonicalize item attributes
|-- extract_causal_attributes/        # LightGCN, support, intervention, OMP
|-- causal_joint_training/            # Final model trained with causal labels
|-- configs/amazon/                   # Train-mode configs for the main pipeline
|-- scripts/                          # Stage-level Bash runners
|-- run_full_pipeline.sh              # Full pipeline runner
|-- requirements.txt                  # Combined dependency file
`-- .gitignore                        # Keeps generated artifacts out of Git
```

## Required Inputs

Place the source dataset under `XRec/data/amazon/` at the repository root.
The default configs expect these files:

```text
XRec/data/amazon/item_profile.json
XRec/data/amazon/user_profile.json
XRec/data/amazon/total_trn_new.csv
XRec/data/amazon/total_val_new.csv
XRec/data/amazon/total_tst_new.csv
XRec/data/amazon/trn.pkl
XRec/data/amazon/val.pkl
XRec/data/amazon/tst.pkl
```

The attribute extraction stage also needs a DeepSeek API key in
`attribute_pipeline/.env`:

```bash
cp attribute_pipeline/.env.example attribute_pipeline/.env
# edit attribute_pipeline/.env and set DEEPSEEK_API_KEY
```

The final text generation stage expects the local LLM path configured in
`causal_joint_training/config.yaml`, currently `./Llama-2-7b-chat-hf`.

## Setup

Install all Python dependencies from the aggregate root:

```bash
python -m pip install -r requirements.txt
```

Some stages use optional GPU acceleration. The train configs in
`configs/amazon/` use `device: auto` so the code can fall back to CPU when CUDA
is not available.

The orchestration scripts are Bash scripts. On Windows, run them from Git Bash
or WSL. If Bash is not available, use the equivalent manual Python commands
shown below.

## Run The Full Pipeline

```bash
bash run_full_pipeline.sh
```

Useful environment overrides:

```bash
PYTHON_BIN=python3 bash run_full_pipeline.sh
ATTRIBUTE_LIMIT=10 bash run_full_pipeline.sh
SKIP_ATTRIBUTES=1 bash run_full_pipeline.sh
SKIP_JOINT_TRAINING=1 bash run_full_pipeline.sh
```

Because Step 1 is already included, the usual next command is:

```bash
SKIP_ATTRIBUTES=1 bash run_full_pipeline.sh
```

For development smoke runs, combine limits:

```bash
ATTRIBUTE_LIMIT=10 SUPPORT_LIMIT=10 INTERVENTION_LIMIT=10 OMP_LIMIT=10 \
  bash run_full_pipeline.sh
```

## Stage 1: Extract Item Attributes

This aggregate folder already includes the current Stage 1 outputs copied from
the original `attribute_pipeline/outputs/amazon/`. Re-run this stage only when
you want to rebuild the attribute vocabulary or item-attribute files.

Run:

```bash
bash attribute_pipeline/run_pipeline.sh all --workers 5 --resume
```

Outputs:

```text
attribute_pipeline/outputs/amazon/
|-- raw_item_attributes.jsonl
|-- normalized_item_attributes.jsonl
|-- attribute_frequencies.json
|-- attribute_embeddings.npz
|-- embedding_metadata.json
|-- clusters.json
|-- vocabulary.json
|-- item_attribute_ids.json
|-- item_attributes.json
|-- item_attribute_matrix.npz
|-- matrix_rows.json
|-- matrix_columns.json
`-- item_attributes_im_ex.json
```

The main downstream files are:

- `vocabulary.json`: canonical global attribute vocabulary.
- `item_attribute_ids.json`: item to canonical attribute IDs.
- `item_attributes_im_ex.json`: explicit plus implicit item attributes.
- `attribute_embeddings.npz`: semantic attribute embeddings.

## Stage 2: Extract Causal Attributes

Run the full causal-attribute stage:

```bash
bash scripts/run_causal_attributes.sh
```

Equivalent manual commands:

```bash
cd extract_causal_attributes/lightgcn_cf
python train.py --config config.yaml
cd ../..

python -m extract_causal_attributes.build_training_attribute_support \
  --config configs/amazon/attribute_support_train.yaml --overwrite

python extract_causal_attributes/intervention/build_intervention_matrices.py \
  --config configs/amazon/intervention_train.yaml --overwrite

python extract_causal_attributes/intervention/omp/run_omp.py \
  --config configs/amazon/omp_train.yaml --overwrite
```

Important outputs:

```text
extract_causal_attributes/lightgcn_cf/artifacts/amazon/
|-- id_mappings.json
|-- user_history.json
|-- user_ego_embeddings.pt
|-- item_ego_embeddings.pt
|-- user_baseline_embeddings.pt
|-- item_baseline_embeddings.pt
`-- best_checkpoint.pt

extract_causal_attributes/artifacts/amazon/
|-- trn_attribute_support.jsonl
`-- trn_attribute_support.summary.json

extract_causal_attributes/intervention/artifacts/amazon/
|-- manifest.jsonl
|-- summary.json
|-- run_config.json
|-- vocabulary.json
`-- shards/interventions_*.npz

extract_causal_attributes/intervention/omp/artifacts/amazon/
|-- manifest.jsonl
|-- summary.json
|-- run_config.json
|-- vocabulary.json
`-- shards/omp_vectors_*.npz
```

Use the configs under `configs/amazon/` for the main training pipeline. Some
copied default configs are useful for test/report runs and may point at `tst`
artifacts.

## What Is A Causal Attribute Label?

For each user-item explanation pair, the support builder finds target
attributes that are semantically supported by items in the user's history.

The intervention builder samples many perturbations of those support items.
Each intervention row has:

- `A`: a sparse binary row over the canonical attribute vocabulary.
- `y_delta`: the LightGCN score drop, computed as baseline score minus
  perturbed score.

OMP then solves for a sparse attribute coefficient vector per pair. A recovered
pair is stored in:

- `extract_causal_attributes/intervention/omp/artifacts/amazon/manifest.jsonl`
- `extract_causal_attributes/intervention/omp/artifacts/amazon/shards/omp_vectors_*.npz`

The manifest identifies `pair_index`, `user_id`, `target_item_id`,
`vector_shard`, and `vector_row`. The shard stores sparse `coef_indices` and
`coef_data`; map `coef_indices` through `vocabulary.json` to get the causal
attribute names and weights. `causal_joint_training` loads only recovered OMP
rows that pass the configured residual threshold.

## Stage 3: Causal Joint Training

Run:

```bash
bash scripts/run_joint_training.sh
```

Equivalent manual commands:

```bash
python -m causal_joint_training.preflight
python -m causal_joint_training.train
python -m causal_joint_training.evaluate
python -m causal_joint_training.generate --split test
```

Outputs:

```text
causal_joint_training/artifacts/amazon/
|-- run_config.json
|-- training_history.json
|-- latest.pt
|-- best_ndcg20.pt
|-- best_generation_loss.pt
|-- best_causal_recall5.pt
`-- generated_explanations.jsonl
```

## Optional Legacy Branch: Item-Only Direct Perturbation

`extract_causal_attributes/lightgcn_cf` also contains an item-only experimental
branch:

```text
train item-only LightGCN
-> save item-only embeddings
-> directly run attribute/support perturbation
-> compare score, rank, and metric changes
```

This branch is useful for direct perturbation experiments, but it is not the
main path used by intervention, OMP, and causal joint training.

## Tests

Run from the aggregate root:

```bash
python -m pip install -r requirements.txt
python -m pytest attribute_pipeline/tests -q
python -m unittest discover -s extract_causal_attributes/tests
python -m unittest discover -s extract_causal_attributes/lightgcn_cf/tests -v
python -m unittest discover -s extract_causal_attributes/intervention/tests
python -m unittest discover -s extract_causal_attributes/intervention/omp/tests
python -m unittest discover -s causal_joint_training/tests
```

These tests validate source logic. Full pipeline execution still requires the
external dataset, API key, embedding model, and local LLM files.

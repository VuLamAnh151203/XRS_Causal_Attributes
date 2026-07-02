# Extract Causal Attributes

This folder builds causal attribute labels for each user-item explanation pair.
It connects item attributes, LightGCN recommendation scores, intervention
matrices, and OMP sparse recovery.

It also contains an optional direct perturbation mode that chooses causal
attributes from direct score drops without building intervention matrices or
running OMP.

## Main Flow

```text
attribute_pipeline outputs
LightGCN artifacts
XRec explanation pairs
-> build training attribute support
-> build intervention matrices A and y_delta
-> run OMP
-> causal attribute labels per pair
```

## Prerequisites

Run the attribute pipeline first:

```bash
bash attribute_pipeline/run_pipeline.sh all --workers 5 --resume
```

Then pretrain LightGCN:

```bash
cd extract_causal_attributes/lightgcn_cf
python train.py --config config.yaml
cd ../..
```

LightGCN writes mappings, histories, and embeddings to:

```text
extract_causal_attributes/lightgcn_cf/artifacts/amazon/
```

## Build Training Attribute Support

Use the train-mode aggregate config:

```bash
python -m extract_causal_attributes.build_training_attribute_support \
  --config configs/amazon/attribute_support_train.yaml --overwrite
```

Outputs:

```text
extract_causal_attributes/artifacts/amazon/trn_attribute_support.jsonl
extract_causal_attributes/artifacts/amazon/trn_attribute_support.summary.json
```

Each support record contains a user-item pair, target item attributes, and
history items that semantically support each target attribute.

## Build Intervention Matrices

```bash
python extract_causal_attributes/intervention/build_intervention_matrices.py \
  --config configs/amazon/intervention_train.yaml --overwrite
```

Outputs:

```text
extract_causal_attributes/intervention/artifacts/amazon/
|-- manifest.jsonl
|-- summary.json
|-- run_config.json
|-- vocabulary.json
`-- shards/interventions_*.npz
```

Each intervention shard stores sparse matrix rows `A` and score changes
`y_delta` for sampled support-item removals.

## Run OMP

```bash
python extract_causal_attributes/intervention/omp/run_omp.py \
  --config configs/amazon/omp_train.yaml --overwrite
```

Outputs:

```text
extract_causal_attributes/intervention/omp/artifacts/amazon/
|-- manifest.jsonl
|-- summary.json
|-- run_config.json
|-- vocabulary.json
`-- shards/omp_vectors_*.npz
```

The OMP output is the causal attribute label source for
`causal_joint_training`.

## Direct Perturbation Mode Without OMP

Direct mode runs:

```text
training attribute support
LightGCN artifacts
-> remove support items for each candidate attribute
-> measure score_drop
-> choose top positive score_drop attributes
```

Run from the aggregate root:

```bash
bash scripts/run_direct_causal_attributes.sh
```

Or through the shared causal runner:

```bash
CAUSAL_MODE=direct bash scripts/run_causal_attributes.sh
```

Outputs:

```text
extract_causal_attributes/direct_perturbation/artifacts/amazon/
|-- direct_attribute_drop_effects.json
|-- direct_causal_attributes.jsonl
|-- summary.json
`-- direct_omp_compatible/
    |-- manifest.jsonl
    |-- summary.json
    |-- run_config.json
    |-- vocabulary.json
    `-- shards/direct_vectors_000000.npz
```

`direct_causal_attributes.jsonl` is the direct causal-attribute report. The
`direct_omp_compatible/` directory can be used by the joint trainer with
`causal_joint_training/config_direct.yaml`.

## Full Stage Runner

From the aggregate root:

```bash
bash scripts/run_causal_attributes.sh
```

Useful overrides:

```bash
SUPPORT_LIMIT=20 INTERVENTION_LIMIT=20 OMP_LIMIT=20 bash scripts/run_causal_attributes.sh
SKIP_LIGHTGCN=1 bash scripts/run_causal_attributes.sh
OVERWRITE= bash scripts/run_causal_attributes.sh
```

## Notes On Configs

Use `configs/amazon/*_train.yaml` for the main training pipeline. The copied
default configs inside this folder may point at test artifacts for report or
diagnostic runs.

## Tests

```bash
python -m unittest discover -s extract_causal_attributes/tests
python -m unittest discover -s extract_causal_attributes/intervention/tests
python -m unittest discover -s extract_causal_attributes/intervention/omp/tests
```

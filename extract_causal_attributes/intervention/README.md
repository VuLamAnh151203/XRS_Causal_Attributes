# Intervention And OMP

This folder turns attribute-support records into intervention matrices, then
uses OMP to recover sparse causal attribute vectors.

## Position In The Pipeline

```text
training attribute support
LightGCN artifacts
canonical vocabulary
-> build intervention rows A and score deltas y_delta
-> run OMP
-> causal attributes per pair
```

## Build Intervention Matrices

Run from the aggregate root:

```bash
python extract_causal_attributes/intervention/build_intervention_matrices.py \
  --config configs/amazon/intervention_train.yaml --overwrite
```

Inputs:

```text
extract_causal_attributes/artifacts/amazon/trn_attribute_support.jsonl
attribute_pipeline/outputs/amazon/vocabulary.json
extract_causal_attributes/lightgcn_cf/artifacts/amazon/user_history.json
extract_causal_attributes/lightgcn_cf/artifacts/amazon/id_mappings.json
extract_causal_attributes/lightgcn_cf/artifacts/amazon/user_ego_embeddings.pt
extract_causal_attributes/lightgcn_cf/artifacts/amazon/item_ego_embeddings.pt
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

The manifest stores one record per pair and points to the shard slice for that
pair. Each `.npz` shard contains:

```text
A_data, A_indices, A_indptr, A_shape
y_delta
y_h
pair_index
intervention_index
removed_item_ids
removed_item_indptr
removed_item_count
```

`A` is a sparse binary matrix over the canonical attribute vocabulary.
`y_delta` is the LightGCN score drop for the intervention.

## Run OMP

```bash
python extract_causal_attributes/intervention/omp/run_omp.py \
  --config configs/amazon/omp_train.yaml --overwrite
```

Inputs:

```text
extract_causal_attributes/intervention/artifacts/amazon/manifest.jsonl
extract_causal_attributes/intervention/artifacts/amazon/shards/interventions_*.npz
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

Recovered OMP rows contain sparse coefficient vectors. The manifest stores
`pair_index`, raw and internal user/item IDs, `status`, `vector_shard`,
`vector_row`, and diagnostics such as `relative_residual`.

Coefficient shards contain:

```text
coef_data
coef_indices
coef_indptr
coef_shape
pair_index
user_index
target_item_index
```

Map `coef_indices` through `vocabulary.json` to get causal attribute names.

## Pair Report

`build_pair_report.py` reads intervention and OMP artifacts to produce
diagnostic reports for individual pairs. Use:

```bash
python extract_causal_attributes/intervention/build_pair_report.py --help
```

## Tests

```bash
python -m unittest discover -s extract_causal_attributes/intervention/tests
python -m unittest discover -s extract_causal_attributes/intervention/omp/tests
```

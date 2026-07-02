# LightGCN CF Module

This folder trains LightGCN and exports the recommendation artifacts needed by
the causal intervention pipeline.

## Main Pipeline Role

```text
XRec recommendation interactions
-> train LightGCN
-> export id mappings, user history, ego embeddings, baseline embeddings
-> intervention scoring uses these artifacts
```

The main causal path uses standard LightGCN:

```bash
cd extract_causal_attributes/lightgcn_cf
python train.py --config config.yaml
```

Default outputs:

```text
extract_causal_attributes/lightgcn_cf/artifacts/amazon/
|-- best_checkpoint.pt
|-- data_summary.json
|-- id_mappings.json
|-- train_pairs.pt
|-- training_history.json
|-- user_history.json
|-- user_ego_embeddings.pt
|-- item_ego_embeddings.pt
|-- user_baseline_embeddings.pt
|-- item_baseline_embeddings.pt
`-- validation_metrics.json
```

`user_ego_embeddings.pt` and `item_ego_embeddings.pt` are graph-independent
learned embeddings. `user_baseline_embeddings.pt` and
`item_baseline_embeddings.pt` are the propagated baseline embeddings used for
baseline scores.

## Evaluate

```bash
cd extract_causal_attributes/lightgcn_cf
python evaluate.py --config config.yaml --split val
```

Validation ranks items outside each user's training history and reports recall
and nDCG metrics.

## Direct Edge Perturbation

Drop one or more history edges for a user:

```bash
cd extract_causal_attributes/lightgcn_cf
python perturb.py --config config.yaml \
  --user-id USER_ID \
  --drop-item-id ITEM_A \
  --drop-item-id ITEM_B \
  --top-k 20
```

This recomputes graph propagation after removing selected user-item edges.

## Optional Item-Only Branch

The item-only path is an earlier direct-perturbation branch:

```text
train item-only LightGCN
-> save item-only embeddings
-> directly run attribute/support perturbation
-> compare score, rank, and metric changes
```

Run item-only training:

```bash
cd extract_causal_attributes/lightgcn_cf
python train_item_only.py --config config_item_only.yaml
```

Evaluate item-only:

```bash
python evaluate_item_only.py --config config_item_only.yaml --split val
```

Direct item-only support scoring scripts:

```bash
python score_item_only_attribute_support_jsonl.py --help
python evaluate_item_only_attribute_perturbation.py --help
```

This branch is not used by the main intervention + OMP + joint-training path.

## Direct Causal Selection Without OMP

The aggregate project also exposes a direct mode that uses this folder's
attribute-support perturbation scorer with standard LightGCN artifacts:

```bash
bash scripts/run_direct_causal_attributes.sh
```

It scores each target attribute by dropping its support items and measuring
`score_drop`, then selects the top positive attributes without OMP.

## Tests

```bash
python -m unittest discover -s extract_causal_attributes/lightgcn_cf/tests -v
```

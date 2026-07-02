# Causal Joint Training

This folder trains the final model using recommendation data, item attributes,
LightGCN embeddings, and OMP causal attribute labels.

It is intentionally standalone: it reads artifacts produced by earlier stages
but does not import their training or intervention code.

## Inputs

Configured in `causal_joint_training/config.yaml`:

```text
extract_causal_attributes/lightgcn_cf/artifacts/amazon/id_mappings.json
extract_causal_attributes/lightgcn_cf/artifacts/amazon/user_baseline_embeddings.pt
extract_causal_attributes/lightgcn_cf/artifacts/amazon/item_baseline_embeddings.pt
attribute_pipeline/outputs/amazon/vocabulary.json
attribute_pipeline/outputs/amazon/item_attribute_ids.json
attribute_pipeline/outputs/amazon/attribute_embeddings.npz
extract_causal_attributes/intervention/omp/artifacts/amazon/
XRec/data/amazon/total_trn_new.csv
XRec/data/amazon/total_val_new.csv
XRec/data/amazon/total_tst_new.csv
XRec/data/amazon/trn.pkl
XRec/data/amazon/val.pkl
XRec/data/amazon/tst.pkl
XRec/data/amazon/user_profile.json
XRec/data/amazon/item_profile.json
```

The trainer uses `trn.pkl` explanation rows that have accepted OMP causal
labels. `total_trn_new.csv` is used as recommendation background history.

For direct perturbation labels without OMP, use
`causal_joint_training/config_direct.yaml`. It points `paths.omp_dir` at the
direct OMP-compatible artifact directory:

```text
extract_causal_attributes/direct_perturbation/artifacts/amazon/direct_omp_compatible
```

## Preflight

Validate inputs before a long training run:

```bash
python -m causal_joint_training.preflight
```

This checks artifact schemas, ID alignment, OMP vocabulary compatibility, and
how many recovered causal labels pass the residual threshold.

## Train

```bash
python -m causal_joint_training.train
```

Outputs:

```text
causal_joint_training/artifacts/amazon/
|-- run_config.json
|-- training_history.json
|-- latest.pt
|-- best_ndcg20.pt
|-- best_generation_loss.pt
`-- best_causal_recall5.pt
```

## Evaluate

```bash
python -m causal_joint_training.evaluate
```

The evaluation uses the saved checkpoint from
`causal_joint_training/artifacts/amazon/latest.pt` unless `--checkpoint` is
provided.

## Generate Explanations

```bash
python -m causal_joint_training.generate --split test
```

Output:

```text
causal_joint_training/artifacts/amazon/generated_explanations.jsonl
```

The default generation config expects a local LLM at `./Llama-2-7b-chat-hf`.
Install `bitsandbytes` for 4-bit loading, or set
`generation.load_in_4bit: false` in `config.yaml`.

## Stage Runner

From the aggregate root:

```bash
bash scripts/run_joint_training.sh
```

Useful overrides:

```bash
GENERATE_SPLIT=validation bash scripts/run_joint_training.sh
GENERATE_LIMIT=20 bash scripts/run_joint_training.sh
SKIP_TRAIN=1 bash scripts/run_joint_training.sh
```

Train with direct perturbation labels:

```bash
CONFIG_PATH=causal_joint_training/config_direct.yaml bash scripts/run_joint_training.sh
```

The copied `run_causal_joint_training.sh` also exists, but it regenerates
causal artifacts with the copied default configs. For the main aggregate
pipeline, prefer the top-level scripts and `configs/amazon/*_train.yaml`.

## Tests

```bash
python -m unittest discover -s causal_joint_training/tests
```

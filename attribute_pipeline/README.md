# Attribute Pipeline

This folder extracts item attributes and converts them into a canonical
vocabulary used by the causal stages.

The current `outputs/amazon/` artifacts are bundled in this aggregate project,
so Step 1 has already been run. Re-run this folder only when you want to rebuild
attributes from `item_profile.json`.

## Role In The Full Pipeline

```text
item_profile.json
-> raw extracted attributes
-> normalized attributes
-> semantic embeddings
-> clustered canonical vocabulary
-> explicit plus implicit item attributes
```

Downstream stages use:

- `outputs/amazon/vocabulary.json`
- `outputs/amazon/item_attribute_ids.json`
- `outputs/amazon/item_attributes_im_ex.json`
- `outputs/amazon/attribute_embeddings.npz`

## Setup

Install dependencies from the aggregate root:

```bash
python -m pip install -r requirements.txt
```

Create the local API key file:

```bash
cp attribute_pipeline/.env.example attribute_pipeline/.env
# set DEEPSEEK_API_KEY in attribute_pipeline/.env
```

The default input path is:

```text
XRec/data/amazon/item_profile.json
```

Override it when needed:

```bash
INPUT_PATH=/path/to/item_profile.json bash attribute_pipeline/run_pipeline.sh all
```

## Run

Pilot run:

```bash
bash attribute_pipeline/run_pipeline.sh all --workers 5 --limit 10 --resume
```

Full run:

```bash
bash attribute_pipeline/run_pipeline.sh all --workers 5 --resume
```

Individual stages:

```bash
bash attribute_pipeline/run_pipeline.sh extract --workers 5 --resume
bash attribute_pipeline/run_pipeline.sh embed
bash attribute_pipeline/run_pipeline.sh cluster
bash attribute_pipeline/run_pipeline.sh implicit --top-k 10
```

## Outputs

All default outputs are written to:

```text
attribute_pipeline/outputs/amazon/
```

Important files:

```text
raw_item_attributes.jsonl          # raw LLM extraction output
normalized_item_attributes.jsonl   # normalized attribute phrases per item
attribute_frequencies.json         # phrase counts
attribute_embeddings.npz           # semantic vectors for phrases
embedding_metadata.json            # embedding phrase order and metadata
clusters.json                      # cluster assignments
vocabulary.json                    # canonical attribute ID to phrase
item_attribute_ids.json            # item ID to canonical attribute IDs
item_attributes.json               # explicit canonical item attributes
item_attribute_matrix.npz          # item-attribute sparse matrix
item_attributes_im_ex.json         # explicit plus implicit attributes
```

`item_attributes_im_ex.json` is the main input for building training attribute
support in `extract_causal_attributes`.

## Tests

```bash
python -m pip install -r requirements.txt
python -m pytest attribute_pipeline/tests -q
```

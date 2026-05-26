# Dataset Validation

Run basic validation before training or comparing Korean AES models.

## Structural Checks

- Confirm every row has essay text and score labels.
- Verify label ranges and score scaling.
- Check train, validation, and test split sizes.
- Count empty essays and missing labels.

## Leakage Checks

- Check duplicate essay ids across splits.
- Compare prompt or topic distributions across splits.
- Confirm generated embeddings align with the intended split.

## Record

Save row counts, filtering rules, split seed, and preprocessing script revision with the experiment output.

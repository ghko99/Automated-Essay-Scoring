# Model Artifact Notes

Automated essay scoring runs can create large artifacts. Keep artifact handling consistent across experiments.

## Generated Artifacts

- Model checkpoints.
- Generated embeddings or NumPy arrays.
- Prediction CSV files.
- Training logs and tensorboard outputs.
- Temporary spreadsheet exports.

## Storage

Store large artifacts outside Git or through Git LFS when intentional. Keep each run folder self-contained with command line, config snapshot, checkpoint id, prediction output, and metric summary.

## Promotion

Before promoting a model, verify the tokenizer, preprocessing settings, label scale, and evaluation script match the reported result.

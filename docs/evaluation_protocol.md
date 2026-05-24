# Evaluation Protocol

Use a stable evaluation protocol for Korean automated essay scoring comparisons.

## Required Metadata

- Git commit and branch.
- Dataset split and preprocessing revision.
- Model architecture and checkpoint path.
- Random seed and hardware.
- Metric script and package versions.

## Metrics

Report agreement and error metrics together when possible: QWK, correlation, MAE, and RMSE. If multiple rubrics are predicted, include per-rubric metrics as well as the aggregate.

## Audit

Before committing a result summary, inspect examples with the largest absolute errors and confirm the prediction file matches the intended test split.

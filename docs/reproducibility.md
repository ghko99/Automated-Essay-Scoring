# Reproducibility Notes

This repository separates embedding extraction and GRU training for Korean AES experiments.

## Recommended run order

```bash
python3 aes_embedding.py
python3 aes_train.py
```

## Record with each experiment

- Git commit SHA.
- Python, TensorFlow, Keras, and scikit-learn versions.
- Input dataset revision and split.
- Embedding model and checkpoint details.
- Random seed and training configuration.

## Generated files

Keep embeddings, trained models, output CSV files, and logs outside Git. Store the final metrics together with the configuration used to create them.

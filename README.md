# model_spark

An interactive CLI for AutoML on local tabular data, powered by [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html).

## Quick start

```bash
uv sync
uv run model_spark
```

Choose one of the menu options:

- `train`: train a model from a local CSV/TSV file
- `predict`: score new data using a saved model directory

## Train

1. Select a local CSV or TSV file.
2. Choose the target column.
3. Choose the task type: classification or regression.
4. Set a training time limit.

AutoGluon will try multiple tabular models and build an ensemble. Training time depends on dataset size, task, and the time limit you set.

The output is a model directory, not a single `.joblib` file. Model directories can be large.

## Predict

1. Select a saved model directory.
2. Provide a CSV/TSV with the same feature columns used during training (omit the target column).
3. `model_spark` validates:
   - missing or unexpected columns
   - feature order
   - numeric vs. categorical column types

If validation passes, predictions are written to a new CSV file.

New categorical values are allowed during prediction; structural mismatches and type mismatches are rejected.

## Notes and limits

- Best for local tabular data files
- Training can be slow and memory-intensive on large datasets
- CPU use only
- Saved AutoGluon models are directories and can take significant disk space
- AutoGluon 1.4 supports Python 3.10–3.12

> AutoGluon 1.4 is compatible with Python 3.10–3.12. This project pins Python to that supported range to avoid dependency resolution issues with newer Python versions.

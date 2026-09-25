# model_spark

An interactive CLI for AutoML on local tabular data, powered by [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html).

## Quick start

```bash
uv sync
uv run model_spark
```

Then choose one of the menu options:

- `train` to build a model from a local CSV/TSV
- `predict` to score new data with an existing model

---

## What this tool does

`model_spark` helps you do two common tasks:

### 1) Train a model

- Pick a local CSV or TSV file
- Choose the target column
- Choose the task type: classification or regression
- Set a training time limit

AutoGluon then tries many tabular model families and combines them into an ensemble.

### 2) Predict with a saved model

- Load a saved model directory
- Provide a CSV/TSV with the same feature columns
- `model_spark` checks that the columns and data types match the training data
- It writes predictions to a new CSV file

---

## Typical workflow

### Train

```text
train
-> select file
-> choose target column
-> choose task type
-> choose time limit
-> wait for AutoGluon to train
-> save model directory
```

What to expect:

- Training can take seconds, minutes, or much longer depending on data size and time limit
- The output is a model directory, not a single `.joblib` file
- Model folders can be large because they contain ensembles, metadata, and preprocessing artifacts

### Predict

```text
predict
-> select saved model directory
-> select CSV/TSV with features
-> validation checks run automatically
-> predictions are written to a new CSV
```

What to expect:

- Missing columns, extra columns, wrong feature order, or type mismatches are rejected
- New categorical values are usually allowed during prediction because AutoGluon handles categorical preprocessing
- Predictions are typically much faster than training

---

## Input requirements

### Training data

- CSV or TSV format
- One target column that you want to predict
- Other columns are treated as features
- Numeric columns and categorical columns are supported

### Prediction data

- CSV or TSV format
- Must contain the feature columns used during training
- Column names and feature order should match the training data
- Numeric vs. categorical column types must still match

> A structural mismatch is rejected before prediction is made.

---

## Good habits

- Start with a small time limit to test the workflow quickly
- Use a clean dataset with obvious target and feature columns
- Keep your saved model directories together with the data they were trained on
- Validate a sample of predictions before relying on them in production

---

## Limitations and caveats

### Data limitations

- Best suited for local tabular data files
- Large datasets can require significant CPU and memory
- Not designed for time-series forecasting workflows

### Model limitations

- AutoGluon models are directories, not simple single-file models
- Ensemble models are harder to inspect than a single algorithm
- Model performance depends strongly on how similar new data is to the training data

### Prediction limitations

- If the input data distribution changes a lot, predictions may degrade
- Regression models do not extrapolate well beyond the training range
- Very large or very sparse datasets may be slow or memory-heavy

### Environment limitations

- AutoGluon 1.4 is compatible with Python 3.10–3.12
- This project pins Python to that supported range to avoid dependency issues
- Installation can take time because AutoGluon and its dependencies are fairly heavy

---

## Resource notes

### Disk space

Saved models can be large, sometimes hundreds of MB or more depending on the dataset and time limit.

### RAM

Training can consume a lot of memory. Large tabular datasets may need more than a typical laptop has available.

### CPU/GPU

- CPU-only training is supported
- GPU acceleration may help in some cases, but it is not required
- Training speed depends on your dataset and hardware

---

## Troubleshooting

### "File not found"

Check that the path is correct and that the file exists locally.

### "Column mismatch"

Make sure the prediction file has the same feature columns as the training file, in the expected order and types.

### "Slow training"

Reduce the time limit or use a smaller dataset while testing.

### "Memory errors"

Use a smaller dataset or a machine with more RAM.

---

## Python version

> AutoGluon 1.4 is compatible with Python 3.10–3.12. This project pins Python to that supported range to avoid dependency resolution issues with newer Python versions.

```bash
python --version
```

If your Python version is outside that range, install or select a supported version before running the CLI.

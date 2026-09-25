# model_spark

An interactive CLI for AutoML on local tabular data, powered by [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html).

## Usage

```bash
uv sync
uv run model_spark
```

### Train

Choose `train`, provide a local CSV or TSV file, select the target and task type, then set a training time limit. AutoGluon searches across multiple tabular model families and creates ensembles instead of this application manually comparing a few sklearn estimators. The resulting model is stored as an AutoGluon model directory, for example `target_autogluon_model/`.

### Predict

Choose `predict`, load the saved model directory, and provide a CSV/TSV containing the feature columns. Before prediction, model_spark checks:

- missing and unexpected columns,
- feature order,
- numeric versus categorical feature kinds.

Predictions are written to a new CSV file. New categorical values are allowed because AutoGluon handles categorical feature processing; structural and type mismatches are rejected.

AutoGluon models are directories rather than single `.joblib` files and can be large. Training time and resource usage depend on the dataset and selected time limit. The model directory contains serialized artifacts; only load models from trusted sources.

> AutoGluon currently has platform and Python-version-specific dependencies. If `uv sync` cannot resolve the environment, use a Python version supported by the installed AutoGluon release (typically Python 3.10–3.13) rather than forcing Python 3.14.

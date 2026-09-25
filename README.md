# model_spark

A streamlined interactive CLI to train and use machine-learning models on local tabular data.

## Usage

```bash
uv sync
uv run model_spark
```

The CLI supports two workflows:

- **Train**: select a CSV/TSV file, preview its columns and missing values, choose a target, and let model_spark compare a linear model with a random forest. The best pipeline is validated, retrained on all labeled rows, and saved as a `.joblib` model.
- **Predict**: load a saved model and a CSV/TSV file. The CLI verifies that feature names, data kinds, and categorical values match the training data before writing predictions to a new CSV.

Models include preprocessing (missing-value handling, scaling, and categorical encoding), so the prediction workflow can use the saved artifact without additional setup. Never load model files from untrusted sources because joblib uses pickle internally.

# model_spark

An interactive CLI for AutoML on local tabular data, powered by [AutoGluon Tabular](https://auto.gluon.ai/stable/tutorials/tabular/index.html).

## Installation & Setup

```bash
uv sync
uv run model_spark
```

**Requirements:**
- Python 3.10–3.12 (AutoGluon 1.4 compatibility)
- Local machine with sufficient disk space and RAM (see [Resource Requirements](#resource-requirements) below)

## Quick Start

When you run the CLI, you'll be presented with an interactive menu to choose between two operations:

1. **Train**: Create a new AutoML model from your data
2. **Predict**: Use a saved model to make predictions on new data

---

## Train

### How It Works

1. **Select file**: Provide a local CSV or TSV file path
2. **Choose target**: Select the column you want to predict
3. **Select task type**: Specify whether this is a classification or regression task
4. **Set time limit**: Define how long AutoGluon should spend searching for the best model (in seconds)

AutoGluon will automatically:
- Explore multiple tabular model families (gradient boosting, neural networks, random forests, etc.)
- Create ensembles that combine predictions from different models
- Handle feature preprocessing and encoding
- Return the best-performing model based on your dataset

### What to Expect

- **Training duration**: Can range from seconds to hours depending on:
  - Dataset size (rows and columns)
  - Time limit you specify
  - Model complexity and ensemble building
  - Available system resources (CPU/GPU, RAM)
  
- **Output**: A model directory containing:
  - Serialized model artifacts and ensembles
  - Feature metadata and preprocessing information
  - Performance metrics and model rankings
  - Configuration files for reproducibility

- **Performance**: The model is optimized for accuracy on tabular data. AutoGluon typically outperforms manually tuned models due to its ensemble approach.

### Input Data Requirements

- **Format**: CSV or TSV files
- **Target column**: Must exist and contain your labels
  - For classification: categorical values (will be automatically encoded)
  - For regression: numeric values
- **Features**: All other columns are used as input features
  - Can be numeric or categorical
  - AutoGluon handles mixed types automatically
  - Missing values (NaN) are supported and handled during preprocessing

### Example

```bash
$ uv run model_spark
# Select: train
# File: ./data/sales.csv
# Target column: revenue
# Task type: regression
# Time limit: 600 (seconds, 10 minutes)
# Result: Model saved to ./AutogluonModels/sales_model/
```

---

## Predict

### How It Works

1. **Load model**: Provide the path to a previously saved model directory
2. **Provide data**: Supply a CSV/TSV file with feature columns matching the training data
3. **Validation checks**: model_spark automatically verifies data compatibility:
   - All required columns are present
   - No unexpected extra columns
   - Feature order matches training
   - Numeric columns are numeric, categorical columns are categorical
4. **Generate predictions**: New predictions are written to a new CSV file

### What to Expect

- **Input validation**: Structural mismatches (missing columns, type errors) will cause predictions to be rejected with clear error messages
- **Categorical handling**: New categorical values not seen during training are allowed—AutoGluon's feature processing handles these gracefully
- **Output**: CSV file with predictions and optionally prediction probabilities (for classification)
- **Performance**: Prediction is typically very fast, limited mainly by I/O and data size

### Input Data Requirements

- **Feature columns**: Must match the training data exactly
  - Same column names
  - Same order (though order validation occurs before prediction)
  - Same data types (numeric/categorical)
- **New values**: 
  - Numeric features: Any value is accepted
  - Categorical features: New categories are allowed (AutoGluon will handle them)
- **Missing values**: Handled during prediction (consistent with training)

### Example

```bash
$ uv run model_spark
# Select: predict
# Model directory: ./AutogluonModels/sales_model/
# Data file: ./data/sales_new.csv
# Validation: ✓ All checks passed
# Predictions saved to: ./sales_new_predictions.csv
```

---

## Model Storage & Resource Requirements

### Disk Space
- **Model directories**: Can be quite large (50MB to several GB depending on your dataset)
  - Each model file contains serialized Python objects
  - Ensemble models may contain multiple sub-models
  - Feature metadata and preprocessing artifacts add overhead

### Memory (RAM)
- **During training**: AutoGluon may require 2–4x your dataset size in RAM
- **During prediction**: Typically much smaller, dependent on model size
- **Recommendation**: Monitor system resources if working with large datasets (>1GB)

### GPU Support
- AutoGluon can leverage GPUs for faster training (optional)
- CPU-only training is fully supported
- GPU acceleration primarily benefits neural network and gradient boosting models

---

## Limitations

### Data Limitations
- **Local files only**: Currently works with local CSV/TSV files (not remote URLs or databases)
- **Table size**: Best suited for datasets under a few GB (local memory constraints)
- **Time series**: Not optimized for time series data (treats each row independently)

### Model Limitations
- **Interpretability**: Ensemble models can be less interpretable than single models
- **Deployment**: Models are directories, not single files—requires careful handling for production
- **Python version**: Locked to Python 3.10–3.12 due to AutoGluon compatibility
- **Feature engineering**: Limited automatic feature engineering; feature preprocessing is automatic but feature creation is not

### Prediction Limitations
- **Out-of-domain performance**: Model quality degrades for data significantly different from training data
- **Extrapolation**: Regression models don't extrapolate well beyond training data ranges
- **Drift**: Model performance may degrade if the data distribution changes over time

### Platform Limitations
- **OS compatibility**: Tested on Linux and macOS; Windows support may vary
- **Dependencies**: Relies on external packages (AutoGluon, Pandas); installation can be slow due to compilation

---

## Python Version Compatibility

> AutoGluon 1.4 is compatible with Python 3.10–3.12. This project pins Python to that supported range to avoid dependency resolution issues with newer Python versions.

Using Python outside this range may result in dependency conflicts or unexpected behavior. Always verify your Python version before running:

```bash
python --version
```

---

## Best Practices

1. **Prepare your data**: Clean and validate CSV/TSV files before training
2. **Set realistic time limits**: Start with 60–300 seconds for testing; use 600+ seconds for production
3. **Monitor resources**: Watch RAM/CPU usage during training on large datasets
4. **Keep model directories**: Don't manually edit or move model directories; treat them as opaque units
5. **Test predictions**: Always validate predictions on a sample before deploying to production

---

## Troubleshooting

- **"File not found"**: Verify the file path is absolute or relative to your current working directory
- **"Memory error"**: Reduce dataset size or increase available RAM
- **"Type mismatch"**: Ensure prediction data has identical column types as training data
- **Slow training**: Reduce time limit for faster iterations, or upgrade hardware for better performance

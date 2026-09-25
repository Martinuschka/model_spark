"""Interactive Rich CLI for small, explainable AutoML workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

console = Console()


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]model_spark[/bold cyan]\n[dim]Ignite your model![/dim]",
            border_style="cyan",
        )
    )


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if file_path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        console.print("[yellow]The selected file is not a CSV-like file; trying to read it anyway.[/yellow]")
    separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(file_path, sep=separator)
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("The data must contain at least two columns and one row.")
    return frame


def _data_kind(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        return "categorical"
    return "other"


def _show_data(frame: pd.DataFrame) -> None:
    table = Table(title=f"Data preview ({len(frame):,} rows × {len(frame.columns)} columns)")
    table.add_column("Column", style="cyan")
    table.add_column("Type")
    table.add_column("Missing")
    table.add_column("Example")
    for name in frame.columns:
        values = frame[name].dropna()
        example = "" if values.empty else str(values.iloc[0])
        table.add_row(name, str(frame[name].dtype), str(int(frame[name].isna().sum())), example[:40])
    console.print(table)


def _make_pipeline(frame: pd.DataFrame, task: str) -> Pipeline:
    numeric = [name for name in frame.columns if _data_kind(frame[name]) == "numeric"]
    categorical = [name for name in frame.columns if name not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="error"))]), categorical))
    preprocess = ColumnTransformer(transformers=transformers)
    if task == "classification":
        estimator: Any = LogisticRegression(max_iter=1000)
    else:
        estimator = Ridge()
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _candidates(task: str) -> list[tuple[str, Any]]:
    if task == "classification":
        return [
            ("Logistic regression", LogisticRegression(max_iter=1000)),
            ("Random forest", RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)),
        ]
    return [
        ("Ridge regression", Ridge()),
        ("Random forest", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ]


def _train(frame: pd.DataFrame, target: str, task: str) -> tuple[Pipeline, dict[str, Any]]:
    if frame[target].isna().all():
        raise ValueError("The target column contains no usable values.")
    data = frame.dropna(subset=[target]).copy()
    x = data.drop(columns=[target])
    y = data[target]
    if task == "classification" and y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Regression requires a numeric target column.")
    if len(data) < 6:
        raise ValueError("At least six labeled rows are required for a useful validation split.")

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    results: list[tuple[float, str, Pipeline]] = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task_id = progress.add_task("Trying candidate models...", total=None)
        for name, estimator in _candidates(task):
            pipeline = _make_pipeline(x, task)
            pipeline.set_params(model=estimator)
            pipeline.fit(x_train, y_train)
            prediction = pipeline.predict(x_test)
            if task == "classification":
                score = float(accuracy_score(y_test, prediction))
            else:
                score = float(r2_score(y_test, prediction))
            results.append((score, name, pipeline))
            progress.update(task_id, description=f"Tried {name}")

    score, name, pipeline = max(results, key=lambda item: item[0])
    pipeline.fit(x, y)
    kinds = {column: _data_kind(x[column]) for column in x.columns}
    categories = {
        column: sorted(x[column].dropna().astype(str).unique().tolist())
        for column, kind in kinds.items()
        if kind == "categorical"
    }
    metadata = {
        "format_version": 1,
        "target": target,
        "task": task,
        "feature_columns": list(x.columns),
        "feature_kinds": kinds,
        "categorical_values": categories,
        "model_name": name,
        "validation_score": score,
        "rows_trained": len(data),
    }
    return pipeline, metadata


def _save_model(pipeline: Pipeline, metadata: dict[str, Any], path: str) -> None:
    output = Path(path).expanduser()
    if output.suffix.lower() != ".joblib":
        output = output.with_suffix(".joblib")
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, output)
    console.print(f"[green]Saved model to {output}[/green]")


def _load_bundle(path: str) -> dict[str, Any]:
    bundle = joblib.load(Path(path).expanduser())
    if not isinstance(bundle, dict) or "pipeline" not in bundle or "metadata" not in bundle:
        raise ValueError("This file is not a model_spark model.")
    return bundle


def _validate_input(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    expected = metadata["feature_columns"]
    actual = list(frame.columns)
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise ValueError("Input data does not match the trained model (" + "; ".join(details) + ").")
    for column, expected_kind in metadata["feature_kinds"].items():
        actual_kind = _data_kind(frame[column])
        if actual_kind != expected_kind:
            raise ValueError(f"Column '{column}' is {actual_kind}, but the model expects {expected_kind} data.")
        if expected_kind == "categorical":
            known = set(metadata.get("categorical_values", {}).get(column, []))
            supplied = set(frame[column].dropna().astype(str).unique())
            unknown = sorted(supplied - known)
            if unknown:
                raise ValueError(f"Column '{column}' contains unseen categories: {', '.join(unknown[:5])}.")
    return frame[expected]


def train_workflow() -> None:
    frame = _read_csv(Prompt.ask("Path to training CSV"))
    _show_data(frame)
    target = Prompt.ask("Target column", choices=[str(column) for column in frame.columns])
    inferred = "classification" if _data_kind(frame[target]) == "categorical" else "regression"
    task = Prompt.ask("Task type", choices=["classification", "regression"], default=inferred)
    pipeline, metadata = _train(frame, target, task)
    metric = "accuracy" if task == "classification" else "R²"
    console.print(f"[bold green]Best model:[/bold green] {metadata['model_name']} ({metric}: {metadata['validation_score']:.3f})")
    if Confirm.ask("Save this trained model?", default=True):
        _save_model(pipeline, metadata, Prompt.ask("Output model path", default=f"{target}_model.joblib"))


def predict_workflow() -> None:
    bundle = _load_bundle(Prompt.ask("Path to stored .joblib model"))
    frame = _read_csv(Prompt.ask("Path to prediction CSV"))
    features = _validate_input(frame, bundle["metadata"])
    predictions = bundle["pipeline"].predict(features)
    output = frame.copy()
    output["prediction"] = predictions
    destination = Path(Prompt.ask("Output CSV path", default="predictions.csv")).expanduser()
    output.to_csv(destination, index=False)
    console.print(f"[green]Wrote {len(output):,} predictions to {destination}[/green]")


def run() -> None:
    print_banner()
    while True:
        choice = Prompt.ask("What would you like to do?", choices=["train", "predict", "quit"], default="train")
        try:
            if choice == "train":
                train_workflow()
            elif choice == "predict":
                predict_workflow()
            else:
                console.print("Goodbye!")
                return
        except (FileNotFoundError, ValueError, OSError) as error:
            console.print(f"[bold red]Error:[/bold red] {error}")
        if not Confirm.ask("Return to the main menu?", default=True):
            return


if __name__ == "__main__":
    run()

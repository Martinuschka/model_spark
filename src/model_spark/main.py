"""Interactive Rich CLI backed by AutoGluon's tabular AutoML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from model_spark.file_scanner import select_csv_file, select_model_directory

console = Console()
_METADATA_FILE = "model_spark_metadata.json"


def print_banner() -> None:
    print(
        r"""
        ╔═══════════════════════════════╗
        ║                               ║
        ║      ███████████████████      ║
        ║      █     █     █     █      ║
        ║      █     █     ███████      ║
        ║      █     █     █     █      ║
        ║                               ║
        ╚═══════════════════════════════╝
        """
    )

    console.print(
        Panel.fit(
            "[bold cyan]model_spark[/bold cyan]\n[dim]AutoGluon tabular AutoML — ignite your model![/dim]",
            border_style="cyan",
        )
    )


def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    separator = "\t" if file_path.suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(file_path, sep=separator)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "The data must contain at least two columns and one row."
        ) from error
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("The data must contain at least two columns and one row.")
    return frame


def _data_kind(series: pd.Series) -> str:
    # Bool is numeric in pandas, so check it before is_numeric_dtype.
    if (
        pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "other"


def _infer_task(series: pd.Series) -> str:
    """Infer an AutoGluon problem_type from the target column."""
    if _data_kind(series) == "categorical":
        return "binary" if series.nunique(dropna=True) <= 2 else "multiclass"
    return "regression"


def _show_data(frame: pd.DataFrame) -> None:
    table = Table(
        title=f"Data preview ({len(frame):,} rows × {len(frame.columns)} columns)"
    )
    table.add_column("Column", style="cyan")
    table.add_column("Type")
    table.add_column("Missing")
    table.add_column("Example")
    for name in frame.columns:
        values = frame[name].dropna()
        example = "" if values.empty else str(values.iloc[0])
        table.add_row(
            name,
            str(frame[name].dtype),
            str(int(frame[name].isna().sum())),
            example[:40],
        )
    console.print(table)


def _metadata(
    frame: pd.DataFrame, target: str, task: str, model_path: Path
) -> dict[str, Any]:
    features = frame.drop(columns=[target])
    return {
        "format_version": 2,
        "backend": "autogluon.tabular",
        "target": target,
        "task": task,
        "feature_columns": list(features.columns),
        "feature_kinds": {
            column: _data_kind(features[column]) for column in features.columns
        },
        "training_rows": len(frame),
        "model_path": str(model_path),
    }


def _write_metadata(model_path: Path, metadata: dict[str, Any]) -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / _METADATA_FILE).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _read_metadata(model_path: Path) -> dict[str, Any]:
    metadata_path = model_path / _METADATA_FILE
    if not metadata_path.is_file():
        raise ValueError(
            f"'{model_path}' is not a model_spark AutoGluon model directory."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("The model metadata is not valid JSON.") from error
    if metadata.get("backend") != "autogluon.tabular":
        raise ValueError("This model was not created with the AutoGluon backend.")
    return metadata


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
        raise ValueError(
            "Input data does not match the trained model (" + "; ".join(details) + ")."
        )

    mismatches = []
    for column, expected_kind in metadata["feature_kinds"].items():
        actual_kind = _data_kind(frame[column])
        if actual_kind != expected_kind:
            mismatches.append(
                f"{column}: expected {expected_kind}, received {actual_kind}"
            )
    if mismatches:
        raise ValueError(
            "Input data has incompatible column types: " + "; ".join(mismatches)
        )
    return frame[expected].copy()


def _train(
    frame: pd.DataFrame, target: str, task: str, model_path: Path, time_limit: int
) -> tuple[TabularPredictor, dict[str, Any]]:
    if frame[target].isna().all():
        raise ValueError("The target column contains no usable values.")
    data = frame.dropna(subset=[target]).copy()
    if len(data) < 10:
        raise ValueError(
            "At least ten labeled rows are required for AutoGluon training."
        )
    if task in {"binary", "multiclass"} and data[target].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if task == "regression" and not pd.api.types.is_numeric_dtype(data[target]):
        raise ValueError("Regression requires a numeric target column.")

    predictor = TabularPredictor(label=target, problem_type=task, path=str(model_path))
    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"), console=console
    ) as progress:
        progress.add_task(
            "AutoGluon is searching, training, and ensembling models...", total=None
        )
        predictor.fit(
            train_data=data,
            presets="medium_quality",
            time_limit=time_limit,
            verbosity=0,
        )
    metadata = _metadata(data, target, task, model_path)
    leaderboard = predictor.leaderboard(data, silent=True)
    metadata["best_model"] = (
        str(leaderboard.iloc[0]["model"])
        if not leaderboard.empty
        else "AutoGluon ensemble"
    )
    metadata["validation_score"] = (
        float(leaderboard.iloc[0]["score_val"]) if not leaderboard.empty else None
    )
    _write_metadata(model_path, metadata)
    return predictor, metadata


def train_workflow() -> None:
    frame = _read_csv(select_csv_file())
    _show_data(frame)
    target = Prompt.ask(
        "Target column", choices=[str(column) for column in frame.columns]
    )
    task = Prompt.ask(
        "Task type",
        choices=["binary", "multiclass", "regression"],
        default=_infer_task(frame[target]),
    )
    time_limit = IntPrompt.ask("Maximum training time in seconds", default=300)
    default_path = f"{target}_autogluon_model"
    model_path = Path(
        Prompt.ask("Output model directory", default=default_path)
    ).expanduser()
    if (
        model_path.exists()
        and any(model_path.iterdir())
        and not Confirm.ask(f"'{model_path}' is not empty. Replace it?", default=False)
    ):
        return
    predictor, metadata = _train(frame, target, task, model_path, time_limit)
    score = metadata.get("validation_score")
    score_text = "unavailable" if score is None else f"{score:.3f}"
    console.print(
        f"[bold green]AutoGluon selected:[/bold green] {metadata['best_model']} (validation score: {score_text})"
    )
    console.print(f"[green]Saved model directory to {model_path}[/green]")
    console.print(
        f"[dim]AutoGluon summary: {predictor.fit_summary(verbosity=0).get('num_bag_folds', 0)} bag folds[/dim]"
    )


def predict_workflow() -> None:
    model_path = Path(select_model_directory()).expanduser()
    metadata = _read_metadata(model_path)
    predictor = TabularPredictor.load(str(model_path))
    frame = _read_csv(select_csv_file())
    features = _validate_input(frame, metadata)
    predictions = predictor.predict(features)
    output = frame.copy()
    output["prediction"] = predictions.to_numpy()
    destination = Path(
        Prompt.ask("Output CSV path", default="predictions.csv")
    ).expanduser()
    output.to_csv(destination, index=False)
    console.print(f"[green]Wrote {len(output):,} predictions to {destination}[/green]")


def run() -> None:
    print_banner()
    while True:
        choice = Prompt.ask(
            "What would you like to do?",
            choices=["train", "predict", "quit"],
            default="train",
        )
        try:
            if choice == "train":
                train_workflow()
            elif choice == "predict":
                predict_workflow()
            else:
                console.print("Goodbye!")
                return
        except (FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            console.print(f"[bold red]Error:[/bold red] {error}")
        if not Confirm.ask("Return to the main menu?", default=True):
            return


if __name__ == "__main__":
    run()

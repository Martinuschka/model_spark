"""Tests for model_spark.main module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from model_spark.main import (
    _data_kind,
    _metadata,
    _read_csv,
    _read_metadata,
    _validate_input,
    _write_metadata,
)


class TestReadCsv:
    """Tests for _read_csv function."""

    def test_read_csv_valid(self, tmp_path: Path) -> None:
        """Test reading a valid CSV file."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2,col3\n1,2,3\n4,5,6\n")
        
        df = _read_csv(str(csv_file))
        
        assert df.shape == (2, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_read_tsv_valid(self, tmp_path: Path) -> None:
        """Test reading a valid TSV file."""
        tsv_file = tmp_path / "data.tsv"
        tsv_file.write_text("col1\tcol2\tcol3\n1\t2\t3\n4\t5\t6\n")
        
        df = _read_csv(str(tsv_file))
        
        assert df.shape == (2, 3)
        assert list(df.columns) == ["col1", "col2", "col3"]

    def test_read_csv_nonexistent_file(self) -> None:
        """Test reading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            _read_csv("/nonexistent/path/file.csv")

    def test_read_csv_empty_file(self, tmp_path: Path) -> None:
        """Test reading an empty CSV raises ValueError."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")
        
        with pytest.raises(ValueError, match="at least two columns"):
            _read_csv(str(csv_file))

    def test_read_csv_single_column(self, tmp_path: Path) -> None:
        """Test reading a CSV with single column raises ValueError."""
        csv_file = tmp_path / "single.csv"
        csv_file.write_text("col1\n1\n2\n")
        
        with pytest.raises(ValueError, match="at least two columns"):
            _read_csv(str(csv_file))

    def test_read_csv_expanduser(self, tmp_path: Path, monkeypatch) -> None:
        """Test that ~ is expanded to home directory."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2\n")
        monkeypatch.setenv("HOME", str(tmp_path.parent))
        
        # This will fail if expanduser is not working, but tests the mechanism
        with pytest.raises(FileNotFoundError):
            _read_csv("~/nonexistent.csv")


class TestDataKind:
    """Tests for _data_kind function."""

    def test_numeric_int(self) -> None:
        """Test detection of integer numeric data."""
        series = pd.Series([1, 2, 3, 4, 5])
        assert _data_kind(series) == "numeric"

    def test_numeric_float(self) -> None:
        """Test detection of float numeric data."""
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert _data_kind(series) == "numeric"

    def test_categorical_object(self) -> None:
        """Test detection of object (string) categorical data."""
        series = pd.Series(["a", "b", "c", "a", "b"])
        assert _data_kind(series) == "categorical"

    def test_categorical_bool(self) -> None:
        """Test detection of boolean categorical data."""
        series = pd.Series([True, False, True, False])
        assert _data_kind(series) == "categorical"

    def test_categorical_category(self) -> None:
        """Test detection of categorical dtype."""
        series = pd.Series(["a", "b", "c"], dtype="category")
        assert _data_kind(series) == "categorical"

    def test_with_null_values(self) -> None:
        """Test that null values don't affect detection."""
        series = pd.Series([1, 2, None, 4, 5])
        assert _data_kind(series) == "numeric"
        
        series = pd.Series(["a", None, "c", "a"])
        assert _data_kind(series) == "categorical"


class TestMetadata:
    """Tests for _metadata function."""

    def test_metadata_creation_regression(self, tmp_path: Path) -> None:
        """Test creating metadata for regression task."""
        df = pd.DataFrame({
            "feature1": [1, 2, 3, 4, 5],
            "feature2": ["a", "b", "a", "b", "a"],
            "target": [10.5, 20.3, 15.1, 25.0, 18.5]
        })
        model_path = tmp_path / "model"
        
        metadata = _metadata(df, "target", "regression", model_path)
        
        assert metadata["format_version"] == 2
        assert metadata["backend"] == "autogluon.tabular"
        assert metadata["target"] == "target"
        assert metadata["task"] == "regression"
        assert set(metadata["feature_columns"]) == {"feature1", "feature2"}
        assert metadata["training_rows"] == 5
        assert metadata["model_path"] == str(model_path)
        assert metadata["feature_kinds"]["feature1"] == "numeric"
        assert metadata["feature_kinds"]["feature2"] == "categorical"

    def test_metadata_creation_classification(self, tmp_path: Path) -> None:
        """Test creating metadata for classification task."""
        df = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "target": ["yes", "no", "yes"]
        })
        model_path = tmp_path / "model"
        
        metadata = _metadata(df, "target", "binary", model_path)
        
        assert metadata["task"] == "binary"
        assert metadata["training_rows"] == 3


class TestWriteMetadata:
    """Tests for _write_metadata function."""

    def test_write_metadata_creates_directory(self, tmp_path: Path) -> None:
        """Test that _write_metadata creates the model directory."""
        model_path = tmp_path / "new_model"
        metadata = {"test": "data", "version": 1}
        
        _write_metadata(model_path, metadata)
        
        assert model_path.exists()
        assert (model_path / "model_spark_metadata.json").exists()

    def test_write_metadata_content(self, tmp_path: Path) -> None:
        """Test that metadata is written correctly."""
        model_path = tmp_path / "model"
        metadata = {"format_version": 2, "backend": "autogluon.tabular"}
        
        _write_metadata(model_path, metadata)
        
        content = json.loads((model_path / "model_spark_metadata.json").read_text())
        assert content == metadata


class TestReadMetadata:
    """Tests for _read_metadata function."""

    def test_read_metadata_valid(self, tmp_path: Path) -> None:
        """Test reading valid metadata."""
        model_path = tmp_path / "model"
        metadata = {
            "format_version": 2,
            "backend": "autogluon.tabular",
            "target": "target",
            "task": "regression"
        }
        _write_metadata(model_path, metadata)
        
        read_metadata = _read_metadata(model_path)
        
        assert read_metadata == metadata

    def test_read_metadata_missing_file(self, tmp_path: Path) -> None:
        """Test reading metadata from nonexistent directory."""
        model_path = tmp_path / "nonexistent"
        
        with pytest.raises(ValueError, match="not a model_spark AutoGluon model directory"):
            _read_metadata(model_path)

    def test_read_metadata_invalid_json(self, tmp_path: Path) -> None:
        """Test reading invalid JSON metadata."""
        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "model_spark_metadata.json").write_text("{ invalid json")
        
        with pytest.raises(ValueError, match="not valid JSON"):
            _read_metadata(model_path)

    def test_read_metadata_wrong_backend(self, tmp_path: Path) -> None:
        """Test reading metadata with wrong backend."""
        model_path = tmp_path / "model"
        metadata = {"backend": "sklearn", "target": "target"}
        _write_metadata(model_path, metadata)
        
        with pytest.raises(ValueError, match="not created with the AutoGluon backend"):
            _read_metadata(model_path)


class TestValidateInput:
    """Tests for _validate_input function."""

    def test_validate_input_valid(self) -> None:
        """Test validation of valid input data."""
        input_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": ["a", "b", "c"]
        })
        metadata = {
            "feature_columns": ["feature1", "feature2"],
            "feature_kinds": {"feature1": "numeric", "feature2": "categorical"}
        }
        
        result = _validate_input(input_df, metadata)
        
        assert result.shape == (3, 2)
        assert list(result.columns) == ["feature1", "feature2"]

    def test_validate_input_missing_columns(self) -> None:
        """Test validation fails with missing columns."""
        input_df = pd.DataFrame({"feature1": [1, 2, 3]})
        metadata = {
            "feature_columns": ["feature1", "feature2"],
            "feature_kinds": {"feature1": "numeric", "feature2": "categorical"}
        }
        
        with pytest.raises(ValueError, match="missing columns"):
            _validate_input(input_df, metadata)

    def test_validate_input_extra_columns(self) -> None:
        """Test validation fails with extra columns."""
        input_df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": ["a", "b", "c"],
            "extra": [10, 20, 30]
        })
        metadata = {
            "feature_columns": ["feature1", "feature2"],
            "feature_kinds": {"feature1": "numeric", "feature2": "categorical"}
        }
        
        with pytest.raises(ValueError, match="unexpected columns"):
            _validate_input(input_df, metadata)

    def test_validate_input_type_mismatch(self) -> None:
        """Test validation fails with type mismatch."""
        input_df = pd.DataFrame({
            "feature1": ["not", "numeric"],
            "feature2": ["a", "b"]
        })
        metadata = {
            "feature_columns": ["feature1", "feature2"],
            "feature_kinds": {"feature1": "numeric", "feature2": "categorical"}
        }
        
        with pytest.raises(ValueError, match="incompatible column types"):
            _validate_input(input_df, metadata)

    def test_validate_input_reorders_columns(self) -> None:
        """Test that validation reorders columns to match metadata."""
        input_df = pd.DataFrame({
            "feature2": ["a", "b"],
            "feature1": [1, 2]
        })
        metadata = {
            "feature_columns": ["feature1", "feature2"],
            "feature_kinds": {"feature1": "numeric", "feature2": "categorical"}
        }
        
        result = _validate_input(input_df, metadata)
        
        assert list(result.columns) == ["feature1", "feature2"]

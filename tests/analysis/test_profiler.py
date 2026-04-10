import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.profiler import profile_dataset

TITANIC_CSV = Path("projects/titanic/data/raw/train.csv")
TITANIC_YAML = Path("projects/titanic/project.yaml")
FIXTURES = Path("tests/fixtures")

NUMERIC_STATS_KEYS = {"count", "mean", "std", "min", "p25", "p50", "p75", "max"}
CATEGORICAL_STATS_KEYS = {"count", "unique", "top", "freq"}


@pytest.fixture
def titanic_profile(tmp_path):
    output = tmp_path / "profile.json"
    return profile_dataset(str(TITANIC_CSV), str(output), str(TITANIC_YAML))


@pytest.fixture
def titanic_schema(titanic_profile):
    return {col["name"]: col for col in titanic_profile["schema"]["columns"]}


# --- Schema ---

def test_schema_infers_all_dtypes(titanic_profile):
    columns = titanic_profile["schema"]["columns"]
    assert len(columns) == 12
    assert all("pandas_dtype" in col for col in columns)
    assert all(col["pandas_dtype"] != "" for col in columns)


def test_semantic_type_identifier(titanic_schema):
    assert titanic_schema["PassengerId"]["inferred_semantic_type"] == "identifier"


def test_semantic_type_binary_target(titanic_schema):
    assert titanic_schema["Survived"]["inferred_semantic_type"] == "binary_target"


def test_semantic_type_continuous(titanic_schema):
    assert titanic_schema["Age"]["inferred_semantic_type"] == "continuous_numeric"


def test_semantic_type_low_cardinality_categorical(titanic_schema):
    assert titanic_schema["Sex"]["inferred_semantic_type"] == "low_cardinality_categorical"


def test_semantic_type_high_cardinality_categorical(titanic_schema):
    assert titanic_schema["Name"]["inferred_semantic_type"] == "high_cardinality_categorical"


# --- Basic stats ---

def test_numeric_stats_keys_complete(titanic_profile):
    numeric_cols = ["PassengerId", "Age", "Fare", "SibSp", "Parch", "Pclass"]
    for col in numeric_cols:
        assert NUMERIC_STATS_KEYS == set(titanic_profile["basic_stats"][col].keys()), col


def test_categorical_stats_keys_complete(titanic_profile):
    categorical_cols = ["Sex", "Embarked", "Name", "Ticket"]
    for col in categorical_cols:
        assert CATEGORICAL_STATS_KEYS == set(titanic_profile["basic_stats"][col].keys()), col


# --- Source integrity ---

def test_source_hash_stable(tmp_path):
    out1 = tmp_path / "p1.json"
    out2 = tmp_path / "p2.json"
    p1 = profile_dataset(str(TITANIC_CSV), str(out1))
    p2 = profile_dataset(str(TITANIC_CSV), str(out2))
    assert p1["source"]["sha256"] == p2["source"]["sha256"]


def test_output_written(tmp_path):
    output = tmp_path / "profile.json"
    profile_dataset(str(TITANIC_CSV), str(output))
    assert output.exists()


# --- Robustness ---

def test_bad_path_exits_nonzero(tmp_path):
    output = tmp_path / "profile.json"
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.profiler",
         "--input", "nonexistent.csv",
         "--output", str(output)],
        capture_output=True,
    )
    assert result.returncode == 1
    assert output.exists()
    error = json.loads(output.read_text())
    assert error["error"] is True
    assert "message" in error


def test_no_nan_in_json(tmp_path):
    output = tmp_path / "profile.json"
    profile_dataset(str(TITANIC_CSV), str(output))
    # json.loads raises ValueError if NaN literals are present
    data = json.loads(output.read_text())
    assert isinstance(data, dict)


# --- Fixtures ---

def test_fixture_simple_numeric(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert profile["source"]["row_count"] == 20
    assert profile["source"]["column_count"] == 3
    assert output.exists()


def test_fixture_mixed_types(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "mixed_types.csv"), str(output))
    assert profile["source"]["row_count"] == 20
    assert profile["source"]["column_count"] == 5
    schema = {col["name"]: col for col in profile["schema"]["columns"]}
    assert schema["name"]["inferred_semantic_type"] == "low_cardinality_categorical"
    assert schema["score"]["inferred_semantic_type"] == "continuous_numeric"
    assert output.exists()

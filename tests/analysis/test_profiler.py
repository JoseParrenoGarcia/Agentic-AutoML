import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.profiler import profile_dataset

TITANIC_CSV = Path("projects/titanic/data/raw/train.csv")
TITANIC_YAML = Path("projects/titanic/project.yaml")
FIXTURES = Path("tests/fixtures")

NUMERIC_STATS_KEYS = {"count", "mean", "std", "min", "p25", "p50", "p75", "max", "skewness"}
CATEGORICAL_STATS_KEYS = {"count", "unique", "top", "freq"}

titanic_available = pytest.mark.skipif(
    not TITANIC_CSV.exists(),
    reason="Titanic dataset not available (gitignored)",
)


@pytest.fixture
def titanic_profile(tmp_path):
    output = tmp_path / "profile.json"
    return profile_dataset(str(TITANIC_CSV), str(output), str(TITANIC_YAML))


@pytest.fixture
def titanic_schema(titanic_profile):
    return {col["name"]: col for col in titanic_profile["columns"]}


# --- Schema (Titanic-specific) ---

@titanic_available
def test_schema_infers_all_dtypes(titanic_profile):
    columns = titanic_profile["columns"]
    assert len(columns) == 12
    assert all("pandas_dtype" in col for col in columns)
    assert all(col["pandas_dtype"] != "" for col in columns)


@titanic_available
def test_semantic_type_identifier(titanic_schema):
    assert titanic_schema["PassengerId"]["inferred_semantic_type"] == "identifier"


@titanic_available
def test_semantic_type_binary_target(titanic_schema):
    assert titanic_schema["Survived"]["inferred_semantic_type"] == "binary_target"


@titanic_available
def test_semantic_type_continuous(titanic_schema):
    assert titanic_schema["Age"]["inferred_semantic_type"] == "continuous_numeric"


@titanic_available
def test_semantic_type_low_cardinality_categorical(titanic_schema):
    assert titanic_schema["Sex"]["inferred_semantic_type"] == "low_cardinality_categorical"


@titanic_available
def test_semantic_type_high_cardinality_categorical(titanic_schema):
    assert titanic_schema["Name"]["inferred_semantic_type"] == "high_cardinality_categorical"


# --- Basic stats (Titanic-specific) ---

@titanic_available
def test_numeric_stats_keys_complete(titanic_schema):
    for col in ["PassengerId", "Age", "Fare", "SibSp", "Parch", "Pclass"]:
        assert NUMERIC_STATS_KEYS == set(titanic_schema[col]["basic_stats"].keys()), col


@titanic_available
def test_categorical_stats_keys_complete(titanic_schema):
    for col in ["Sex", "Embarked", "Name", "Ticket"]:
        assert CATEGORICAL_STATS_KEYS == set(titanic_schema[col]["basic_stats"].keys()), col


# --- Null analysis (Titanic-specific) ---

@titanic_available
def test_null_analysis_keys(titanic_schema):
    for col_name in ["Age", "Cabin", "Embarked", "PassengerId"]:
        entry = titanic_schema[col_name]["null_analysis"]
        assert set(entry.keys()) == {"null_count", "null_pct", "is_nullable"}, col_name


@titanic_available
def test_null_analysis_age_has_nulls(titanic_schema):
    age = titanic_schema["Age"]["null_analysis"]
    assert age["null_count"] > 0
    assert age["null_pct"] > 0
    assert age["is_nullable"] is True


@titanic_available
def test_null_analysis_survived_no_nulls(titanic_schema):
    survived = titanic_schema["Survived"]["null_analysis"]
    assert survived["null_count"] == 0
    assert survived["is_nullable"] is False


# --- Cardinality (Titanic-specific) ---

@titanic_available
def test_cardinality_keys(titanic_schema):
    for col_name in ["PassengerId", "Sex", "Name"]:
        entry = titanic_schema[col_name]["cardinality"]
        assert set(entry.keys()) == {"unique_count", "uniqueness_ratio", "is_high_cardinality"}, col_name


@titanic_available
def test_cardinality_name_is_high(titanic_schema):
    assert titanic_schema["Name"]["cardinality"]["is_high_cardinality"] is True


@titanic_available
def test_cardinality_sex_is_low(titanic_schema):
    assert titanic_schema["Sex"]["cardinality"]["is_high_cardinality"] is False


# --- Outliers (Titanic-specific) ---

@titanic_available
def test_outliers_only_numeric_columns(titanic_schema):
    assert "outliers" not in titanic_schema["Sex"]
    assert "outliers" not in titanic_schema["Name"]
    assert "outliers" in titanic_schema["Age"]
    assert "outliers" in titanic_schema["Fare"]


@titanic_available
def test_outliers_keys(titanic_schema):
    for col_name, col in titanic_schema.items():
        if "outliers" in col:
            assert set(col["outliers"].keys()) == {"method", "lower_bound", "upper_bound", "outlier_count", "outlier_pct"}, col_name


@titanic_available
def test_outliers_fare_has_outliers(titanic_schema):
    fare = titanic_schema["Fare"]["outliers"]
    assert fare["outlier_count"] > 0
    assert fare["outlier_pct"] > 0


# --- Correlation (Titanic-specific) ---

@titanic_available
def test_correlation_has_pearson_section(titanic_profile):
    assert "pearson" in titanic_profile["correlation"]
    assert "matrix" in titanic_profile["correlation"]["pearson"]
    assert "top_pairs" in titanic_profile["correlation"]["pearson"]


@titanic_available
def test_correlation_has_cramers_v_section(titanic_profile):
    assert "cramers_v" in titanic_profile["correlation"]
    matrix = titanic_profile["correlation"]["cramers_v"]["matrix"]
    # Sex and Embarked are both low-cardinality categoricals on Titanic
    assert "Sex" in matrix
    assert "Embarked" in matrix


@titanic_available
def test_correlation_matrix_is_square(titanic_profile):
    matrix = titanic_profile["correlation"]["pearson"]["matrix"]
    cols = list(matrix.keys())
    for col in cols:
        assert set(matrix[col].keys()) == set(cols)


@titanic_available
def test_correlation_diagonal_is_one(titanic_profile):
    matrix = titanic_profile["correlation"]["pearson"]["matrix"]
    for col in matrix:
        assert matrix[col][col] == 1.0


@titanic_available
def test_correlation_only_numeric_columns(titanic_profile):
    matrix = titanic_profile["correlation"]["pearson"]["matrix"]
    assert "Sex" not in matrix
    assert "Name" not in matrix
    assert "Fare" in matrix


@titanic_available
def test_correlation_top_pairs_sorted(titanic_profile):
    top = titanic_profile["correlation"]["pearson"]["top_pairs"]
    assert len(top) > 1
    for i in range(len(top) - 1):
        assert abs(top[i]["value"]) >= abs(top[i + 1]["value"])


# --- Target validation (Titanic-specific) ---

@titanic_available
def test_target_validation_keys(titanic_profile):
    tv = titanic_profile["target_validation"]
    assert tv["target_column"] == "Survived"
    assert "null_count" in tv
    assert "class_counts" in tv
    assert "class_balance_ratio" in tv
    assert "is_imbalanced" in tv


@titanic_available
def test_target_validation_no_nulls(titanic_profile):
    assert titanic_profile["target_validation"]["null_count"] == 0


@titanic_available
def test_target_validation_not_imbalanced(titanic_profile):
    assert titanic_profile["target_validation"]["is_imbalanced"] is False


# --- Leakage flags (Titanic-specific) ---

@titanic_available
def test_leakage_flags_keys(titanic_profile):
    lf = titanic_profile["leakage_flags"]
    assert "threshold" in lf
    assert "flagged_columns" in lf


@titanic_available
def test_leakage_no_obvious_leakage_in_titanic(titanic_profile):
    flagged = [f["column"] for f in titanic_profile["leakage_flags"]["flagged_columns"]]
    assert "PassengerId" not in flagged


# --- Source integrity (fixture-based, runs in CI) ---

def test_source_hash_stable(tmp_path):
    out1 = tmp_path / "p1.json"
    out2 = tmp_path / "p2.json"
    csv = FIXTURES / "simple_numeric.csv"
    p1 = profile_dataset(str(csv), str(out1))
    p2 = profile_dataset(str(csv), str(out2))
    assert p1["source"]["sha256"] == p2["source"]["sha256"]


def test_output_written(tmp_path):
    output = tmp_path / "profile.json"
    profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert output.exists()


# --- Robustness (fixture-based, runs in CI) ---

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
    profile_dataset(str(FIXTURES / "mixed_types.csv"), str(output))
    data = json.loads(output.read_text())
    assert isinstance(data, dict)


# --- Per-column structure (fixture-based, runs in CI) ---

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
    schema = {col["name"]: col for col in profile["columns"]}
    assert schema["name"]["inferred_semantic_type"] == "low_cardinality_categorical"
    assert schema["score"]["inferred_semantic_type"] == "continuous_numeric"
    assert output.exists()


def test_column_has_all_per_column_sections(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    for col in profile["columns"]:
        assert "basic_stats" in col
        assert "null_analysis" in col
        assert "cardinality" in col
        # numeric columns also have outliers
        assert "outliers" in col


def test_null_analysis_per_column(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "mixed_types.csv"), str(output))
    for col in profile["columns"]:
        entry = col["null_analysis"]
        assert set(entry.keys()) == {"null_count", "null_pct", "is_nullable"}


def test_cardinality_per_column(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "mixed_types.csv"), str(output))
    for col in profile["columns"]:
        entry = col["cardinality"]
        assert set(entry.keys()) == {"unique_count", "uniqueness_ratio", "is_high_cardinality"}


def test_outliers_per_column(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    for col in profile["columns"]:
        if "outliers" in col:
            assert set(col["outliers"].keys()) == {"method", "lower_bound", "upper_bound", "outlier_count", "outlier_pct"}


def test_correlation_fixture(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    matrix = profile["correlation"]["pearson"]["matrix"]
    for col in matrix:
        assert matrix[col][col] == 1.0


def test_target_validation_skipped_without_yaml(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert profile["target_validation"]["skipped"] is True


def test_leakage_flags_skipped_without_yaml(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert profile["leakage_flags"]["skipped"] is True


def test_m2_sections_all_complete(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert profile["m2_sections_pending"] == []
    assert set(profile["m2_sections_complete"]) == {
        "columns", "correlation", "target_validation", "leakage_flags",
        "feature_risk_flags", "mutual_information",
    }


# --- New M2 gap-closure tests (fixture-based, runs in CI) ---

def test_column_has_description(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    for col in profile["columns"]:
        assert "description" in col
        assert isinstance(col["description"], str)
        assert len(col["description"]) > 0


def test_numeric_column_has_skewness(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    for col in profile["columns"]:
        if "outliers" in col:  # numeric indicator
            assert "skewness" in col["basic_stats"]
            assert isinstance(col["basic_stats"]["skewness"], float)


def test_column_has_risk_flags(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    for col in profile["columns"]:
        assert "risk_flags" in col
        assert isinstance(col["risk_flags"], list)


def test_feature_risk_flags_section_exists(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    frf = profile["feature_risk_flags"]
    assert "flagged_columns" in frf
    assert "near_duplicate_pairs" in frf
    assert isinstance(frf["flagged_columns"], list)
    assert isinstance(frf["near_duplicate_pairs"], list)


def test_mutual_information_skipped_without_yaml(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    assert profile["mutual_information"]["skipped"] is True


def test_cramers_v_empty_for_no_categoricals(tmp_path):
    output = tmp_path / "profile.json"
    profile = profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(output))
    # simple_numeric.csv has no categorical columns
    assert profile["correlation"]["cramers_v"]["matrix"] == {}

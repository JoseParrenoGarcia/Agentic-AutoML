from pathlib import Path

import pytest

from src.analysis.profiler import profile_dataset

FIXTURES = Path("tests/fixtures")


@pytest.fixture
def edge_profile(tmp_path):
    output = tmp_path / "profile.json"
    return profile_dataset(str(FIXTURES / "edge_cases.csv"), str(output))


@pytest.fixture
def edge_schema(edge_profile):
    return {col["name"]: col for col in edge_profile["columns"]}


@pytest.fixture
def no_numeric_profile(tmp_path):
    output = tmp_path / "profile.json"
    return profile_dataset(str(FIXTURES / "no_numeric.csv"), str(output))


# --- 100% null column ---

def test_all_null_null_count(edge_schema):
    na = edge_schema["all_null"]["null_analysis"]
    assert na["null_count"] == 20
    assert na["null_pct"] == 100.0
    assert na["is_nullable"] is True


def test_all_null_stats_do_not_crash(edge_schema):
    # all_null is float (pandas infers float for empty numeric col)
    # basic_stats should exist without raising
    assert "basic_stats" in edge_schema["all_null"]


def test_all_null_cardinality(edge_schema):
    card = edge_schema["all_null"]["cardinality"]
    assert card["unique_count"] == 0


# --- Constant column ---

def test_constant_outlier_count_is_zero(edge_schema):
    outliers = edge_schema["constant"]["outliers"]
    assert outliers["outlier_count"] == 0


def test_constant_iqr_bounds_are_equal(edge_schema):
    outliers = edge_schema["constant"]["outliers"]
    assert outliers["lower_bound"] == outliers["upper_bound"]


def test_constant_cardinality(edge_schema):
    card = edge_schema["constant"]["cardinality"]
    assert card["unique_count"] == 1
    assert card["is_high_cardinality"] is False


# --- Binary integer column ---

def test_binary_int_infers_binary_flag(edge_schema):
    assert edge_schema["binary_int"]["inferred_semantic_type"] == "binary_flag"


# --- No numeric columns ---

def test_no_numeric_correlation_matrix_empty(no_numeric_profile):
    matrix = no_numeric_profile["correlation"]["matrix"]
    assert matrix == {}


def test_no_numeric_no_outliers_key(no_numeric_profile):
    for col in no_numeric_profile["columns"]:
        assert "outliers" not in col


def test_no_numeric_profile_completes(no_numeric_profile):
    assert no_numeric_profile["m2_sections_pending"] == []

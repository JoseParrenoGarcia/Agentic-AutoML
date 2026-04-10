import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.plots import generate_plots

TITANIC_CSV = Path("projects/titanic/data/raw/train.csv")
TITANIC_PROFILE = Path("projects/titanic/artifacts/data/profile.json")
FIXTURES = Path("tests/fixtures")

titanic_available = pytest.mark.skipif(
    not TITANIC_CSV.exists() or not TITANIC_PROFILE.exists(),
    reason="Titanic dataset or profile not available",
)


# --- Titanic-specific (skip in CI) ---

@titanic_available
def test_expected_plots_written(tmp_path):
    result = generate_plots(str(TITANIC_CSV), str(TITANIC_PROFILE), str(tmp_path))
    paths = [Path(p) for p in result["paths"]]
    names = {p.name for p in paths}

    assert "correlation_heatmap.png" in names
    assert "dist_Age.png" in names
    assert "target_vs_Age.png" in names
    assert "target_vs_Sex.png" in names


@titanic_available
def test_identifier_column_has_no_dist_plot(tmp_path):
    result = generate_plots(str(TITANIC_CSV), str(TITANIC_PROFILE), str(tmp_path))
    names = {Path(p).name for p in result["paths"]}
    assert "dist_PassengerId.png" not in names


@titanic_available
def test_target_column_has_no_target_vs_plot(tmp_path):
    result = generate_plots(str(TITANIC_CSV), str(TITANIC_PROFILE), str(tmp_path))
    names = {Path(p).name for p in result["paths"]}
    assert "target_vs_Survived.png" not in names


@titanic_available
def test_all_plots_are_non_empty(tmp_path):
    result = generate_plots(str(TITANIC_CSV), str(TITANIC_PROFILE), str(tmp_path))
    for p in result["paths"]:
        assert Path(p).stat().st_size > 0, f"Empty plot: {p}"


# --- Fixture-based (runs in CI) ---

@pytest.fixture
def simple_profile(tmp_path):
    """Generate a profile.json from simple_numeric.csv for use in plot tests."""
    from src.analysis.profiler import profile_dataset
    profile_path = tmp_path / "profile.json"
    profile_dataset(str(FIXTURES / "simple_numeric.csv"), str(profile_path))
    return profile_path


def test_plots_written_to_output_dir(tmp_path, simple_profile):
    plots_dir = tmp_path / "plots"
    result = generate_plots(
        str(FIXTURES / "simple_numeric.csv"),
        str(simple_profile),
        str(plots_dir),
    )
    assert plots_dir.exists()
    assert result["plots_written"] > 0


def test_correlation_heatmap_written(tmp_path, simple_profile):
    plots_dir = tmp_path / "plots"
    result = generate_plots(
        str(FIXTURES / "simple_numeric.csv"),
        str(simple_profile),
        str(plots_dir),
    )
    names = {Path(p).name for p in result["paths"]}
    assert "correlation_heatmap.png" in names


def test_all_fixture_plots_non_empty(tmp_path, simple_profile):
    plots_dir = tmp_path / "plots"
    result = generate_plots(
        str(FIXTURES / "simple_numeric.csv"),
        str(simple_profile),
        str(plots_dir),
    )
    for p in result["paths"]:
        assert Path(p).stat().st_size > 0, f"Empty plot: {p}"


def test_bad_profile_path_exits_nonzero(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "src.analysis.plots",
         "--input", str(FIXTURES / "simple_numeric.csv"),
         "--profile", "nonexistent.json",
         "--output-dir", str(tmp_path)],
        capture_output=True,
    )
    assert result.returncode == 1

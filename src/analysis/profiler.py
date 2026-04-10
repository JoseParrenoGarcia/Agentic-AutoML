import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "0.1.0"


class _SafeEncoder(json.JSONEncoder):
    """Convert float NaN/inf to null — JSON spec does not allow NaN literals."""

    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitise(o), _one_shot)

    def _sanitise(self, obj):
        if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitise(v) for v in obj]
        return obj


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _infer_semantic_type(
    col: pd.Series,
    col_name: str,
    target_column: str | None,
    task_type: str | None,
) -> str:
    if target_column and col_name == target_column:
        if task_type == "binary_classification":
            return "binary_target"
        if task_type == "regression":
            return "regression_target"
        return "target"

    n_unique = col.nunique(dropna=True)
    n_total = len(col)

    if pd.api.types.is_bool_dtype(col):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"

    if pd.api.types.is_integer_dtype(col):
        if n_unique == n_total:
            return "identifier"
        if n_unique == 2:
            return "binary_flag"
        if n_unique <= 20:
            return "low_cardinality_integer"
        return "continuous_numeric"

    if pd.api.types.is_float_dtype(col):
        if n_unique <= 2:
            return "binary_flag"
        return "continuous_numeric"

    if pd.api.types.is_object_dtype(col) or isinstance(col.dtype, (pd.CategoricalDtype, pd.StringDtype)):
        if n_unique <= 20:
            return "low_cardinality_categorical"
        return "high_cardinality_categorical"

    return "unknown"


def _sample_values(col: pd.Series, n: int = 3) -> list:
    vals = col.dropna().head(n).tolist()
    return [v.item() if hasattr(v, "item") else v for v in vals]


def _basic_stats_numeric(col: pd.Series) -> dict:
    desc = col.describe(percentiles=[0.25, 0.5, 0.75])
    return {
        "count": int(desc["count"]),
        "mean": float(desc["mean"]),
        "std": float(desc["std"]),
        "min": float(desc["min"]),
        "p25": float(desc["25%"]),
        "p50": float(desc["50%"]),
        "p75": float(desc["75%"]),
        "max": float(desc["max"]),
    }


def _basic_stats_categorical(col: pd.Series) -> dict:
    desc = col.describe()
    return {
        "count": int(desc["count"]),
        "unique": int(desc["unique"]),
        "top": desc["top"],
        "freq": int(desc["freq"]),
    }


def profile_dataset(
    input_path: str,
    output_path: str,
    project_yaml_path: str | None = None,
) -> dict:
    """Infer schema and compute basic stats. Write profile.json. Return the profile dict."""
    input_path = str(input_path)
    output_path = str(output_path)

    target_column = None
    task_type = None
    if project_yaml_path and Path(project_yaml_path).exists():
        import yaml
        with open(project_yaml_path) as f:
            project = yaml.safe_load(f)
        target_column = project.get("target_column")
        task_type = project.get("task_type")

    df = pd.read_csv(input_path)

    columns = []
    for col_name in df.columns:
        col = df[col_name]
        columns.append({
            "name": col_name,
            "pandas_dtype": str(col.dtype),
            "inferred_semantic_type": _infer_semantic_type(col, col_name, target_column, task_type),
            "sample_values": _sample_values(col),
        })

    basic_stats = {}
    for col_name in df.columns:
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            basic_stats[col_name] = _basic_stats_numeric(col)
        else:
            try:
                basic_stats[col_name] = _basic_stats_categorical(col)
            except Exception:
                basic_stats[col_name] = {"count": int(col.count())}

    profile = {
        "profiler_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": input_path,
            "sha256": _sha256(input_path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "file_size_bytes": Path(input_path).stat().st_size,
        },
        "schema": {"columns": columns},
        "basic_stats": basic_stats,
        "m2_sections_complete": ["schema", "basic_stats"],
        "m2_sections_pending": [
            "null_analysis",
            "cardinality",
            "outliers",
            "correlation",
            "target_validation",
            "leakage_flags",
        ],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2, cls=_SafeEncoder, allow_nan=False)

    return profile


def main():
    parser = argparse.ArgumentParser(description="M2.1 schema and basic stats profiler")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to write profile.json")
    parser.add_argument("--project", default=None, help="Path to project.yaml (optional)")
    args = parser.parse_args()

    try:
        profile = profile_dataset(args.input, args.output, args.project)
        row_count = profile["source"]["row_count"]
        col_count = profile["source"]["column_count"]
        print(f"Profile written to {args.output} ({row_count} rows, {col_count} columns)")
    except Exception as e:
        import traceback
        error_payload = {
            "error": True,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(error_payload, f, indent=2)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

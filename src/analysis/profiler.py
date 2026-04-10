import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "0.2.0"


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


def _describe_column(col_name: str, inferred_type: str, col: pd.Series) -> str:
    unique = col.nunique(dropna=True)
    sample = col.dropna().head(3).tolist()
    sample_str = ", ".join(str(v) for v in sample)
    nullable = ", nullable" if col.isna().any() else ""
    label = inferred_type.replace("_", " ")
    return f"{col_name}: {label}{nullable}, {unique} unique values. Sample: [{sample_str}]."


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
        "skewness": round(float(col.skew()), 4) if col.count() > 0 else None,
    }


def _basic_stats_categorical(col: pd.Series) -> dict:
    desc = col.describe()
    return {
        "count": int(desc["count"]),
        "unique": int(desc["unique"]),
        "top": desc["top"],
        "freq": int(desc["freq"]),
    }


def _top_pairs(matrix: dict, n: int = 10) -> list[dict]:
    cols = list(matrix.keys())
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({"col_a": a, "col_b": b, "value": round(matrix[a][b], 4)})
    pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
    return pairs[:n]


def _pearson_correlation(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        return {"matrix": {}, "top_pairs": [], "note": "fewer than 2 numeric columns"}
    corr_matrix = numeric_df.corr(method="pearson")
    matrix = {
        col: {other: round(float(corr_matrix.loc[col, other]), 4) for other in corr_matrix.columns}
        for col in corr_matrix.columns
    }
    return {"matrix": matrix, "top_pairs": _top_pairs(matrix)}


def _cramers_v_single(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = int(ct.values.sum())
    r, k = ct.shape
    denom = min(r - 1, k - 1)
    return float(np.sqrt(chi2 / n / denom)) if denom > 0 and n > 0 else 0.0


def _cramers_v_correlation(df: pd.DataFrame) -> dict:
    cat_df = df.select_dtypes(include=["object", "category", "string"])
    # skip very high-cardinality columns to avoid massive crosstabs
    cat_df = cat_df[[c for c in cat_df.columns if cat_df[c].nunique() <= 50]]
    cols = list(cat_df.columns)
    if len(cols) < 2:
        return {"matrix": {}, "top_pairs": [], "note": "fewer than 2 categorical columns"}

    # compute upper triangle, fill matrix symmetrically
    computed: dict[tuple[str, str], float] = {}
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            v = round(_cramers_v_single(cat_df[a], cat_df[b]), 4)
            computed[(a, b)] = v
            computed[(b, a)] = v

    matrix = {
        a: {b: (1.0 if a == b else computed.get((a, b), 0.0)) for b in cols}
        for a in cols
    }
    return {"matrix": matrix, "top_pairs": _top_pairs(matrix)}


def _correlation(df: pd.DataFrame) -> dict:
    return {
        "pearson": _pearson_correlation(df),
        "cramers_v": _cramers_v_correlation(df),
    }


def _target_validation(
    df: pd.DataFrame,
    target_column: str | None,
    task_type: str | None,
) -> dict:
    if not target_column or target_column not in df.columns:
        return {"skipped": True, "reason": "no target_column configured"}

    col = df[target_column]
    result: dict = {
        "target_column": target_column,
        "null_count": int(col.isna().sum()),
    }

    if task_type == "binary_classification":
        counts = {str(k): int(v) for k, v in col.value_counts().to_dict().items()}
        sorted_vals = sorted(counts.values())
        minority, majority = sorted_vals[0], sorted_vals[-1]
        result["class_counts"] = counts
        result["class_balance_ratio"] = round(minority / majority, 4) if majority > 0 else None
        result["is_imbalanced"] = (minority / majority < 0.2) if majority > 0 else False
    elif task_type == "regression":
        result["min"] = float(col.min())
        result["max"] = float(col.max())
        result["mean"] = float(col.mean())
        result["std"] = float(col.std())

    return result


def _leakage_flags(df: pd.DataFrame, target_column: str | None) -> dict:
    if not target_column or target_column not in df.columns:
        return {"skipped": True, "reason": "no target_column configured"}

    target = df[target_column]
    if not pd.api.types.is_numeric_dtype(target):
        return {"skipped": True, "reason": "target column is not numeric"}

    numeric_df = df.select_dtypes(include="number")
    flagged = []
    for col_name in numeric_df.columns:
        if col_name == target_column:
            continue
        corr = numeric_df[col_name].corr(target)
        if pd.notna(corr) and abs(corr) > 0.95:
            flagged.append({"column": col_name, "correlation_with_target": round(float(corr), 4)})

    return {"threshold": 0.95, "flagged_columns": flagged}


def _near_duplicate_pairs(df: pd.DataFrame, threshold: float = 0.98) -> list[dict]:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr(method="pearson")
    cols = corr.columns.tolist()
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append({"col_a": a, "col_b": b, "pearson_r": round(float(v), 4)})
    return pairs


def _dataset_description(input_path: str, df: pd.DataFrame, project: dict | None) -> str:
    """Heuristic one-line description for the dataset. Uses project.yaml if available."""
    if project and project.get("description"):
        return str(project["description"])
    filename = Path(input_path).stem.replace("_", " ").replace("-", " ")
    rows, cols = df.shape
    parts = [f"{filename}: {rows} rows, {cols} columns."]
    target = project.get("target_column") if project else None
    task = project.get("task_type") if project else None
    if target and task:
        parts.append(f"{task.replace('_', ' ')} on '{target}'.")
    elif target:
        parts.append(f"Target: '{target}'.")
    return " ".join(parts)


def _feature_risk_flags(columns: list[dict], df: pd.DataFrame) -> dict:
    flagged = [
        {"column": col["name"], "flags": col["risk_flags"]}
        for col in columns if col.get("risk_flags")
    ]
    near_dups = _near_duplicate_pairs(df)
    return {"flagged_columns": flagged, "near_duplicate_pairs": near_dups}


def _mutual_information(
    df: pd.DataFrame,
    target_column: str | None,
    task_type: str | None,
) -> dict:
    if not target_column or target_column not in df.columns:
        return {"skipped": True, "reason": "no target_column configured"}

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression  # noqa: PLC0415
    from sklearn.preprocessing import OrdinalEncoder  # noqa: PLC0415

    X_frames: list[pd.Series] = []
    discrete_mask: list[bool] = []
    included_cols: list[str] = []

    for col_name in df.columns:
        if col_name == target_column:
            continue
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            X_frames.append(col.fillna(col.median()).rename(col_name))
            discrete_mask.append(False)
            included_cols.append(col_name)
        elif col.nunique() <= 50:
            # ordinal-encode low-cardinality categoricals; treat NaN as its own category
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            encoded = enc.fit_transform(col.fillna("__missing__").values.reshape(-1, 1)).flatten()
            X_frames.append(pd.Series(encoded, name=col_name))
            discrete_mask.append(True)
            included_cols.append(col_name)
        # skip high-cardinality text columns (Name, Ticket etc.)

    if not X_frames:
        return {"skipped": True, "reason": "no encodable feature columns"}

    X = pd.concat(X_frames, axis=1)
    y = df[target_column]

    if task_type in ("binary_classification", "classification"):
        mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)
    else:
        mi = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)

    scores = sorted(
        [{"column": c, "mi_score": round(float(v), 4)} for c, v in zip(included_cols, mi)],
        key=lambda x: -x["mi_score"],
    )
    return {"scores": scores}


def _build_column(
    col: pd.Series,
    col_name: str,
    n_rows: int,
    target_column: str | None,
    task_type: str | None,
) -> dict:
    inferred_type = _infer_semantic_type(col, col_name, target_column, task_type)
    entry: dict = {
        "name": col_name,
        "pandas_dtype": str(col.dtype),
        "inferred_semantic_type": inferred_type,
        "description": _describe_column(col_name, inferred_type, col),
        "sample_values": _sample_values(col),
    }

    # basic_stats
    if pd.api.types.is_numeric_dtype(col):
        entry["basic_stats"] = _basic_stats_numeric(col)
    else:
        try:
            entry["basic_stats"] = _basic_stats_categorical(col)
        except Exception:
            entry["basic_stats"] = {"count": int(col.count())}

    # null_analysis
    null_count = int(col.isna().sum())
    entry["null_analysis"] = {
        "null_count": null_count,
        "null_pct": round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0,
        "is_nullable": null_count > 0,
    }

    # cardinality
    n_unique = int(col.nunique(dropna=True))
    entry["cardinality"] = {
        "unique_count": n_unique,
        "uniqueness_ratio": round(n_unique / n_rows, 4) if n_rows > 0 else 0.0,
        "is_high_cardinality": n_unique > 20,
    }

    # outliers — numeric only
    if pd.api.types.is_numeric_dtype(col):
        non_null = col.dropna()
        q1 = float(non_null.quantile(0.25))
        q3 = float(non_null.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((non_null < lower) | (non_null > upper)).sum())
        entry["outliers"] = {
            "method": "iqr",
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": outlier_count,
            "outlier_pct": round(outlier_count / len(non_null) * 100, 2) if len(non_null) > 0 else 0.0,
        }

    # risk_flags
    risk_flags: list[str] = []
    if pd.api.types.is_numeric_dtype(col):
        non_null = col.dropna()
        if col.nunique(dropna=True) <= 1:
            risk_flags.append("zero_variance")
        elif non_null.count() > 0 and abs(float(non_null.skew())) > 1.0:
            risk_flags.append("high_skew")
    entry["risk_flags"] = risk_flags

    return entry


def profile_dataset(
    input_path: str,
    output_path: str,
    project_yaml_path: str | None = None,
) -> dict:
    """Profile a CSV dataset. Write profile.json. Return the profile dict."""
    input_path = str(input_path)
    output_path = str(output_path)

    target_column = None
    task_type = None
    project: dict | None = None
    if project_yaml_path and Path(project_yaml_path).exists():
        import yaml
        with open(project_yaml_path) as f:
            project = yaml.safe_load(f) or {}
        target_column = project.get("target_column")
        task_type = project.get("task_type")

    df = pd.read_csv(input_path)
    n_rows = len(df)

    columns = [
        _build_column(df[col_name], col_name, n_rows, target_column, task_type)
        for col_name in df.columns
    ]

    profile = {
        "profiler_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": input_path,
            "description": _dataset_description(input_path, df, project),
            "sha256": _sha256(input_path),
            "row_count": n_rows,
            "column_count": len(df.columns),
            "file_size_bytes": Path(input_path).stat().st_size,
        },
        "columns": columns,
        "correlation": _correlation(df),
        "target_validation": _target_validation(df, target_column, task_type),
        "leakage_flags": _leakage_flags(df, target_column),
        "feature_risk_flags": _feature_risk_flags(columns, df),
        "mutual_information": _mutual_information(df, target_column, task_type),
        "m2_sections_complete": [
            "columns", "correlation", "target_validation", "leakage_flags",
            "feature_risk_flags", "mutual_information",
        ],
        "m2_sections_pending": [],
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(profile, f, indent=2, cls=_SafeEncoder, allow_nan=False)

    return profile


def main():
    parser = argparse.ArgumentParser(description="M2 dataset profiler")
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

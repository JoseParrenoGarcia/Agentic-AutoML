# Plan: M2 Gap Closure — Semantic Layer, Risk Flags, Categorical Correlation

## Context

A cross-session review of M2 against the PRD (section 6.1) identified four gaps that must be closed before M3 (the Planner agent) can be built:

1. **Semantic layer absent** — PRD calls column descriptions a "first-class output". Without them, the Planner reads `Parch: continuous_numeric` with no domain context.
2. **Feature-risk flags absent** — PRD explicitly lists `(skew, zero-variance, near-duplicates)` as a profiler responsibility.
3. **Categorical correlation absent** — `_correlation()` is Pearson-only. Categorical associations (e.g. Sex vs Pclass on Titanic) are invisible.
4. **Correlation output is LLM-hostile** — full matrix is noise; planners need a ranked `top_pairs` list.

Nice-to-have: mutual information with target (sklearn, 5 lines).

---

## Files to Modify

| File | Change |
|------|--------|
| `src/analysis/profiler.py` | Core changes (5 additions) |
| `src/analysis/plots.py` | Update correlation key path |
| `tests/analysis/test_profiler.py` | Update broken tests + add new tests |
| `tests/analysis/test_profiler_edge_cases.py` | Add zero_variance flag test |
| `requirements.txt` | Add `scikit-learn>=1.3.0` (for MI only — defer if MI deferred) |

---

## Implementation

### 1. `_build_column()` — add `description` field

Heuristic, deterministic sentence. No LLM call; downstream agents can improve it.

```python
def _describe_column(col_name: str, inferred_type: str, col: pd.Series) -> str:
    unique = col.nunique(dropna=True)
    sample = col.dropna().head(3).tolist()
    sample_str = ", ".join(str(v) for v in sample)
    nullable = ", nullable" if col.isna().any() else ""
    label = inferred_type.replace("_", " ")
    return f"{col_name}: {label}{nullable}, {unique} unique values. Sample: [{sample_str}]."
```

Add to `_build_column()` as the first field after `inferred_semantic_type`:
```python
entry["description"] = _describe_column(col_name, entry["inferred_semantic_type"], col)
```

### 2. `_basic_stats_numeric()` — add `skewness`

Add one line inside the return dict:
```python
"skewness": round(float(col.skew()), 4) if col.count() > 0 else None,
```

### 3. `_build_column()` — add `risk_flags` list

After the `outliers` block:
```python
risk_flags: list[str] = []
if pd.api.types.is_numeric_dtype(col):
    non_null = col.dropna()
    if col.nunique(dropna=True) <= 1:
        risk_flags.append("zero_variance")
    elif non_null.count() > 0 and abs(float(non_null.skew())) > 1.0:
        risk_flags.append("high_skew")
entry["risk_flags"] = risk_flags
```

### 4. Restructure `_correlation()` — Pearson + Cramér's V + `top_pairs`

New shape:
```json
{
    "pearson":   { "matrix": {...}, "top_pairs": [{col_a, col_b, value}, ...] },
    "cramers_v": { "matrix": {...}, "top_pairs": [{col_a, col_b, value}, ...] }
}
```

Shared helper:
```python
def _top_pairs(matrix: dict, n: int = 10) -> list[dict]:
    cols = list(matrix.keys())
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append({"col_a": a, "col_b": b, "value": round(matrix[a][b], 4)})
    pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
    return pairs[:n]
```

Cramér's V helper (`scipy.stats` already in requirements.txt):
```python
from scipy import stats  # add to imports

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = int(ct.values.sum())
    r, k = ct.shape
    denom = min(r - 1, k - 1)
    return float(np.sqrt(chi2 / n / denom)) if denom > 0 and n > 0 else 0.0
```

Categorical columns: `select_dtypes(include=["object", "category"])` AND `nunique() <= 50`.

### 5. Add `_feature_risk_flags()` top-level section

```python
def _feature_risk_flags(columns: list[dict], df: pd.DataFrame) -> dict:
    flagged = [
        {"column": col["name"], "flags": col["risk_flags"]}
        for col in columns if col.get("risk_flags")
    ]
    near_dups = _near_duplicate_pairs(df, threshold=0.98)
    return {"flagged_columns": flagged, "near_duplicate_pairs": near_dups}

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
```

Add to `profile_dataset()` return dict:
```python
"feature_risk_flags": _feature_risk_flags(columns, df),
```

Update sentinel:
```python
"m2_sections_complete": [
    "columns", "correlation", "target_validation", "leakage_flags", "feature_risk_flags",
],
```

### 6. Mutual information with target

Add `_mutual_information()` using `sklearn.feature_selection`. Skips when no target or no numeric features. Adds `mutual_information` to `m2_sections_complete`.

```python
def _mutual_information(df: pd.DataFrame, target_column: str | None, task_type: str | None) -> dict:
    if not target_column or target_column not in df.columns:
        return {"skipped": True, "reason": "no target_column configured"}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_column]
    if not feature_cols:
        return {"skipped": True, "reason": "no numeric feature columns"}

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target_column]

    if task_type in ("binary_classification", "classification"):
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    else:
        mi = mutual_info_regression(X, y, discrete_features=False, random_state=42)

    scores = sorted(
        [{"column": c, "mi_score": round(float(v), 4)} for c, v in zip(feature_cols, mi)],
        key=lambda x: -x["mi_score"],
    )
    return {"scores": scores}
```

Add to `profile_dataset()` return dict and update `m2_sections_complete`:
```python
"mutual_information": _mutual_information(df, target_column, task_type),
"m2_sections_complete": [
    "columns", "correlation", "target_validation", "leakage_flags",
    "feature_risk_flags", "mutual_information",
],
```

Add `scikit-learn>=1.3.0` to `requirements.txt`.

---

## `src/analysis/plots.py` — one-line fix

```python
# Before:
matrix = profile["correlation"]["matrix"]
# After:
matrix = profile["correlation"]["pearson"]["matrix"]
```

---

## Tests

### `tests/analysis/test_profiler.py` — breaking changes to update
- All `profile["correlation"]["matrix"]` accesses → `profile["correlation"]["pearson"]["matrix"]`
- `test_correlation_method_is_pearson` → `test_correlation_has_pearson_section`

### New tests (Titanic-skippable where needed)
- `test_column_has_description` — non-empty string on every column
- `test_numeric_column_has_skewness` — `basic_stats["skewness"]` is float for Fare
- `test_feature_risk_flags_section_exists` — root key with `flagged_columns` + `near_duplicate_pairs`
- `test_correlation_has_cramers_v` — `cramers_v.matrix` non-empty (Sex × Pclass)
- `test_correlation_top_pairs_sorted` — abs(top_pairs[0].value) ≥ abs(top_pairs[1].value)
- `test_m2_sections_includes_feature_risk_flags`

### `tests/analysis/test_profiler_edge_cases.py` — one new test
- `test_constant_has_zero_variance_flag` — `edge_schema["constant"]["risk_flags"] == ["zero_variance"]`

---

## Verification

```bash
source .venv/bin/activate
python -m pytest tests/ -v

# spot-check (if Titanic data present)
python -m src.analysis.profiler --input projects/titanic/data/raw/train.csv \
  --output /tmp/profile_test.json
python -c "
import json; p = json.load(open('/tmp/profile_test.json'))
print('description:', p['columns'][0]['description'])
print('risk_flags:', [c for c in p['columns'] if c['risk_flags']])
print('feature_risk_flags:', p['feature_risk_flags'])
print('top pearson pair:', p['correlation']['pearson']['top_pairs'][:2])
print('cramers_v matrix keys:', list(p['correlation']['cramers_v']['matrix'].keys()))
"
```

"""
M2.4 — Plot generation.

Reads profile.json (for metadata) and the raw CSV (for data).
Writes PNGs to the output directory:
  - dist_<col>.png          — distribution per column
  - correlation_heatmap.png — Pearson heatmap (numeric columns)
  - target_vs_<col>.png     — target vs each feature (when target is configured)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in CI and agent calls
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (8, 5)
DPI = 120
MAX_CATEGORIES = 15  # cap bar charts to top N categories


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def _dist_numeric(col: pd.Series, col_name: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        col.dropna().plot.hist(bins=30, ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution — {col_name}")
    path = out_dir / f"dist_{col_name}.png"
    _save(fig, path)
    return path


def _dist_categorical(col: pd.Series, col_name: str, out_dir: Path) -> Path:
    counts = col.value_counts().head(MAX_CATEGORIES)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        counts.plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
        ax.invert_yaxis()
        ax.set_xlabel("Count")
        ax.set_title(f"Distribution — {col_name} (top {len(counts)})")
    path = out_dir / f"dist_{col_name}.png"
    _save(fig, path)
    return path


def plot_distributions(df: pd.DataFrame, profile: dict, out_dir: Path) -> list[Path]:
    col_meta = {c["name"]: c for c in profile["columns"]}
    paths = []
    for col_name in df.columns:
        meta = col_meta.get(col_name, {})
        semantic = meta.get("inferred_semantic_type", "")
        if semantic == "identifier":
            continue  # distributions of ID columns carry no signal
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            paths.append(_dist_numeric(col, col_name, out_dir))
        else:
            paths.append(_dist_categorical(col, col_name, out_dir))
    return paths


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(profile: dict, out_dir: Path) -> Path | None:
    matrix = profile["correlation"]["pearson"]["matrix"]
    if not matrix:
        return None
    cols = list(matrix.keys())
    data = [[matrix[r][c] for c in cols] for r in cols]
    corr_df = pd.DataFrame(data, index=cols, columns=cols)

    n = len(cols)
    fig_size = max(6, n * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        corr_df,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap (Pearson)", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    path = out_dir / "correlation_heatmap.png"
    _save(fig, path)
    return path


# ---------------------------------------------------------------------------
# Target vs feature plots
# ---------------------------------------------------------------------------

def _target_vs_numeric(
    df: pd.DataFrame,
    feature: str,
    target: str,
    task_type: str,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        if task_type == "binary_classification":
            classes = sorted(df[target].dropna().unique())
            palette = sns.color_palette("Set2", len(classes))
            for cls, color in zip(classes, palette):
                subset = df.loc[df[target] == cls, feature].dropna()
                subset.plot.kde(ax=ax, label=str(cls), color=color, linewidth=2)
            ax.set_xlabel(feature)
            ax.set_ylabel("Density")
            ax.set_title(f"{target} vs {feature}")
            ax.legend(title=target)
        else:
            ax.scatter(df[feature], df[target], alpha=0.4, s=20, color="#4C72B0")
            ax.set_xlabel(feature)
            ax.set_ylabel(target)
            ax.set_title(f"{target} vs {feature}")
    path = out_dir / f"target_vs_{feature}.png"
    _save(fig, path)
    return path


def _target_vs_categorical(
    df: pd.DataFrame,
    feature: str,
    target: str,
    task_type: str,
    out_dir: Path,
) -> Path:
    top_cats = df[feature].value_counts().head(MAX_CATEGORIES).index
    subset = df[df[feature].isin(top_cats)].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    with plt.style.context(STYLE):
        if task_type == "binary_classification":
            # survival rate per category
            rate = subset.groupby(feature)[target].mean().reindex(top_cats)
            rate.sort_values().plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
            ax.invert_yaxis()
            ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.set_xlabel(f"{target} rate")
            ax.set_title(f"{target} rate by {feature}")
        else:
            subset.boxplot(column=target, by=feature, ax=ax, rot=45)
            ax.set_title(f"{target} by {feature}")
            plt.suptitle("")
    path = out_dir / f"target_vs_{feature}.png"
    _save(fig, path)
    return path


def plot_target_vs_features(
    df: pd.DataFrame,
    profile: dict,
    out_dir: Path,
) -> list[Path]:
    tv = profile.get("target_validation", {})
    if tv.get("skipped"):
        return []

    target = tv["target_column"]
    task_type = profile.get("target_validation", {}).get("target_column")
    # derive task_type from m2_sections_complete context — read from column metadata
    col_meta = {c["name"]: c for c in profile["columns"]}
    target_semantic = col_meta.get(target, {}).get("inferred_semantic_type", "")
    if target_semantic == "binary_target":
        task_type = "binary_classification"
    elif target_semantic == "regression_target":
        task_type = "regression"
    else:
        task_type = "unknown"

    paths = []
    for col_name in df.columns:
        if col_name == target:
            continue
        meta = col_meta.get(col_name, {})
        semantic = meta.get("inferred_semantic_type", "")
        if semantic == "identifier":
            continue
        col = df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            paths.append(_target_vs_numeric(df, col_name, target, task_type, out_dir))
        else:
            paths.append(_target_vs_categorical(df, col_name, target, task_type, out_dir))
    return paths


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_plots(
    input_path: str,
    profile_path: str,
    output_dir: str,
) -> dict:
    """Generate all plots. Returns a summary dict with paths of files written."""
    df = pd.read_csv(input_path)
    with open(profile_path) as f:
        profile = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_paths = plot_distributions(df, profile, out_dir)
    heatmap_path = plot_correlation_heatmap(profile, out_dir)
    target_paths = plot_target_vs_features(df, profile, out_dir)

    written = [str(p) for p in dist_paths + target_paths if p]
    if heatmap_path:
        written.append(str(heatmap_path))

    return {"plots_written": len(written), "paths": sorted(written)}


def main():
    parser = argparse.ArgumentParser(description="M2.4 plot generation")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--profile", required=True, help="Path to profile.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write plots")
    args = parser.parse_args()

    try:
        result = generate_plots(args.input, args.profile, args.output_dir)
        print(f"Plots written: {result['plots_written']} → {args.output_dir}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

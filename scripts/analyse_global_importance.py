"""
Global SHAP feature importance analysis across architectures and regimes.

Generates:
- Cross-architecture SHAP comparison heatmaps
- Per-architecture feature importance rankings
- Correlation heatmaps between fuel_norm and other key features

Notes:
- Expects SHAP CSVs in data/shap named like:
    shap_{algo}_{regime}_seed={seed}.csv
  e.g.:
    shap_ppo_safe_seed=0.csv
    shap_a2c_rulebook_seed=2.csv
    shap_dqn_unconstrained_seed=1.csv

- Tries to save charts as PNG first.
- If static image export fails (e.g. kaleido missing), falls back to HTML.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SHAP_DIR = Path("data/shap")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FILENAME_PATTERN = re.compile(
    r"^shap_(?P<algo>.+?)_(?P<regime>.+?)_seed=(?P<seed>\d+)$"
)


def parse_shap_filename(path: Path) -> dict:
    """
    Parse filenames of the form:
        shap_{algo}_{regime}_seed={seed}.csv

    Returns a dict with algo, regime, seed.
    """
    stem = path.stem
    match = FILENAME_PATTERN.match(stem)
    if not match:
        raise ValueError(
            f"Could not parse SHAP filename '{path.name}'. "
            "Expected format: shap_{algo}_{regime}_seed={seed}.csv"
        )

    return {
        "algo": match.group("algo"),
        "regime": match.group("regime"),
        "seed": int(match.group("seed")),
    }


def write_metadata(base_path: Path, caption: str, description: str):
    """Write chart metadata JSON alongside output artifact."""
    meta_path = base_path.with_suffix(base_path.suffix + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "caption": caption,
                "description": description,
            },
            f,
            indent=2,
        )


def save_figure(fig, base_name: str, caption: str, description: str) -> Path:
    """
    Try saving a figure as PNG first.
    If that fails, save as interactive HTML instead.

    Returns the actual output path used.
    """
    png_path = OUTPUT_DIR / f"{base_name}.png"
    html_path = OUTPUT_DIR / f"{base_name}.html"

    try:
        fig.write_image(str(png_path))
        write_metadata(png_path, caption, description)
        print(f"✓ Saved PNG: {png_path}")
        return png_path
    except Exception as e:
        print(f"  ⚠ PNG export failed for {base_name}: {e}")
        print("  → Falling back to interactive HTML export")
        fig.write_html(str(html_path))
        write_metadata(html_path, caption, description)
        print(f"✓ Saved HTML: {html_path}")
        return html_path


def load_all_shap() -> pd.DataFrame:
    """Load all SHAP CSVs and combine them into a single DataFrame."""
    files = sorted(SHAP_DIR.glob("shap_*.csv"))
    if not files:
        raise RuntimeError("No SHAP CSVs found in data/shap")

    dfs = []
    for f in files:
        parsed = parse_shap_filename(f)
        df = pd.read_csv(f)

        required_cols = {"feature", "mean_abs_shap"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"{f.name} is missing required columns: {sorted(missing)}. "
                f"Found columns: {list(df.columns)}"
            )

        df["algo"] = parsed["algo"]
        df["regime"] = parsed["regime"]
        df["seed"] = parsed["seed"]
        df["source_file"] = f.name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def create_cross_architecture_heatmap(shap_all: pd.DataFrame) -> pd.DataFrame:
    """Create heatmap showing feature importance across all architectures and regimes."""
    pivot = shap_all.pivot_table(
        index="feature",
        columns=["algo", "regime"],
        values="mean_abs_shap",
        aggfunc="mean",
    )

    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop("total", axis=1)

    top_features = pivot.head(15)

    fig = go.Figure(
        data=go.Heatmap(
            z=top_features.values,
            x=[f"{algo}_{regime}" for algo, regime in top_features.columns],
            y=top_features.index.tolist(),
            colorscale="Viridis",
            text=np.round(top_features.values, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            colorbar=dict(title="Mean |SHAP|"),
        )
    )

    fig.update_layout(
        title="Feature Importance Heatmap Across Architectures & Regimes (Top 15)",
        xaxis_title="Architecture_Regime",
        yaxis_title="State Feature",
        height=700,
        width=1400,
        font=dict(size=10),
    )
    fig.update_xaxes(tickangle=45)

    save_figure(
        fig,
        base_name="chart_shap_cross_architecture_heatmap",
        caption="Feature importance heatmap across all architectures and regimes",
        description=(
            "Heatmap showing mean absolute SHAP values for the top 15 features "
            "across all algorithm-regime combinations. Darker colors indicate "
            "higher feature importance."
        ),
    )

    top_features.to_csv(OUTPUT_DIR / "shap_cross_architecture_matrix.csv")
    print("✓ Saved CSV: output/shap_cross_architecture_matrix.csv")
    return top_features


def create_per_architecture_rankings(shap_all: pd.DataFrame):
    """Create bar charts showing top features for each architecture."""
    algos = sorted(shap_all["algo"].unique())

    for algo in algos:
        algo_data = shap_all[shap_all["algo"] == algo].copy()

        avg = (
            algo_data.groupby("feature")["mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
        )

        fig = px.bar(
            x=avg.values,
            y=avg.index,
            orientation="h",
            title=f"Top 12 Feature Importances: {algo.upper()} (avg across regimes)",
            labels={"x": "Mean |SHAP|", "y": "Feature"},
        )

        fig.update_layout(
            height=500,
            width=800,
            yaxis=dict(autorange="reversed"),
        )

        save_figure(
            fig,
            base_name=f"chart_shap_{algo}_top12",
            caption=f"Top 12 features for {algo.upper()} architecture",
            description=(
                f"Horizontal bar chart showing the most important state features "
                f"for {algo.upper()}, averaged across all reward regimes."
            ),
        )


def create_fuel_correlation_heatmaps(shap_all: pd.DataFrame):
    """
    Create correlation views for fuel_norm and other features.

    These are correlations of SHAP importance patterns across regimes/seeds,
    not raw state-value correlations from the simulator.
    """
    algos = sorted(shap_all["algo"].unique())

    for algo in algos:
        algo_data = shap_all[shap_all["algo"] == algo].copy()

        pivot = algo_data.pivot_table(
            index="feature",
            columns=["regime", "seed"],
            values="mean_abs_shap",
            aggfunc="mean",
        )

        if pivot.shape[1] < 2:
            print(f"  ⚠ Skipping fuel correlation for {algo.upper()}: not enough columns")
            continue

        corr = pivot.T.corr()

        if "fuel_norm" in corr.index:
            fuel_corr = corr["fuel_norm"].drop("fuel_norm").sort_values(ascending=False)
            top_corr = fuel_corr.head(12)

            fig = px.bar(
                x=top_corr.values,
                y=top_corr.index,
                orientation="h",
                title=f"Features Most Correlated with fuel_norm: {algo.upper()}",
                labels={"x": "SHAP Correlation", "y": "Feature"},
                color=top_corr.values,
                color_continuous_scale="RdBu_r",
            )

            fig.update_layout(
                height=500,
                width=850,
                yaxis=dict(autorange="reversed"),
                showlegend=False,
            )

            save_figure(
                fig,
                base_name=f"chart_fuel_correlation_{algo}",
                caption=f"Fuel-norm feature correlation for {algo.upper()}",
                description=(
                    f"Bar chart showing features whose SHAP importance patterns "
                    f"correlate most strongly with fuel_norm for {algo.upper()}. "
                    f"Positive correlation means they tend to rise or fall together."
                ),
            )
        else:
            print(f"  ⚠ fuel_norm not found in {algo.upper()} data")

    if "ppo" in algos:
        ppo_data = shap_all[shap_all["algo"] == "ppo"].copy()
        pivot = ppo_data.pivot_table(
            index="feature",
            columns=["regime", "seed"],
            values="mean_abs_shap",
            aggfunc="mean",
        )

        if pivot.shape[1] >= 2:
            corr = pivot.T.corr()

            key_features = [
                "fuel_norm",
                "tyre_age_norm",
                "tyre_wear_norm",
                "lap_fraction",
                "race_time_norm",
                "track_status_GREEN",
                "track_status_YELLOW",
                "risk_indicator",
                "pit_count",
                "gap_ahead_norm",
                "gap_behind_norm",
            ]

            existing_features = [f for f in key_features if f in corr.index]

            if len(existing_features) > 1:
                subset_corr = corr.loc[existing_features, existing_features]

                fig = go.Figure(
                    data=go.Heatmap(
                        z=subset_corr.values,
                        x=subset_corr.columns.tolist(),
                        y=subset_corr.index.tolist(),
                        colorscale="RdBu_r",
                        zmid=0,
                        text=np.round(subset_corr.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 9},
                        colorbar=dict(title="Correlation"),
                    )
                )

                fig.update_layout(
                    title="Feature SHAP Correlation Matrix: PPO (Key Features)",
                    xaxis_title="Feature",
                    yaxis_title="Feature",
                    height=700,
                    width=850,
                    font=dict(size=10),
                )
                fig.update_xaxes(tickangle=45)

                save_figure(
                    fig,
                    base_name="chart_feature_correlation_matrix_ppo",
                    caption="Feature SHAP correlation matrix for PPO",
                    description=(
                        "Correlation heatmap showing how feature importances co-vary "
                        "across reward regimes and seeds for PPO. Red indicates "
                        "positive correlation, blue indicates negative correlation."
                    ),
                )

                subset_corr.to_csv(OUTPUT_DIR / "feature_correlation_matrix_ppo.csv")
                print("✓ Saved CSV: output/feature_correlation_matrix_ppo.csv")


def create_regime_comparison_charts(shap_all: pd.DataFrame):
    """Create grouped bar charts comparing feature importance across regimes."""
    algos = sorted(shap_all["algo"].unique())

    for algo in algos:
        algo_data = shap_all[shap_all["algo"] == algo].copy()

        top_features = (
            algo_data.groupby("feature")["mean_abs_shap"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .index
        )

        filtered = algo_data[algo_data["feature"].isin(top_features)]

        regime_avg = (
            filtered.groupby(["feature", "regime"])["mean_abs_shap"]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            regime_avg,
            x="feature",
            y="mean_abs_shap",
            color="regime",
            barmode="group",
            title=f"Feature Importance by Regime: {algo.upper()} (Top 10 Features)",
            labels={"mean_abs_shap": "Mean |SHAP|", "feature": "Feature"},
        )

        fig.update_layout(
            height=500,
            width=1000,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
            ),
        )
        fig.update_xaxes(tickangle=45)

        save_figure(
            fig,
            base_name=f"chart_regime_comparison_{algo}",
            caption=f"Regime comparison for {algo.upper()}",
            description=(
                f"Grouped bar chart comparing feature importances across reward "
                f"regimes for {algo.upper()}'s top 10 features."
            ),
        )


def create_summary_statistics(shap_all: pd.DataFrame):
    """Generate summary CSV files."""
    overall_ranking = (
        shap_all.groupby("feature")["mean_abs_shap"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean", ascending=False)
    )
    overall_ranking.to_csv(OUTPUT_DIR / "shap_overall_feature_ranking.csv")
    print("✓ Saved CSV: output/shap_overall_feature_ranking.csv")

    arch_ranking = (
        shap_all.groupby(["algo", "feature"])["mean_abs_shap"]
        .mean()
        .reset_index()
        .sort_values(["algo", "mean_abs_shap"], ascending=[True, False])
    )
    arch_ranking.to_csv(OUTPUT_DIR / "shap_per_architecture_ranking.csv", index=False)
    print("✓ Saved CSV: output/shap_per_architecture_ranking.csv")

    regime_ranking = (
        shap_all.groupby(["regime", "feature"])["mean_abs_shap"]
        .mean()
        .reset_index()
        .sort_values(["regime", "mean_abs_shap"], ascending=[True, False])
    )
    regime_ranking.to_csv(OUTPUT_DIR / "shap_per_regime_ranking.csv", index=False)
    print("✓ Saved CSV: output/shap_per_regime_ranking.csv")

    by_file = (
        shap_all.groupby(["algo", "regime", "seed", "feature"])["mean_abs_shap"]
        .mean()
        .reset_index()
        .sort_values(["algo", "regime", "seed", "mean_abs_shap"], ascending=[True, True, True, False])
    )
    by_file.to_csv(OUTPUT_DIR / "shap_by_algo_regime_seed.csv", index=False)
    print("✓ Saved CSV: output/shap_by_algo_regime_seed.csv")


def main():
    print("\n" + "=" * 80)
    print("GLOBAL SHAP FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80 + "\n")

    print("Loading SHAP data...")
    shap_all = load_all_shap()

    print(f"  Loaded {len(shap_all)} feature records")
    print(f"  Architectures: {sorted(shap_all['algo'].unique())}")
    print(f"  Regimes: {sorted(shap_all['regime'].unique())}")
    print(f"  Seeds: {sorted(shap_all['seed'].unique())}")
    print(f"  Features: {shap_all['feature'].nunique()} unique")
    print(f"  Source files: {shap_all['source_file'].nunique()}")

    print("\n" + "-" * 80)
    print("Creating visualizations...")
    print("-" * 80 + "\n")

    print("1. Cross-architecture heatmap...")
    create_cross_architecture_heatmap(shap_all)

    print("\n2. Per-architecture rankings...")
    create_per_architecture_rankings(shap_all)

    print("\n3. Fuel correlation analysis...")
    create_fuel_correlation_heatmaps(shap_all)

    print("\n4. Regime comparison charts...")
    create_regime_comparison_charts(shap_all)

    print("\n5. Summary statistics...")
    create_summary_statistics(shap_all)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.absolute()}")
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        if f.suffix in [".png", ".html", ".csv"]:
            print(f"  - {f.name}")


if __name__ == "__main__":
    main()
import pandas as pd
from pathlib import Path
import json
import plotly.express as px

SHAP_DIR = Path("data/shap")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_shap():
    files = list(SHAP_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError("No SHAP CSVs found in data/shap")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Expect columns: feature, mean_abs_shap, maybe algo/regime/seed
        if "algo" not in df.columns or "regime" not in df.columns:
            # infer from filename: shap_{algo}_{regime}_seed=0.csv
            stem = f.stem
            parts = stem.split("_")  # ["shap", algo, regime, "seed=0"]
            algo = parts[1] if len(parts) > 1 else "unknown"
            regime = parts[2] if len(parts) > 2 else "unknown"
            df["algo"] = algo
            df["regime"] = regime
        if "seed" not in df.columns:
            seed = 0
            if "seed=" in f.stem:
                try:
                    seed = int(f.stem.split("seed=")[-1])
                except ValueError:
                    seed = 0
            df["seed"] = seed
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def top_features(shap_all: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if "feature" not in shap_all.columns or "mean_abs_shap" not in shap_all.columns:
        raise RuntimeError("Expected 'feature' and 'mean_abs_shap' columns in SHAP CSVs")

    top_list = []
    for (algo, regime), grp in shap_all.groupby(["algo", "regime"]):
        top = grp.sort_values("mean_abs_shap", ascending=False).head(k).copy()
        top["algo"] = algo
        top["regime"] = regime
        top_list.append(top)

    top_k = pd.concat(top_list, ignore_index=True)
    top_k.to_csv(OUTPUT_DIR / "shap_top10_features.csv", index=False)
    return top_k


def plot_ppo_safe(shap_all: pd.DataFrame):
    mask = (shap_all["algo"] == "ppo") & (
        shap_all["regime"].str.lower().str.contains("safe")
    )
    subset = shap_all[mask]
    if subset.empty:
        print("No PPO safe SHAP data found; skipping plot")
        return

    top = subset.sort_values("mean_abs_shap", ascending=False).head(10)

    fig = px.bar(
        top,
        x="feature",
        y="mean_abs_shap",
        title="Top-10 state features driving PPO decisions (safe regime)",
        labels={"feature": "State feature", "mean_abs_shap": "Mean |SHAP|"},
    )
    fig.update_xaxes(title_text="State feature")
    fig.update_yaxes(title_text="Mean |SHAP|")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )

    # Use write_html instead of write_image to avoid Kaleido dependency
    out_html = OUTPUT_DIR / "chart_shap_ppo_safe.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    with open(OUTPUT_DIR / "chart_shap_ppo_safe.meta.json", "w") as f:
        json.dump(
            {
                "caption": "Top-10 PPO feature importances in safe regime",
                "description": (
                    "Interactive bar chart of mean absolute SHAP values showing the most "
                    "influential state variables for PPO under the safety-constrained "
                    "reward."
                ),
            },
            f,
        )


def main():
    shap_all = load_shap()
    top_k = top_features(shap_all)
    print("Wrote shap_top10_features.csv with shape:", top_k.shape)
    plot_ppo_safe(shap_all)
    print("SHAP analysis complete. Outputs written to:")
    for p in OUTPUT_DIR.glob("*"):
        print(" -", p)


if __name__ == "__main__":
    main()

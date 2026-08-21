import pandas as pd
from pathlib import Path
import json
import plotly.express as px

RESULTS_DIR = Path("data/experiment_results")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_metrics():
    files = list(RESULTS_DIR.glob("*.csv"))
    if not files:
        raise RuntimeError("No metric CSVs found in data/experiment_results")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Infer algo/regime/budget/seed from filename, e.g. ppo_safe_50k_seed0.csv
        stem = f.stem  # e.g. ppo_safe_50k_seed0
        parts = stem.split("_")
        if len(parts) >= 4:
            algo, regime, budget, seed_part = parts[0], parts[1], parts[2], parts[3]
        else:
            algo = parts[0]
            regime = parts[1] if len(parts) > 1 else "unknown"
            budget = parts[2] if len(parts) > 2 else "unknown"
            seed_part = parts[3] if len(parts) > 3 else "seed0"
        df["algo"] = algo
        df["regime"] = regime
        df["steps_or_episodes"] = budget  # e.g. "50k" or "200ep"
        df["seed"] = int(seed_part.replace("seed", "")) if "seed" in seed_part else 0
        df["source_file"] = f.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarise_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    # Use column names from eval CSVs
    metric_cols = [
        "race_time",
        "crashes",
        "catastrophic",
        "pitstops",
    ]
    missing = [c for c in metric_cols if c not in metrics.columns]
    if missing:
        raise RuntimeError(f"Missing expected metric columns: {missing}")

    grouped = (
        metrics.groupby(["algo", "regime", "steps_or_episodes"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten multi-index columns
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    grouped.to_csv(OUTPUT_DIR / "rl_algorithms_summary.csv", index=False)
    return grouped


def plot_crash_vs_algo(summary: pd.DataFrame):
    # Focus on 50k budget for now
    sub = summary[summary["steps_or_episodes"] == "50k"].copy()
    if sub.empty:
        print("No 50k budget rows found for crash-rate plot")
        return

    fig = px.bar(
        sub,
        x="algo",
        y="crashes_mean",
        color="regime",
        barmode="group",
        title="Mean crashes by algorithm and reward regime (50k)",
        labels={"algo": "Algorithm", "crashes_mean": "Crashes per episode"},
    )
    fig.update_xaxes(title_text="Algorithm")
    fig.update_yaxes(title_text="Crashes per episode")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        )
    )

    # Write interactive HTML instead of static image to avoid Kaleido/Chrome dependency
    out_html = OUTPUT_DIR / "chart_crash_rate_algos.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    with open(OUTPUT_DIR / "chart_crash_rate_algos.meta.json", "w") as f:
        json.dump(
            {
                "caption": "Crashes by algorithm and reward regime (50k)",
                "description": (
                    "Interactive bar chart comparing mean crashes per episode across "
                    "PPO, A2C, DQN, SARSA and REINFORCE under the three reward "
                    "regimes at 50k."
                ),
            },
            f,
        )


def plot_race_time_vs_algo(summary: pd.DataFrame):
    sub = summary[summary["steps_or_episodes"] == "50k"].copy()
    if sub.empty:
        print("No 50k budget rows found for race-time plot")
        return

    fig = px.bar(
        sub,
        x="algo",
        y="race_time_mean",
        color="regime",
        barmode="group",
        title="Mean race time by algorithm and reward regime (50k)",
        labels={"algo": "Algorithm", "race_time_mean": "Race time (s)"},
    )
    fig.update_xaxes(title_text="Algorithm")
    fig.update_yaxes(title_text="Race time (s)")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        )
    )

    # Write interactive HTML instead of static image to avoid Kaleido/Chrome dependency
    out_html = OUTPUT_DIR / "chart_race_time_algos.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    with open(OUTPUT_DIR / "chart_race_time_algos.meta.json", "w") as f:
        json.dump(
            {
                "caption": "Race time by algorithm and reward regime (50k)",
                "description": (
                    "Interactive bar chart comparing mean race time across algorithms "
                    "under the three reward regimes at 50k."
                ),
            },
            f,
        )


def main():
    metrics = load_metrics()
    summary = summarise_metrics(metrics)
    plot_crash_vs_algo(summary)
    plot_race_time_vs_algo(summary)
    print("Analysis complete. Outputs written to:")
    for p in OUTPUT_DIR.glob("*"):
        print(" -", p)


if __name__ == "__main__":
    main()

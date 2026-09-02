"""Turn existing SHAP + evaluation artefacts into thesis-ready figures/tables.

READ-ONLY with respect to the environment, training and evaluation pipeline.
This script does not import f1_env, does not construct an environment, does
not retrain or re-evaluate anything. It only reads:

  - outputs/experiments/eval/final-v1/*_steps=200000_eval=100.csv
        (15 files, 300 rows each = 3 seeds x 100 eval episodes; 4500 rows
        total. This is the frozen per-episode evaluation grid. NOTE: the
        task brief refers to this data as "FINAL_45_seed_level.csv" /
        "FINAL_15_aggregate.csv" / "seed_consistency.csv" / a
        "regime_comparisons.csv" -- none of those four filenames exist
        anywhere in this repository (verified by `find` across the whole
        tree). The 15 CSVs above are the real, on-disk artefact that
        matches the described 5x3x3x100=4500-row design, so this script
        treats them as that data and computes the seed-consistency /
        regime-comparison views itself, reproducibly, from those files.)
  - data/shap/shap_{algo}_{regime}_seed={seed}.csv (45 files)
  - output/shap_*.csv (derived SHAP aggregate tables already committed)

The s_0..s_17 feature-name mapping (FEATURE_NAMES, below) is HARD-CODED in
this script, not read from any doc at runtime. It was independently
verified by hand against src/f1_rl_safety/f1_env.py::F1RaceEnv._get_obs()
(see the comment above FEATURE_NAMES) and is consistent with
docs/shap-thesis-notes.md, but this script does not open either
docs/shap-thesis-notes.md or docs/phase-05-geometry-schema.md at runtime,
and geometry is not used to produce any SHAP feature name.

This script regenerates the FIGURES and TABLES below. It does NOT
generate output/thesis-shap-figures/README.md -- that file is
hand-maintained documentation. After regenerating outputs with this
script, the README must be reviewed by hand for consistency with the
regenerated figures/tables.

Outputs (written only under output/thesis-shap-figures/):
  - output/thesis-shap-figures/figures/*.png
  - output/thesis-shap-figures/tables/*.csv
  (output/thesis-shap-figures/README.md is NOT written by this script --
  see above.)

Run (regenerates figures/ and tables/ only; then review README.md by hand):
    PYTHONPATH=src .venv_f1/bin/python scripts/build_thesis_shap_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "outputs/experiments/eval/final-v1"
SHAP_RAW_DIR = REPO_ROOT / "data/shap"
SHAP_AGG_DIR = REPO_ROOT / "output"

OUT_DIR = REPO_ROOT / "output/thesis-shap-figures"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

ALGOS = ["ppo", "a2c", "dqn", "sarsa", "reinforce"]
REGIMES = ["unconstrained", "rulebook", "safe"]
SEEDS = [0, 1, 2]
N_LAPS = 52  # src/f1_rl_safety/f1_env.py F1RaceEnv default n_laps=52

# ---------------------------------------------------------------------------
# Style: matplotlib only (no seaborn installed in .venv_f1). Palette taken
# from the dataviz skill's validated default palette (references/palette.md)
# -- fixed categorical order, one sequential (blue) ramp, blue<->red diverging.
# ---------------------------------------------------------------------------

CAT = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

ALGO_COLOR = {
    "ppo": CAT["blue"],
    "a2c": CAT["orange"],
    "dqn": CAT["aqua"],
    "sarsa": CAT["yellow"],
    "reinforce": CAT["magenta"],
}
REGIME_COLOR = {
    "unconstrained": CAT["blue"],
    "rulebook": CAT["orange"],
    "safe": CAT["aqua"],
}
SEQ_BLUE = "#2a78d6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_PRIMARY,
    "axes.titlecolor": INK_PRIMARY,
    "text.color": INK_PRIMARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.facecolor": SURFACE,
})


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


# ---------------------------------------------------------------------------
# s_0..s_17 -> real feature name mapping.
# HARD-CODED here, not read from any file at runtime. Independently
# verified by hand against src/f1_rl_safety/f1_env.py::F1RaceEnv._get_obs()
# concatenation order (see the quoted source block in README.md), and
# consistent with docs/shap-thesis-notes.md. Geometry
# (docs/phase-05-geometry-schema.md) is not used to produce any of these
# names -- no SHAP feature is a segment/region identifier.
# ---------------------------------------------------------------------------

FEATURE_NAMES = {
    "s_0": "lap_fraction",
    "s_1": "race_time_norm",
    "s_2": "pos_norm",
    "s_3": "gap_ahead",
    "s_4": "gap_behind",
    "s_5": "tyre_age_norm",
    "s_6": "tyre_wear",
    "s_7": "fuel_norm",
    "s_8": "track_status_0",
    "s_9": "track_status_1",
    "s_10": "track_status_2",
    "s_11": "tyre_compound_0",
    "s_12": "tyre_compound_1",
    "s_13": "tyre_compound_2",
    "s_14": "tyre_compound_3",
    "s_15": "tyre_compound_4",
    "s_16": "risk_indicator",
    "s_17": "pit_count_norm",
}


def rename_feature_col(df, col="feature"):
    df = df.copy()
    df[col] = df[col].map(FEATURE_NAMES).fillna(df[col])
    return df


print("=" * 80)
print("STEP 1: Load evaluation grid (outputs/experiments/eval/final-v1)")
print("=" * 80)

eval_files = sorted(EVAL_DIR.glob("*_steps=200000_eval=100.csv"))
eval_frames = [pd.read_csv(f) for f in eval_files]
eval_all = pd.concat(eval_frames, ignore_index=True)
print(f"Loaded {len(eval_files)} files, {len(eval_all)} episode rows "
      f"(expected 15 x 300 = 4500)")
assert len(eval_all) == 4500, "Unexpected row count in evaluation grid"

# ---------------------------------------------------------------------------
# Per (algo, regime, seed) seed-level aggregate metrics
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 2: Per-seed aggregate metrics")
print("=" * 80)

def seed_level_agg(g):
    n = len(g)
    return pd.Series({
        "n_episodes": n,
        "finish_rate": float((~g["terminated_by_crash"]).mean()),
        "crash_rate": float(g["terminated_by_crash"].mean()),
        "catastrophic_rate": float(g["catastrophic"].mean()),
        "mean_completed_laps": float(g["completed_laps"].mean()),
        "mean_lap_fraction": float((g["completed_laps"] / N_LAPS).mean()),
        "mean_risk": float(g["mean_risk"].mean()),
        "mean_pit_stops": float(g["pit_stops"].mean()),
    })

seed_level = (
    eval_all.groupby(["algo", "regime", "seed"], as_index=False)
    .apply(seed_level_agg, include_groups=False)
)
seed_level = seed_level.sort_values(["algo", "regime", "seed"]).reset_index(drop=True)
seed_level.to_csv(TAB_DIR / "table_seed_level_metrics.csv", index=False)
print(seed_level.head(9).to_string(index=False))

# ---------------------------------------------------------------------------
# Per (algo, regime) aggregate across the 3 seeds (mean, std, min, max)
# ---------------------------------------------------------------------------

metrics = ["finish_rate", "crash_rate", "catastrophic_rate", "mean_completed_laps",
           "mean_lap_fraction", "mean_risk", "mean_pit_stops"]

regime_agg = (
    seed_level.groupby(["algo", "regime"])[metrics]
    .agg(["mean", "std", "min", "max"])
)
regime_agg.columns = [f"{m}_{stat}" for m, stat in regime_agg.columns]
regime_agg = regime_agg.reset_index()
regime_agg.insert(2, "n_seeds", 3)
regime_agg.to_csv(TAB_DIR / "table_regime_aggregate_metrics.csv", index=False)

# ---------------------------------------------------------------------------
# Seed-consistency of pairwise regime comparisons, per algorithm
# For each algo and each of the 3 regime pairs, compute the per-seed signed
# delta (regimeB - regimeA) for each metric, and check whether the sign is
# identical across all 3 seeds.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 3: Seed-consistency of regime comparisons")
print("=" * 80)

REGIME_PAIRS = [("unconstrained", "safe"), ("unconstrained", "rulebook"), ("rulebook", "safe")]
CONSISTENCY_METRICS = ["crash_rate", "finish_rate", "mean_risk"]

rows = []
pivot = seed_level.set_index(["algo", "regime", "seed"])

for algo in ALGOS:
    for regime_a, regime_b in REGIME_PAIRS:
        for metric in CONSISTENCY_METRICS:
            deltas = []
            for seed in SEEDS:
                try:
                    va = pivot.loc[(algo, regime_a, seed), metric]
                    vb = pivot.loc[(algo, regime_b, seed), metric]
                except KeyError:
                    va = vb = np.nan
                deltas.append(vb - va)
            deltas = np.array(deltas, dtype=float)
            signs = np.sign(np.round(deltas, 6))
            nonzero_signs = signs[signs != 0]
            sign_consistent = bool(len(set(nonzero_signs)) <= 1) if len(nonzero_signs) else True
            abs_deltas = np.abs(deltas)
            # outlier flag: one seed's |delta| is >= 3x the median of the
            # other two AND the metric would flip sign/near-zero without it
            if len(abs_deltas) == 3:
                order = np.argsort(abs_deltas)
                smallest_two = abs_deltas[order[:2]]
                largest = abs_deltas[order[2]]
                dominated_by_one_seed = bool(
                    largest > 3 * (smallest_two.mean() + 1e-9)
                    and smallest_two.mean() < 0.05
                )
            else:
                dominated_by_one_seed = False
            rows.append({
                "algo": algo,
                "regime_a": regime_a,
                "regime_b": regime_b,
                "comparison": f"{regime_b}_minus_{regime_a}",
                "metric": metric,
                "delta_seed0": deltas[0],
                "delta_seed1": deltas[1],
                "delta_seed2": deltas[2],
                "mean_delta": float(np.nanmean(deltas)),
                "sign_consistent_across_seeds": sign_consistent,
                "outlier_seed_dominated": dominated_by_one_seed,
            })

seed_consistency = pd.DataFrame(rows)
seed_consistency.to_csv(TAB_DIR / "table_seed_consistency.csv", index=False)

n_consistent = seed_consistency["sign_consistent_across_seeds"].sum()
n_total = len(seed_consistency)
print(f"{n_consistent}/{n_total} (algo, comparison, metric) triples are "
      f"sign-consistent across all 3 seeds "
      f"({100 * n_consistent / n_total:.1f}%)")

print("\nFull consistency table:")
with pd.option_context("display.max_rows", None, "display.width", 160):
    print(seed_consistency[["algo", "comparison", "metric", "mean_delta",
                             "sign_consistent_across_seeds",
                             "outlier_seed_dominated"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Crash location: straight vs corner, per algo x regime
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STEP 4: Crash location by segment type (straight vs corner)")
print("=" * 80)

crashed = eval_all[eval_all["terminated_by_crash"]].copy()

crash_loc = (
    crashed.groupby(["algo", "regime"])["crash_segment_type"]
    .value_counts()
    .unstack(fill_value=0)
    .reindex(columns=["Straight", "Corner"], fill_value=0)
)
crash_loc["n_crashes"] = crash_loc.sum(axis=1)
crash_loc["pct_corner"] = 100 * crash_loc["Corner"] / crash_loc["n_crashes"]
crash_loc["pct_straight"] = 100 * crash_loc["Straight"] / crash_loc["n_crashes"]
n_eps_by_cell = eval_all.groupby(["algo", "regime"]).size()
crash_loc["n_episodes"] = n_eps_by_cell
crash_loc["corner_crash_rate_per_episode"] = crash_loc["Corner"] / crash_loc["n_episodes"]
crash_loc["straight_crash_rate_per_episode"] = crash_loc["Straight"] / crash_loc["n_episodes"]
crash_loc = crash_loc.reset_index()
crash_loc.to_csv(TAB_DIR / "table_crash_location_by_segment_type.csv", index=False)
print(crash_loc.to_string(index=False))

# Per-seed granularity too -- needed to check whether any pooled algo x regime
# ranking above is itself an artefact of one outlier seed (the same trap the
# task brief warns about for REINFORCE-safe).
crash_loc_seed = (
    crashed.groupby(["algo", "regime", "seed"])["crash_segment_type"]
    .value_counts()
    .unstack(fill_value=0)
    .reindex(columns=["Straight", "Corner"], fill_value=0)
)
n_eps_seed = eval_all.groupby(["algo", "regime", "seed"]).size()
crash_loc_seed["n_episodes"] = n_eps_seed
crash_loc_seed["corner_crash_rate_per_episode"] = crash_loc_seed["Corner"] / crash_loc_seed["n_episodes"]
crash_loc_seed["straight_crash_rate_per_episode"] = crash_loc_seed["Straight"] / crash_loc_seed["n_episodes"]
crash_loc_seed = crash_loc_seed.reset_index()
crash_loc_seed.to_csv(TAB_DIR / "table_crash_location_by_segment_type_per_seed.csv", index=False)

# Pit-stop counts, rulebook vs unconstrained, per (algo, seed) -- the
# candidate "reasonably stable pit behaviour" comparison for Pass 2.
pit_pivot = seed_level.pivot_table(index=["algo", "seed"], columns="regime", values="mean_pit_stops")
pit_pivot["rulebook_minus_unconstrained"] = pit_pivot["rulebook"] - pit_pivot["unconstrained"]
pit_pivot = pit_pivot.reset_index()
pit_pivot.to_csv(TAB_DIR / "table_pitstops_rulebook_vs_unconstrained.csv", index=False)
print("\nPit stops, rulebook - unconstrained, per (algo, seed):")
print(pit_pivot.to_string(index=False))

print("\n" + "=" * 80)
print("STEP 5: Load raw per-model SHAP CSVs (data/shap/*.csv, 45 files)")
print("=" * 80)

shap_files = sorted(SHAP_RAW_DIR.glob("shap_*.csv"))
print(f"Found {len(shap_files)} raw SHAP CSVs (expected 5 algos x 3 regimes x 3 seeds = 45)")
assert len(shap_files) == 45, "Unexpected number of raw SHAP CSVs"

shap_rows = []
for f in shap_files:
    # filename: shap_{algo}_{regime}_seed={seed}.csv
    stem = f.stem
    seed = int(stem.split("seed=")[-1])
    body = stem[len("shap_"):stem.index("_seed=")]
    algo, regime = body.split("_", 1)
    df = pd.read_csv(f)
    df["algo"] = algo
    df["regime"] = regime
    df["seed"] = seed
    df["source_file"] = f.name
    shap_rows.append(df)

shap_all = pd.concat(shap_rows, ignore_index=True)

# ---------------------------------------------------------------------------
# Pass 1.3 -- SHAP internal-consistency check.
#
# The requested check (sum of per-feature SHAP == model-output difference
# from the base value, within float tolerance) CANNOT be performed on these
# artefacts. Verified by reading src/f1_rl_safety/shap_surrogates.py: the
# raw signed per-sample SHAP matrix, the KernelExplainer base/expected
# value, and the surrogate's raw predictions are computed in memory
# (`explainer.shap_values(...)`) and immediately reduced to
# `np.mean(np.abs(shap_values), axis=0)` before being written to
# `data/shap/*.csv`. Only that reduced, unsigned, per-feature scalar
# survives to disk -- there is no base value, no signed value and no model
# output anywhere in the repo to check additivity against. This is stated
# in the README rather than silently skipped.
#
# The substitute check that *is* possible from the persisted CSVs: are the
# structurally-zero features (no SHAP importance in any of the 45 runs)
# consistent across every single file, i.e. is "zero importance" itself a
# stable property rather than seed noise?
zero_features = (
    shap_all.groupby("feature")["mean_abs_shap"]
    .agg(["max", "count"])
    .query("max == 0.0")
)
print("Features with mean_abs_shap == 0.0 in ALL 45 (algo, regime, seed) runs:")
print(zero_features)

nonzero_but_small = (
    shap_all.groupby("feature")["mean_abs_shap"].max().sort_values()
)
print("\nPer-feature max mean_abs_shap across all 45 runs (sanity ordering):")
print(nonzero_but_small.to_string())

# Every value must be >= 0 (these are already |SHAP|, so this is a basic
# schema check, not a strong one).
assert (shap_all["mean_abs_shap"] >= 0).all(), "Found a negative mean_abs_shap value"
n_files_per_cell = shap_all.groupby(["algo", "regime", "seed"])["source_file"].nunique()
assert (n_files_per_cell == 1).all()
n_features_per_file = shap_all.groupby("source_file")["feature"].nunique()
print(f"\nAll 45 files have exactly 18 features each: {(n_features_per_file == 18).all()}")

# ---------------------------------------------------------------------------
# Renamed feature tables
# ---------------------------------------------------------------------------

feature_map_df = pd.DataFrame(
    {"s_index": list(FEATURE_NAMES.keys()), "feature_name": list(FEATURE_NAMES.values())}
)
feature_map_df.to_csv(TAB_DIR / "table_shap_feature_mapping.csv", index=False)

shap_all_named = rename_feature_col(shap_all)
shap_all_named.to_csv(TAB_DIR / "table_shap_by_algo_regime_seed_renamed.csv", index=False)

overall_ranking = pd.read_csv(SHAP_AGG_DIR / "shap_overall_feature_ranking.csv")
overall_ranking_named = rename_feature_col(overall_ranking)
overall_ranking_named.to_csv(TAB_DIR / "table_shap_overall_ranking_renamed.csv", index=False)
print("\nOverall SHAP ranking (renamed), from output/shap_overall_feature_ranking.csv:")
print(overall_ranking_named.to_string(index=False))

# ---------------------------------------------------------------------------
# FIGURE 1 -- seed-reliability overview.
# Small multiples, one panel per algorithm; x = regime; y = crash_rate;
# one dot per training seed (n=3) plus a black mean marker. This is the
# figure that motivates everything else: it shows, directly from
# table_seed_level_metrics.csv, how much the 3 seeds within a cell agree
# or disagree.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 1: Seed reliability overview (crash rate)")
print("=" * 80)

fig, axes = plt.subplots(1, 5, figsize=(14, 3.6), sharey=True)
regime_order = ["unconstrained", "rulebook", "safe"]
for ax, algo in zip(axes, ALGOS):
    sub = seed_level[seed_level["algo"] == algo]
    for i, regime in enumerate(regime_order):
        vals = sub[sub["regime"] == regime]["crash_rate"].values
        jitter = np.linspace(-0.09, 0.09, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=32, color=REGIME_COLOR[regime], zorder=3,
            edgecolor="white", linewidth=0.6,
        )
        ax.hlines(vals.mean(), i - 0.22, i + 0.22, color=INK_PRIMARY, linewidth=2.2, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Uncon.", "Rulebook", "Safe"], fontsize=8.5)
    ax.set_title(algo.upper(), fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    style_axes(ax)
axes[0].set_ylabel("Crash rate\n(fraction of 100 episodes, per seed)")
fig.suptitle(
    "Crash rate is seed-sensitive: each dot is one of 3 training seeds, black bar is the seed mean",
    fontsize=11.5, fontweight="bold", y=1.04,
)
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=REGIME_COLOR[r],
                       markersize=8, label=r.capitalize()) for r in regime_order]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
fig.text(0.5, -0.16, "Source: outputs/experiments/eval/final-v1/*_steps=200000_eval=100.csv "
                      "-> table_seed_level_metrics.csv. n = 3 training seeds x 100 eval episodes per dot.",
          ha="center", fontsize=8, color=INK_MUTED)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(FIG_DIR / "fig_seed_reliability_overview.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_seed_reliability_overview.png")

# ---------------------------------------------------------------------------
# FIGURE 2 -- primary comparison #1: PPO & A2C, safe vs unconstrained.
# Both are sign-consistent across all 3 seeds for crash_rate and mean_risk
# (see table_seed_consistency.csv). Bars = seed mean, error bars = seed
# min/max (not SD, since n=3), dots = individual seeds.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 2: PPO & A2C, safe vs unconstrained (crash rate, mean risk)")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
sel_algos = ["ppo", "a2c"]
metrics_2 = [("crash_rate", "Crash rate"), ("mean_risk", "Mean chosen risk level")]

for ax, (metric, label) in zip(axes, metrics_2):
    x = np.arange(len(sel_algos))
    width = 0.32
    for j, regime in enumerate(["unconstrained", "safe"]):
        means, mins, maxs = [], [], []
        for algo in sel_algos:
            vals = seed_level[(seed_level.algo == algo) & (seed_level.regime == regime)][metric].values
            means.append(vals.mean()); mins.append(vals.min()); maxs.append(vals.max())
        means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
        xpos = x + (j - 0.5) * width
        ax.bar(xpos, means, width=width * 0.92, color=REGIME_COLOR[regime],
               label=regime.capitalize(), zorder=2)
        ax.vlines(xpos, mins, maxs, color=INK_PRIMARY, linewidth=1.3, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in sel_algos])
    ax.set_title(label, fontsize=11)
    ax.axhline(0, color=GRID, linewidth=0.8)
    style_axes(ax)

axes[0].set_ylabel("Value (mean over 100 episodes, per seed)")
handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[r]) for r in ["unconstrained", "safe"]]
fig.legend(handles, ["Unconstrained", "Safe"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.suptitle(
    "PPO and A2C: safe reduces crash rate and mean chosen risk vs unconstrained,\n"
    "consistently across all 3 training seeds",
    fontsize=11.5, fontweight="bold", y=1.08,
)
fig.text(
    0.5, -0.06,
    "Bars = mean over 3 seeds; vertical lines = seed min-max range (n=3 seeds, not SD).\n"
    "Source: table_seed_level_metrics.csv, derived from outputs/experiments/eval/final-v1/.",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig(FIG_DIR / "fig_safe_vs_unconstrained_ppo_a2c.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_safe_vs_unconstrained_ppo_a2c.png")

# ---------------------------------------------------------------------------
# FIGURE 3 -- SHAP mechanism behind figure 2: renamed top-8 feature
# importances for PPO & A2C, safe vs unconstrained, averaged over the 3
# seeds (mean of mean_abs_shap; matches the aggregation already used by
# scripts/analyse_global_importance.py's regime-comparison charts).
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 3: SHAP feature importance, PPO & A2C, safe vs unconstrained")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=False)
for ax, algo in zip(axes, sel_algos):
    sub = shap_all_named[(shap_all_named.algo == algo) & (shap_all_named.regime.isin(["unconstrained", "safe"]))]
    top_feats = (
        sub.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(8).index.tolist()
    )
    piv = (
        sub[sub.feature.isin(top_feats)]
        .groupby(["feature", "regime"])["mean_abs_shap"].mean()
        .unstack("regime")
        .reindex(top_feats)
    )
    y = np.arange(len(piv))
    h = 0.36
    ax.barh(y + h / 2, piv["unconstrained"], height=h, color=REGIME_COLOR["unconstrained"], label="Unconstrained")
    ax.barh(y - h / 2, piv["safe"], height=h, color=REGIME_COLOR["safe"], label="Safe")
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_title(algo.upper(), fontsize=11)
    ax.set_xlabel("Mean |SHAP|  (avg. over 3 seeds)")
    style_axes(ax)

handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLOR[r]) for r in ["unconstrained", "safe"]]
fig.legend(handles, ["Unconstrained", "Safe"], loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.suptitle(
    "Top-8 surrogate-model feature importances, PPO & A2C: safe vs unconstrained",
    fontsize=11.5, fontweight="bold", y=1.07,
)
fig.text(
    0.5, -0.05,
    "SHAP explains a linear surrogate fit to each policy's chosen action, not the policy network itself "
    "(see README methodology note). Source: data/shap/shap_{ppo,a2c}_{safe,unconstrained}_seed={0,1,2}.csv.",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.savefig(FIG_DIR / "fig_shap_safe_vs_unconstrained_ppo_a2c.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_shap_safe_vs_unconstrained_ppo_a2c.png")

# ---------------------------------------------------------------------------
# FIGURE 4 -- primary comparison #2: PPO, rulebook vs unconstrained,
# pit-stop count. This is the one place a rulebook-vs-unconstrained effect
# is sign-consistent across all 3 seeds (table_pitstops_rulebook_vs_unconstrained.csv):
# PPO pits more under rulebook in every seed, a small but reliable effect
# consistent with the rulebook regime's reward shaping (RULEBOOK penalises
# finishing with pit_count < 1; see src/f1_rl_safety/f1_env.py:403-406).
# Paired with the SHAP swing in pit_count_norm / tyre features.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 4: PPO pit stops, rulebook vs unconstrained")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

ax = axes[0]
ppo_pits = seed_level[(seed_level.algo == "ppo") & (seed_level.regime.isin(["unconstrained", "rulebook"]))]
x = np.arange(3)
width = 0.34
for j, regime in enumerate(["unconstrained", "rulebook"]):
    vals = ppo_pits[ppo_pits.regime == regime].sort_values("seed")["mean_pit_stops"].values
    ax.bar(x + (j - 0.5) * width, vals, width=width * 0.92, color=REGIME_COLOR[regime], label=regime.capitalize())
ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in SEEDS])
ax.set_ylabel("Mean pit stops per episode")
ax.set_title("PPO: pit stops per seed", fontsize=11)
ax.legend(frameon=False, fontsize=9)
style_axes(ax)

ax = axes[1]
sub = shap_all_named[(shap_all_named.algo == "ppo") & (shap_all_named.regime.isin(["unconstrained", "rulebook"]))]
top_feats = sub.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(8).index.tolist()
piv = sub[sub.feature.isin(top_feats)].groupby(["feature", "regime"])["mean_abs_shap"].mean().unstack("regime").reindex(top_feats)
y = np.arange(len(piv))
h = 0.36
ax.barh(y + h / 2, piv["unconstrained"], height=h, color=REGIME_COLOR["unconstrained"], label="Unconstrained")
ax.barh(y - h / 2, piv["rulebook"], height=h, color=REGIME_COLOR["rulebook"], label="Rulebook")
ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP| (avg. over 3 seeds)")
ax.set_title("PPO: top-8 SHAP features", fontsize=11)
ax.legend(frameon=False, fontsize=9)
style_axes(ax)

fig.suptitle(
    "PPO pits slightly more under rulebook than unconstrained in all 3 seeds (small, reliable effect)",
    fontsize=11.5, fontweight="bold", y=1.06,
)
fig.text(
    0.5, -0.06,
    "Left: table_pitstops_rulebook_vs_unconstrained.csv. Right: data/shap/shap_ppo_{unconstrained,rulebook}_seed={0,1,2}.csv.\n"
    "Effect size is small (~0.3-0.9 extra pit stops/episode); framed as a reliable but modest pattern, not a strategic overhaul.",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.90])
fig.savefig(FIG_DIR / "fig_rulebook_vs_unconstrained_ppo_pitstops.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_rulebook_vs_unconstrained_ppo_pitstops.png")

# ---------------------------------------------------------------------------
# FIGURE 5 -- cautionary / instability example: REINFORCE, safe regime.
# Seed 0 finishes 49% of episodes (crash rate 0.51) and pits ~35.7
# times/episode on average; seeds 1-2 finish ~14-15% (crash rate ~0.85-0.86)
# and never pit. The cell mean is entirely an artefact of one seed's
# degenerate "pit every lap" policy. The SHAP files confirm this
# qualitatively: only seed 0's surrogate assigns pit_count_norm (s_17) any
# importance at all (0.075); seeds 1 and 2 give it exactly 0.0.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 5: REINFORCE-safe -- single-seed-dominated cell (cautionary example)")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

rf = seed_level[seed_level.algo == "reinforce"].copy()
ax = axes[0]
for regime in regime_order:
    vals = rf[rf.regime == regime].sort_values("seed")["finish_rate"].values
    ax.plot(SEEDS, vals, "o-", color=REGIME_COLOR[regime], label=regime.capitalize(), linewidth=1.8, markersize=7)
ax.set_xticks(SEEDS); ax.set_xlabel("Training seed"); ax.set_ylabel("Finish rate")
ax.set_title("Finish rate by seed", fontsize=11)
ax.legend(frameon=False, fontsize=8.5)
style_axes(ax)

ax = axes[1]
for regime in regime_order:
    vals = rf[rf.regime == regime].sort_values("seed")["mean_pit_stops"].values
    ax.plot(SEEDS, vals, "o-", color=REGIME_COLOR[regime], linewidth=1.8, markersize=7)
ax.set_xticks(SEEDS); ax.set_xlabel("Training seed"); ax.set_ylabel("Mean pit stops / episode")
ax.set_title("Pit stops by seed", fontsize=11)
ax.annotate("seed 0: 35.7 pit\nstops/episode", xy=(0, rf[(rf.regime == "safe") & (rf.seed == 0)]["mean_pit_stops"].values[0]),
            xytext=(0.55, 24), fontsize=8.5, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))
style_axes(ax)

ax = axes[2]
rf_safe_shap = shap_all_named[(shap_all_named.algo == "reinforce") & (shap_all_named.regime == "safe")]
top_feats = rf_safe_shap.groupby("feature")["mean_abs_shap"].mean().sort_values(ascending=False).head(6).index.tolist()
piv = rf_safe_shap[rf_safe_shap.feature.isin(top_feats)].pivot_table(index="feature", columns="seed", values="mean_abs_shap").reindex(top_feats)
y = np.arange(len(piv))
h = 0.24
seed_colors = [CAT["blue"], CAT["orange"], CAT["aqua"]]
for i, s in enumerate(SEEDS):
    ax.barh(y + (i - 1) * h, piv[s], height=h * 0.92, color=seed_colors[i], label=f"seed {s}")
ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Mean |SHAP|")
ax.set_title("REINFORCE-safe: SHAP by seed", fontsize=11)
ax.legend(frameon=False, fontsize=8.5)
style_axes(ax)

fig.suptitle(
    "REINFORCE-safe: the regime mean is dominated by one outlier seed running a degenerate pit-every-lap policy",
    fontsize=11.5, fontweight="bold", y=1.07,
)
fig.text(
    0.5, -0.06,
    "Left/middle: table_seed_level_metrics.csv. Right: data/shap/shap_reinforce_safe_seed={0,1,2}.csv -- only seed 0 "
    "attributes any importance to pit_count_norm (0.075 vs 0.0 for seeds 1-2).",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.02, 1, 0.88])
fig.savefig(FIG_DIR / "fig_reinforce_safe_outlier.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_reinforce_safe_outlier.png")

# ---------------------------------------------------------------------------
# FIGURE 6 -- "better at corners vs better at straights", within the safe
# regime (controls for regime so algorithms are compared on a level
# footing). REINFORCE is EXCLUDED from the headline ranking because its
# pooled numbers are the seed-0 pit-every-lap outlier from figure 5 (its
# straight-crash rate collapses to 0.03 in seed 0 vs ~0.28-0.29 in seeds
# 1-2 -- see table_crash_location_by_segment_type_per_seed.csv); it is
# shown on the chart in a muted style with that caveat rather than dropped
# silently.
#
# IMPORTANT (post-audit correction): DQN has the lowest MEAN corner-crash
# rate and PPO has the lowest MEAN straight-crash rate across the 3 safe
# seeds, but neither ranking is seed-consistent. Per
# table_crash_location_by_segment_type_per_seed.csv, seed 2 reverses both:
# PPO's corner rate (0.49) is lower than DQN's (0.55), and DQN's straight
# rate (0.20) is lower than PPO's (0.27). Do not describe either algorithm
# as "the most seed-consistent" at either segment type -- these are
# 3-seed descriptive means, not stable rankings.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 6: Corner vs straight crash rate by algorithm (safe regime)")
print("=" * 80)

safe_loc_seed = crash_loc_seed[crash_loc_seed.regime == "safe"]
order_algos = ["dqn", "ppo", "a2c", "sarsa", "reinforce"]  # dqn/ppo = lowest mean, not seed-uniform (see comment above); reinforce last, flagged

fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(order_algos))
width = 0.34
for j, (col, color, label) in enumerate([
    ("corner_crash_rate_per_episode", CAT["red"], "Corner crashes / episode"),
    ("straight_crash_rate_per_episode", CAT["blue"], "Straight crashes / episode"),
]):
    means, mins, maxs = [], [], []
    for algo in order_algos:
        vals = safe_loc_seed[safe_loc_seed.algo == algo].sort_values("seed")[col].values
        means.append(vals.mean()); mins.append(vals.min()); maxs.append(vals.max())
    means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
    xpos = x + (j - 0.5) * width
    alphas = [1.0 if a != "reinforce" else 0.45 for a in order_algos]
    for xi, m, lo, hi, al in zip(xpos, means, mins, maxs, alphas):
        ax.bar(xi, m, width=width * 0.92, color=color, alpha=al, zorder=2,
               label=label if xi == xpos[0] else None)
        ax.vlines(xi, lo, hi, color=INK_PRIMARY, linewidth=1.2, alpha=al, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([a.upper() for a in order_algos])
ax.set_ylabel("Crashes per episode (safe regime, mean of 3 seeds)")
style_axes(ax)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.annotate("REINFORCE shown faded: its low pooled rate is a\nseed-0 outlier artefact (see fig_reinforce_safe_outlier)",
            xy=(4, 0.05), xytext=(1.3, 0.15), fontsize=8.3, color=INK_SECONDARY,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=1))
fig.suptitle(
    "Safe regime crash profile by segment type: DQN has the lowest mean corner-crash\n"
    "rate, while PPO has the lowest mean straight-crash rate; rankings vary by seed.",
    fontsize=11.5, fontweight="bold", y=0.99,
)
fig.text(
    0.5, -0.08,
    "Vertical lines = seed min-max range (n=3 seeds). Source: table_crash_location_by_segment_type_per_seed.csv, "
    "derived from crash_segment_type in outputs/experiments/eval/final-v1/.",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.88])
fig.savefig(FIG_DIR / "fig_corner_vs_straight_crash_rate.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_corner_vs_straight_crash_rate.png")

# ---------------------------------------------------------------------------
# FIGURE 7 -- reference / supporting figure: cross-architecture SHAP
# heatmap, recreated with renamed features. This reproduces the pivot in
# scripts/analyse_global_importance.py::create_cross_architecture_heatmap
# (same top-15-by-total-importance selection) directly from shap_all_named,
# so it matches output/shap_cross_architecture_matrix.csv / the existing
# output/chart_shap_cross_architecture_heatmap.html, just relabelled and
# restyled for print. Already anointed the "headline" cross-architecture
# figure by docs/shap-thesis-notes.md.
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FIGURE 7: Cross-architecture SHAP heatmap (renamed, reference figure)")
print("=" * 80)

pivot = shap_all_named.pivot_table(index="feature", columns=["algo", "regime"], values="mean_abs_shap", aggfunc="mean")
pivot["total"] = pivot.sum(axis=1)
pivot = pivot.sort_values("total", ascending=False).drop(columns="total").head(15)
col_order = [(a, r) for a in ALGOS for r in REGIMES if (a, r) in pivot.columns]
pivot = pivot[col_order]

fig, ax = plt.subplots(figsize=(12.5, 7))
seq_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#cde2fb", "#3987e5", "#0d366b"])
im = ax.imshow(pivot.values, cmap=seq_cmap, aspect="auto")
ax.set_xticks(range(len(col_order)))
ax.set_xticklabels([f"{a.upper()} / {r}" for a, r in col_order], fontsize=8, rotation=40, ha="right")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9.5)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        txt_color = "white" if v > pivot.values.max() * 0.55 else INK_PRIMARY
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.8, color=txt_color)
cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cbar.set_label("Mean |SHAP|", fontsize=9.5)
ax.set_title(
    "Top-15 SHAP features across all 5 algorithms x 3 regimes (avg. over 3 seeds)",
    fontsize=12.5, fontweight="bold", pad=14,
)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="minor", length=0)
fig.text(
    0.5, 0.01,
    "Reference/supporting figure (not one of the primary seed-consistent comparisons above). "
    "Source: data/shap/shap_*.csv, same pivot as scripts/analyse_global_importance.py's cross-architecture heatmap, features renamed.",
    ha="center", fontsize=8, color=INK_MUTED,
)
fig.tight_layout(rect=[0, 0.03, 1, 1])
fig.savefig(FIG_DIR / "fig_shap_cross_architecture_heatmap_renamed.png", bbox_inches="tight")
plt.close(fig)
print("Saved fig_shap_cross_architecture_heatmap_renamed.png")

# ---------------------------------------------------------------------------
# Curated summary table: the specific comparisons cited in the README,
# in one small, readable table (table_regime_comparison_summary.csv).
# ---------------------------------------------------------------------------

def get_delta_row(algo, ra, rb, metric):
    row = seed_consistency[
        (seed_consistency.algo == algo) & (seed_consistency.regime_a == ra)
        & (seed_consistency.regime_b == rb) & (seed_consistency.metric == metric)
    ].iloc[0]
    return row

summary_rows = []
for algo, ra, rb, metric, verdict in [
    ("ppo", "unconstrained", "safe", "crash_rate", "Primary: reliable, all 3 seeds agree"),
    ("ppo", "unconstrained", "safe", "mean_risk", "Primary: reliable, all 3 seeds agree"),
    ("a2c", "unconstrained", "safe", "crash_rate", "Primary: reliable, all 3 seeds agree"),
    ("a2c", "unconstrained", "safe", "mean_risk", "Primary: reliable, all 3 seeds agree"),
    ("dqn", "unconstrained", "safe", "crash_rate", "Secondary: consistent direction, one seed near-zero effect"),
    ("reinforce", "unconstrained", "safe", "crash_rate", "Caution: sign-consistent but magnitude entirely seed-0-driven"),
    ("reinforce", "unconstrained", "safe", "mean_risk", "Caution: sign INCONSISTENT across seeds"),
    ("sarsa", "unconstrained", "safe", "crash_rate", "Not reliable: sign inconsistent across seeds"),
]:
    r = get_delta_row(algo, ra, rb, metric)
    summary_rows.append({
        "algo": algo, "regime_a": ra, "regime_b": rb, "metric": metric,
        "delta_seed0": r["delta_seed0"], "delta_seed1": r["delta_seed1"], "delta_seed2": r["delta_seed2"],
        "mean_delta": r["mean_delta"], "sign_consistent_across_seeds": r["sign_consistent_across_seeds"],
        "outlier_seed_dominated": r["outlier_seed_dominated"], "verdict": verdict,
    })

# Pit stops, rulebook vs unconstrained, PPO (separate metric, added manually)
ppo_pit = pit_pivot[pit_pivot.algo == "ppo"]
summary_rows.append({
    "algo": "ppo", "regime_a": "unconstrained", "regime_b": "rulebook", "metric": "mean_pit_stops",
    "delta_seed0": ppo_pit.iloc[0]["rulebook_minus_unconstrained"],
    "delta_seed1": ppo_pit.iloc[1]["rulebook_minus_unconstrained"],
    "delta_seed2": ppo_pit.iloc[2]["rulebook_minus_unconstrained"],
    "mean_delta": ppo_pit["rulebook_minus_unconstrained"].mean(),
    "sign_consistent_across_seeds": bool((ppo_pit["rulebook_minus_unconstrained"] > 0).all()),
    "outlier_seed_dominated": False,
    "verdict": "Primary: small but reliable, all 3 seeds agree",
})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TAB_DIR / "table_regime_comparison_summary.csv", index=False)
print("\nCurated regime-comparison summary (table_regime_comparison_summary.csv):")
print(summary_df.to_string(index=False))

print("\n" + "=" * 80)
print("ALL FIGURES AND TABLES WRITTEN")
print("=" * 80)
print(f"Figures: {sorted(p.name for p in FIG_DIR.glob('*.png'))}")
print(f"Tables:  {sorted(p.name for p in TAB_DIR.glob('*.csv'))}")

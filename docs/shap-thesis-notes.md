# SHAP thesis notes

## Purpose

This file describes how the SHAP outputs in `data/shap/` and `output/` map to
thesis figures and tables, and how to interpret the generic `s_0`…`s_17`
feature indices used throughout those artefacts. Use it as a reference when
selecting and captioning SHAP figures/tables for the results and appendix
sections of the thesis.

## Primary figures (headline results)

- **`output/chart_shap_cross_architecture_heatmap.html`** — "Which features
  matter across all 5 algorithms?" The main cross-architecture summary
  figure: a heatmap of mean |SHAP| for the top 15 features across every
  algorithm × regime combination. Belongs in the main results chapter as the
  headline feature-importance figure.
- **`output/chart_regime_comparison_{a2c,dqn,ppo,reinforce,sarsa}.html`**
  (5 files) — "Does the reward regime change what the policy attends to?"
  One grouped bar chart per algorithm, comparing its top-10 feature
  importances across `unconstrained`/`rulebook`/`safe`. These are the
  per-algorithm counterpart to the cross-architecture heatmap and likely
  belong in the main results chapter alongside it (or as a compact grid if
  space is tight).

## Secondary / per-architecture figures

- **`output/chart_shap_{a2c,dqn,ppo,reinforce,sarsa}_top12.html`**
  (5 files) — Top-12 feature importances for a single algorithm, averaged
  across all regimes. Supporting detail rather than a headline claim;
  likely appendix-grade, one figure per algorithm for readers who want the
  full per-architecture picture.
- **`output/chart_shap_ppo_safe.html`** — A narrow, single-condition figure
  (PPO under the `safe` regime only). Better suited as a worked example
  embedded in the text (e.g. "consider PPO under the safe regime...") than
  as a standalone thesis figure, since it duplicates a slice of information
  already present in `chart_regime_comparison_ppo.html`.

## Supporting tables

**Aggregate rankings — candidates for appendix tables:**

- `output/shap_overall_feature_ranking.csv` — mean/std/min/max/count of
  |SHAP| per feature, aggregated across all 45 algo×regime×seed runs.
- `output/shap_per_architecture_ranking.csv` — mean |SHAP| per feature,
  grouped by algorithm.
- `output/shap_per_regime_ranking.csv` — mean |SHAP| per feature, grouped
  by regime.

**Raw/detailed backing data — appendix or supplementary material rather
than in-text tables:**

- `output/shap_cross_architecture_matrix.csv` — the pivoted matrix
  underlying the cross-architecture heatmap (feature × algo/regime).
- `output/shap_top10_features.csv` — top-10 features per algo/regime group,
  the table underlying `analyse_shap.py`'s output.
- `output/shap_by_algo_regime_seed.csv` — full, unaggregated feature
  importances for every one of the 45 (algo, regime, seed) runs.

## Feature naming: `s_0`…`s_17` → real observation features

Every SHAP CSV (`data/shap/shap_*.csv` and the `output/shap_*.csv` summary
tables) labels features generically as `s_0`…`s_17`. These correspond
positionally to the observation vector returned by
`F1RaceEnv._get_obs()` in `src/f1_rl_safety/f1_env.py` (confirmed against
the concatenation order there; the `DiscreteF1ActionWrapper` used for
DQN/SARSA passes this vector through unchanged):

| Index | Feature name |
|---|---|
| `s_0` | `lap_fraction` |
| `s_1` | `race_time_norm` |
| `s_2` | `pos_norm` |
| `s_3` | `gap_ahead` |
| `s_4` | `gap_behind` |
| `s_5` | `tyre_age_norm` |
| `s_6` | `tyre_wear` |
| `s_7` | `fuel_norm` |
| `s_8` | `track_status_0` |
| `s_9` | `track_status_1` |
| `s_10` | `track_status_2` |
| `s_11` | `tyre_compound_0` |
| `s_12` | `tyre_compound_1` |
| `s_13` | `tyre_compound_2` |
| `s_14` | `tyre_compound_3` |
| `s_15` | `tyre_compound_4` |
| `s_16` | `risk_indicator` |
| `s_17` | `pit_count_norm` |

For thesis figures/tables built from these CSVs, either relabel `s_0`…`s_17`
using the mapping above before plotting, or keep the generic labels and
include this table as a legend/caption so a reader can look up what `s_12`
or `s_16` actually means without cross-referencing the code.

## Regenerating SHAP outputs

```bash
# From repo root
PYTHONPATH=src .venv_f1/bin/python scripts/generate_shap_csvs.py
PYTHONPATH=src .venv_f1/bin/python scripts/analyse_shap.py
PYTHONPATH=src .venv_f1/bin/python scripts/analyse_global_importance.py
```

- `data/shap/*.csv` are the ground-truth per-model SHAP values (one file per
  algo × regime × seed).
- `output/shap_*.csv` and `output/chart_shap_*.html` / `chart_regime_comparison_*.html`
  are derived from those CSVs by the two `analyse_*.py` scripts.
- `data/shap/*.png` and `output/*.meta.json` are ignored intermediates —
  regenerated automatically by the commands above, not tracked in git.

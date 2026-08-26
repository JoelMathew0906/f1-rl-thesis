# PPO reference baseline — 3 regimes × 3 seeds × 100k steps (2026-08-26)

Reproducible PPO baseline on the recalibrated environment (post
`CRASH_HAZARD_SCALE` + pit-loss fix). No changes to rewards, dynamics,
wrappers, spaces, hyperparameters, or entrypoints.

- `run_study.sh` — exact training loop (seeds 0/1/2, 100,000 steps each).
- `eval_and_diagnose.py` — 50 deterministic evaluation episodes per model via
  the unmodified `evaluate_rl.evaluate_grid`, plus diagnostic-only replay
  recording the pit action channel (`action[0]`) at every lap-final segment.
- `manifest.csv` — regime, seed, steps, model path, eval CSV, exact commands,
  timestamps for every run.
- `summary_per_seed.csv` / `summary_aggregated.csv` — per-seed metrics and
  across-seed mean/SD (150 episodes per regime).
- `summary_pit_diagnostics.csv` — pit opportunities reached, pit-intent rate,
  action[0] statistics, decoded tyre choices.
- Untracked bulk (kept on disk, not committed): `models/`, `logs/`,
  `trainlogs/`, `eval/` per-episode CSVs, `diagnostics/` per-opportunity CSVs.

Headline: intended risk ordering emerges across seeds (unconstrained +0.76 >
rulebook +0.09 > safe −0.23 mean risk); finish rates 3–11%; pit discovery
essentially absent (1 of 9 models, 2 of 450 episodes). Diagnostics attribute
this to learned avoidance (policy pit-channel mean driven to the clipped
floor despite std ≈ 0.9 sampling pit actions throughout training), not to
lack of exploration. See summary CSVs and the phase2 session report.

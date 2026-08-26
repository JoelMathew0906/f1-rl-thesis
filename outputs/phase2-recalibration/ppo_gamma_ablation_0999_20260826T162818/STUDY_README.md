# PPO gamma ablation — 0.99 (baseline) vs 0.999, 3 regimes × 3 seeds × 100k (2026-08-26)

Tests whether raising the PPO discount factor from 0.99 (SB3 default, used by
the committed baseline study `ppo_baseline_100k_20260826T161335`) to 0.999
resolves the diagnosed long-horizon credit-assignment failure (immediate
21.5 s pit loss vs fresh-tyre benefit accruing over 15–25 laps). Only change:
`--gamma 0.999 --run-name gamma0999`; environment, rewards, evaluator, all
other hyperparameters identical. Models under `models/gamma0999/` — no
collision with any gamma-0.99 artefact.

Headline (mean over 3 seeds, 150 eval episodes/regime, deterministic):
- RULEBOOK: pits/episode 0.013 → 0.580; compliance 1.3% → 56.7%; mean return
  −34.3 → +2.5; finishers 391 s faster. Two of three seeds learned a
  strategic one-stop with SOFT;MEDIUM compound diversity (pit intent at ~3%
  of lap-final opportunities ≈ once per race).
- SAFE: pits 0 → 0.24 (one of three seeds, MEDIUM→MEDIUM tyre management —
  consistent with SAFE having a pit milestone but no compound incentive);
  finish rate 0.107 → 0.167; catastrophic rate 0.173 → 0.140; mean risk
  −0.225 → −0.411.
- UNCONSTRAINED: still zero-stop, high risk (0.68), unchanged within noise.
- Risk ordering preserved and sharpened: 0.680 > 0.197 > −0.411.
- Remaining caveat: pit discovery is seed-bimodal (2/3 rulebook, 1/3 safe) —
  an exploration lottery, honestly reported; use ≥3 seeds per cell downstream.

Files: `manifest.csv`, `summary_per_seed.csv`, `summary_aggregated.csv`,
`summary_pit_diagnostics.csv`, `run_study.sh`, `eval_and_diagnose.py`.
Untracked bulk on disk: `models/`, `logs/`, `trainlogs/`, `eval/`,
`diagnostics/`.

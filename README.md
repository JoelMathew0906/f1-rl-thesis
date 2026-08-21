# F1 RL Thesis: Multi-Algorithm Safety and Reward Comparison

This repository hosts the thesis-focused version of the F1 RL safety project. It extends the original `f1-rl-safety` codebase into a multi-architecture, multi-reward comparison study for a single-car strategic Formula 1 race simulator calibrated on the 2025 British Grand Prix at Silverstone.

The codebase and folder layout mirror `f1-rl-safety`, but this repository is the canonical home for the thesis analysis, notebooks, and comparison artefacts.

## Thesis framing

The thesis asks how different reinforcement learning architectures learn pit-stop and driving-risk strategies under a fixed race environment and three objective regimes, and which architectural choices best balance performance, rule compliance, and safety. It also investigates which state variables drive decision-making under each architecture and regime, using surrogate-based SHAP analysis.

The current design crosses:

- **Five architectures:** PPO, A2C, DQN, SARSA, and REINFORCE.
- **Three regimes:** unconstrained performance, rulebook-aware, and safety-constrained.
- **Common environment:** a strategic single-car Silverstone race simulator calibrated on 2025 lap data (`data/silverstone_2025_laps.csv`).

All generated CSVs, HTML comparison pages, and SHAP images currently live alongside the code in `data/`, `output/`, and the notebooks. As the thesis evolves, these may be reorganised into a dedicated `results/` hierarchy, but this repository intentionally keeps the exploratory artefacts visible to support rapid iteration.

## Repository structure (as of now)

```text
f1-rl-thesis/
├── .gitkeep                       Placeholder (safe to ignore)
├── configs/
│   └── configs_silverstone.yaml   Silverstone configuration stub
├── data/
│   └── silverstone_2025_laps.csv  FastF1-derived calibration data for Silverstone
├── notebooks/
│   ├── 01_environment_and_data.ipynb       Environment + calibration exploration
│   ├── 02_training_and_logs.ipynb          Training runs and logging
│   └── 03_evaluation_and_plots.ipynb       Evaluation, tables, and plots for thesis
├── requirements.txt                Python dependencies (Gymnasium, SB3, PyTorch, FastF1, pandas, SHAP, etc.)
├── scripts/
│   ├── analyse_results.py          Aggregate evaluation CSVs, build comparison tables/plots
│   ├── analyse_shap.py             Aggregate SHAP CSVs, extract top-k feature importances
│   └── analyse_global_importance.py Cross-architecture/regime SHAP analysis
└── src/
    └── f1_rl_safety/
        ├── __init__.py             Package init
        ├── data_loader.py          Data loading and Silverstone calibration utilities
        ├── eval_policies.py        Original PPO evaluation entry point
        ├── evaluate_rl.py          Unified evaluation utilities for all algorithms
        ├── f1_env.py               Gymnasium race environment and reward regimes
        ├── reinforce_agent.py      Episodic REINFORCE policy-gradient implementation
        ├── shap_surrogates.py      Supervised surrogates + SHAP explainability pipeline
        ├── train.py                Original PPO training script
        ├── train_rl.py             Unified multi-architecture training entry point
        ├── value_based.py          DQN/SARSA implementations over discrete action wrapper
        └── wrappers.py             DiscreteF1ActionWrapper for value-based agents
```

If you have additional directories (e.g., `logs/`, `models/`, `output/`) in your local clone, they will appear here as you commit them. They are part of the thesis record and may contain training logs, saved policies, evaluation CSVs, and images.

## How to run experiments

Create and activate a Python environment (Python 3.13.x recommended) and install dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### Training

The unified training script accepts the algorithm, reward regime, budget, and seed. Check its interface for the most up-to-date arguments:

```bash
python -m f1_rl_safety.train_rl --help
```

Example single-configuration runs:

```bash
python -m f1_rl_safety.train_rl --algo ppo       --regime safe        --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo a2c       --regime rulebook    --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo dqn       --regime unconstrained --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo sarsa     --regime safe        --steps 50000 --seed 0
python -m f1_rl_safety.train_rl --algo reinforce --regime safe        --episodes 200 --seed 0
```

PPO, A2C, DQN, and SARSA use a common 50k environment-step budget in the current exploratory grid; REINFORCE is trained for 200 full episodes, which should be interpreted as an approximate budget match rather than an exact equality of transitions.

### Evaluation

Use the unified evaluator to generate per-policy evaluation CSVs (race time, crash and catastrophic rates, pit-stop statistics, risk indicators, etc.):

```bash
python -m f1_rl_safety.evaluate_rl --help
```

The default configuration produces CSV files under `data/experiment_results/` in the original project; in this thesis repository, use `scripts/analyse_results.py` to aggregate whatever evaluation outputs you have committed.

### Analysis and SHAP explainability

From the repository root:

```bash
python scripts/analyse_results.py
python scripts/analyse_shap.py
python scripts/analyse_global_importance.py
```

These scripts expect evaluation CSVs and SHAP CSVs generated by your training and evaluation runs and produce aggregate tables, plots, and SHAP feature-importance summaries suitable for inclusion in the thesis.

The SHAP pipeline in `src/f1_rl_safety/shap_surrogates.py` trains supervised surrogates to approximate the agents’ action policies and computes SHAP values for state features. These are approximate, post-hoc attributions and should be interpreted in the thesis with the usual caveats about surrogate fidelity and correlation versus causation.

## Limitations

The core limitations of the underlying project still apply here:

- Single-track, single-car strategic simulator calibrated on one event.
- Three reward regimes implemented via reward shaping rather than hard constraints.
- Action discretisation for DQN/SARSA, which differs from the continuous-action PPO/A2C/REINFORCE setup.
- Current experiments are single-seed and exploratory; they reveal behavioural patterns but do not support statistical claims.
- REINFORCE uses an episode-based budget rather than a strictly matched environment-step budget.
- SHAP explanations are based on supervised surrogates, not the raw RL algorithms.

The role of this repository is to keep the architecture comparison, evaluation, and SHAP analysis reproducible and thesis-ready.

## Origin

This repository was cloned from `https://github.com/JoelMathew0906/f1-rl-safety` and reorganised as the canonical location for Joel Mathew’s MRes thesis work on RL architectures and reward regimes for F1 pit-stop strategy.

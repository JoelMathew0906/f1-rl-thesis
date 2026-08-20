# F1 RL Thesis: Multi-Architecture Safety and Reward Comparison

This repository hosts the thesis-focused version of the F1 RL safety project. It extends the original `f1-rl-safety` codebase into a multi-architecture, multi-reward comparison study for a single-car strategic Formula 1 race simulator calibrated on the 2024 British Grand Prix at Silverstone.

Unlike the exploratory prototype, this repository is intended as a cleaned research artefact for an MRes thesis, with legacy generated results moved out of the main tree and a clearer separation between code, configuration, and derived data.

## Thesis framing

The thesis asks how different reinforcement learning architectures learn pit-stop and driving-risk strategies under a fixed race environment and three objective regimes, and which architectural choices best balance performance, rule compliance, and safety. It also investigates which state variables drive decision-making under each architecture and regime, using surrogate-based SHAP analysis.

The current design crosses:

- **Five architectures:** PPO, A2C, DQN, SARSA, and REINFORCE.
- **Three regimes:** unconstrained performance, rulebook-aware, and safety-constrained.
- **Common environment:** a strategic single-car Silverstone race simulator calibrated on 2024 data.

All generated CSVs, HTML comparison pages, and SHAP images from the exploratory runs are preserved under `legacy/` to keep the code tree clean while retaining reproducibility.

## Repository structure

```text
f1-rl-thesis/
├── configs/                 Experiment configuration files
├── data/                    Calibration data and minimal example inputs
├── legacy/                  Archived CSVs, HTML outputs, and SHAP images from exploratory runs
├── logs/                    Training logs and TensorBoard outputs
├── models/                  Saved trained policies and checkpoints
├── notebooks/               Exploratory and thesis analysis notebooks
├── scripts/                 Training, evaluation, and analysis entry points
├── src/f1_rl_safety/        Core simulator, agents, and utilities
└── requirements.txt         Python dependencies
```

The code and configuration files mirror the structure of `f1-rl-safety`, while this README and `legacy/` reflect the thesis-specific comparison and the decision to quarantine generated artefacts.

## Legacy artefacts

The `legacy/` directory contains:

- Evaluation CSVs for the 5×3 architecture/regime grid.
- Aggregated comparison CSVs and interactive HTML plots.
- Per-agent and cross-architecture SHAP CSVs.
- SHAP summary PNGs and other images generated during exploratory analysis.

These files document the single-seed, fixed-budget exploratory experiment described in the thesis, but are intentionally kept out of the main code and data directories.

Future thesis runs with matched interaction budgets and multiple seeds should either reuse `legacy/` for archival or introduce versioned result folders under a dedicated `results/` directory.

## Limitations and next steps

The limitations described in the original `f1-rl-safety` README still apply: the simulator is single-track, single-car, and reward-driven; the current experiments are single-seed and exploratory; and SHAP explanations are surrogate-based rather than causal. The thesis repository’s role is to make these experiments reproducible and clearly documented.

As the thesis evolves, new experiments (e.g., matched interaction budgets, multi-seed runs, alternative architectures, mixture-of-experts controllers) should be added via configuration and scripts, keeping the core simulator and reward semantics stable.

## Origin

This repository was cloned from `https://github.com/JoelMathew0906/f1-rl-safety` and reorganised for thesis documentation and artefact management.

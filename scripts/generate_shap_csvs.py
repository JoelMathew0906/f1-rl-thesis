"""Driver for generating SHAP CSVs consumed by scripts/analyse_shap.py and
scripts/analyse_global_importance.py.

This script does not contain any SHAP logic itself. It only loops over
every (algo, regime, seed) combination and calls
``f1_rl_safety.shap_surrogates.train_surrogate_and_shap`` for each one,
pointing it at the correct trained-model directory for that algo and
writing the resulting CSV/PNG pair into ``data/shap/``.

Usage:
    PYTHONPATH=src .venv_f1/bin/python scripts/generate_shap_csvs.py

After it completes, run the aggregation scripts:
    PYTHONPATH=src .venv_f1/bin/python scripts/analyse_shap.py
    PYTHONPATH=src .venv_f1/bin/python scripts/analyse_global_importance.py
"""

from pathlib import Path

from f1_rl_safety.f1_env import RaceRegime
from f1_rl_safety.shap_surrogates import train_surrogate_and_shap

ALGOS = ["ppo", "a2c", "dqn", "sarsa", "reinforce"]
REGIMES = ["unconstrained", "rulebook", "safe"]
SEEDS = [0, 1, 2]

STEPS_OR_EPISODES = 200000
N_EPISODES = 100

MODEL_DIR = {
    "ppo": Path("outputs/experiments/models/final-v1"),
    "a2c": Path("outputs/experiments/models/final-v1"),
    "dqn": Path("outputs/experiments/models/final-v1"),
    "sarsa": Path("outputs/experiments/models"),
    "reinforce": Path("outputs/experiments/models"),
}

OUTPUT_DIR = Path("data/shap")


def main():
    n_ok = 0
    n_failed = 0

    for algo in ALGOS:
        for regime_name in REGIMES:
            for seed in SEEDS:
                regime = RaceRegime[regime_name.upper()]
                print(f"[{algo}, {regime_name}, seed={seed}] generating SHAP CSV...")
                try:
                    train_surrogate_and_shap(
                        algo=algo,
                        regime=regime,
                        seed=seed,
                        steps_or_episodes=STEPS_OR_EPISODES,
                        n_episodes=N_EPISODES,
                        model_dir=MODEL_DIR[algo],
                        output_dir=OUTPUT_DIR,
                    )
                    n_ok += 1
                except Exception as exc:
                    n_failed += 1
                    print(f"  ERROR [{algo}, {regime_name}, seed={seed}]: {exc}")

    print(f"\nDone. {n_ok} succeeded, {n_failed} failed.")


if __name__ == "__main__":
    main()

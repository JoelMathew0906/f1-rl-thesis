from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch

from .f1_env import F1RaceEnv, RaceRegime
from .wrappers import DiscreteF1ActionWrapper


def collect_state_action_data(
    algo: str,
    regime: RaceRegime,
    seed: int,
    steps_or_episodes: int,
    n_episodes: int,
    model_dir: Path,
    max_samples: int = 5000,
):
    """Collect (state, action) pairs from rollouts for surrogate training.

    Returns a pandas DataFrame with columns for state features and an
    action label suitable for supervised learning.
    """

    from .evaluate_rl import _make_env, _load_model, _select_action

    env = _make_env(regime, algo, seed)
    model = _load_model(algo, regime, seed, steps_or_episodes, model_dir)

    rows = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done and len(rows) < max_samples:
            action = _select_action(algo, model, obs, env)

            if algo in {"ppo", "a2c", "dqn", "reinforce"}:
                # For continuous actions, store risk; for discrete, store action index.
                if algo == "dqn" or isinstance(env, DiscreteF1ActionWrapper):
                    action_label = int(action)
                else:
                    # Box action: [pit_decision, tyre_choice, risk_level]
                    action_label = float(action[-1])  # risk component
            elif algo == "sarsa":
                action_label = int(action)
            else:
                raise ValueError(f"Unsupported algo: {algo}")

            row = {
                "action_label": action_label,
                "seed": seed,
            }

            # Flatten observation into named features
            obs_arr = np.array(obs, dtype=np.float32)
            for i, val in enumerate(obs_arr):
                row[f"s_{i}"] = float(val)

            rows.append(row)

            obs, _reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

        if len(rows) >= max_samples:
            break

    env.close()

    df = pd.DataFrame(rows)
    return df


def train_surrogate_and_shap(
    algo: str,
    regime: RaceRegime,
    seed: int,
    steps_or_episodes: int,
    n_episodes: int,
    model_dir: Path,
    output_dir: Path,
):
    """Train a simple surrogate model and compute SHAP values.

    For simplicity, we use a linear surrogate via shap.KernelExplainer
    on a subset of state features, focusing on relative importance.
    """

    df = collect_state_action_data(
        algo=algo,
        regime=regime,
        seed=seed,
        steps_or_episodes=steps_or_episodes,
        n_episodes=n_episodes,
        model_dir=model_dir,
    )

    feature_cols = [c for c in df.columns if c.startswith("s_")]
    X = df[feature_cols].values  # shape (n_samples, n_features)
    y = df["action_label"].values  # shape (n_samples,)

    # Define a simple linear surrogate in PyTorch
    input_dim = X.shape[1]
    model = torch.nn.Linear(input_dim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32).unsqueeze(-1)

    for _epoch in range(200):
        pred = model(X_tensor)
        loss = torch.mean((pred - y_tensor) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Wrap surrogate for SHAP
    def surrogate_fn(x_np):
        x_t = torch.as_tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            out = model(x_t).numpy().reshape(-1)
        return out

    # Use a subset of samples as background
    n_bg = min(200, len(X))
    background = X[np.random.choice(len(X), size=n_bg, replace=False)]
    explainer = shap.KernelExplainer(surrogate_fn, background)

    # Compute SHAP values for a subset of samples
    n_eval = min(100, len(X))
    shap_values = explainer.shap_values(X[:n_eval])  # shape (n_eval, n_features)

    # Aggregate mean |SHAP| per feature (1D arrays for pandas)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0).reshape(-1)

    shap_df = pd.DataFrame(
        {
            "feature": np.array(feature_cols),
            "mean_abs_shap": mean_abs_shap,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"shap_{algo}_{regime.name.lower()}_seed={seed}.csv"
    shap_df.to_csv(csv_path, index=False)

    # Also save a summary plot
    shap.summary_plot(
        shap_values,
        X[:n_eval],
        feature_names=feature_cols,
        show=False,
    )
    png_path = output_dir / f"shap_{algo}_{regime.name.lower()}_seed={seed}.png"
    import matplotlib.pyplot as plt

    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

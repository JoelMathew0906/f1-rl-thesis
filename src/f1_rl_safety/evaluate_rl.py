from collections import Counter
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, A2C, DQN

from .f1_env import F1RaceEnv, RaceRegime
from .wrappers import DiscreteF1ActionWrapper
from .value_based import QNetwork
from .reinforce_agent import PolicyNetwork


ALGOS = ["ppo", "a2c", "dqn", "sarsa", "reinforce"]


def _make_env(regime: RaceRegime, algo: str, seed: int):
    if algo == "dqn" or algo == "sarsa":
        return DiscreteF1ActionWrapper(regime=regime, seed=seed)
    else:
        return F1RaceEnv(regime=regime, seed=seed)


def _load_model(algo: str, regime: RaceRegime, seed: int, steps_or_episodes: int,
                model_dir: Path):
    if algo in {"ppo", "a2c", "dqn"}:
        path = model_dir / algo / regime.name.lower() / \
            f"{algo}_regime={regime.name.lower()}_seed={seed}_steps={steps_or_episodes}.zip"
        cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo]
        return cls.load(str(path))

    if algo == "sarsa":
        path = model_dir / "sarsa" / regime.name.lower() / \
            f"sarsa_regime={regime.name.lower()}_seed={seed}_steps={steps_or_episodes}.pt"
        # For evaluation we only need greedy actions, so we reconstruct QNetwork
        # with the expected input and action dimensions from the environment.
        env = DiscreteF1ActionWrapper(regime=regime, seed=seed)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n
        q_net = QNetwork(obs_dim, n_actions)
        state_dict = torch.load(path, map_location="cpu")
        q_net.load_state_dict(state_dict)
        q_net.eval()
        env.close()
        return q_net

    if algo == "reinforce":
        path = model_dir / "reinforce" / regime.name.lower() / \
            f"reinforce_regime={regime.name.lower()}_seed={seed}_episodes={steps_or_episodes}.pt"
        env = F1RaceEnv(regime=regime, seed=seed)
        obs_dim = env.observation_space.shape[0]
        policy = PolicyNetwork(obs_dim)
        state_dict = torch.load(path, map_location="cpu")
        policy.load_state_dict(state_dict)
        policy.eval()
        env.close()
        return policy

    raise ValueError(f"Unsupported algo: {algo}")


def _select_action(algo: str, model, obs, env):
    if algo in {"ppo", "a2c", "dqn"}:
        action, _ = model.predict(obs, deterministic=True)
        return action

    if algo == "sarsa":
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = model(obs_tensor)[0]
        action = int(torch.argmax(q_values).item())
        return action

    if algo == "reinforce":
        from .reinforce_agent import sample_action

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        pit_logits, tyre_logits, risk_mean, risk_log_std = model(obs_tensor)
        action_tensor, _log_prob = sample_action(
            pit_logits, tyre_logits, risk_mean, risk_log_std
        )
        return action_tensor.detach().cpu().numpy()[0]

    raise ValueError(f"Unsupported algo: {algo}")


def evaluate_model(
    algo: str,
    regime: RaceRegime,
    seed: int,
    steps_or_episodes: int,
    n_episodes: int,
    model_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a trained model/agent over n_episodes.

    Returns summary statistics matching the original PPO evaluation
    script: finish_position, race_time, crashes, catastrophic,
    pitstops, mean_risk, pitstop_distribution.
    """

    env = _make_env(regime, algo, seed)
    model = _load_model(algo, regime, seed, steps_or_episodes, model_dir)

    stats = {
        "finish_position": [],
        "race_time": [],
        "crashes": [],
        "catastrophic": [],
        "pitstops": [],
        "mean_risk": [],
    }

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_risk = []

        while not done:
            action = _select_action(algo, model, obs, env)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_risk.append(info["risk_level"])

        if isinstance(env, F1RaceEnv):
            finish_pos = int(env.position)
            race_time = env.race_time
            crashes = int(env.crashed)
            catastrophic = int(env.catastrophic_event)
            pitstops = int(env.pit_count)
        else:
            # Discrete wrapper exposes underlying env via _env
            finish_pos = int(env._env.position)
            race_time = env._env.race_time
            crashes = int(env._env.crashed)
            catastrophic = int(env._env.catastrophic_event)
            pitstops = int(env._env.pit_count)

        stats["finish_position"].append(finish_pos)
        stats["race_time"].append(race_time)
        stats["crashes"].append(crashes)
        stats["catastrophic"].append(catastrophic)
        stats["pitstops"].append(pitstops)
        stats["mean_risk"].append(float(np.mean(ep_risk)) if ep_risk else 0.0)

    summary = {k: float(np.mean(v)) for k, v in stats.items()}
    summary["pitstop_distribution"] = dict(Counter(stats["pitstops"]))

    env.close()

    return summary


def evaluate_grid(
    algo: str,
    regime: RaceRegime,
    seeds: list[int],
    steps_or_episodes: int,
    n_episodes: int,
    model_dir: Path,
    output_csv: Path,
):
    """Evaluate multiple seeds for a given algo/regime and export CSV summary."""

    rows = []
    for seed in seeds:
        summary = evaluate_model(
            algo=algo,
            regime=regime,
            seed=seed,
            steps_or_episodes=steps_or_episodes,
            n_episodes=n_episodes,
            model_dir=model_dir,
        )
        row = {
            "algo": algo,
            "regime": regime.name.lower(),
            "seed": seed,
            "steps_or_episodes": steps_or_episodes,
            **summary,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

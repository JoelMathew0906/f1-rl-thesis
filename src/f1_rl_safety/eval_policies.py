from collections import Counter
import numpy as np
from stable_baselines3 import PPO
from src.f1_rl_safety.f1_env import F1RaceEnv, RaceRegime


def evaluate_policy(model_path: str, regime: RaceRegime, n_episodes: int = 100, seed: int = 0):
    env = F1RaceEnv(regime=regime, seed=seed)
    model = PPO.load(model_path)

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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_risk.append(info["risk_level"])

        stats["finish_position"].append(int(env.position))
        stats["race_time"].append(env.race_time)
        stats["crashes"].append(int(env.crashed))
        stats["catastrophic"].append(int(env.catastrophic_event))
        stats["pitstops"].append(int(env.pit_count))
        stats["mean_risk"].append(float(np.mean(ep_risk)) if ep_risk else 0.0)

    summary = {k: float(np.mean(v)) for k, v in stats.items()}
    summary["pitstop_distribution"] = dict(Counter(stats["pitstops"]))
    return summary


if __name__ == "__main__":
    for regime in [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]:
        model_path = f"models/ppo_{regime.name.lower()}.zip"
        summary = evaluate_policy(model_path, regime)
        print(regime.name, summary)
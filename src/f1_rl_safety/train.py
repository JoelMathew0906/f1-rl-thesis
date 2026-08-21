import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.f1_rl_safety.f1_env import F1RaceEnv, RaceRegime


def make_env(regime: RaceRegime):
    def _init():
        return F1RaceEnv(regime=regime)
    return _init


def train_regime(regime: RaceRegime, total_timesteps: int = 200_000):
    env = DummyVecEnv([make_env(regime)])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"logs/{regime.name.lower()}",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(f"models/ppo_{regime.name.lower()}.zip")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    for regime in [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]:
        print(f"Training regime: {regime.name}")
        train_regime(regime, total_timesteps=args.steps)
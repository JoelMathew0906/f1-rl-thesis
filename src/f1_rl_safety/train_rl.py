import argparse
from pathlib import Path

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from .f1_env import F1RaceEnv, RaceRegime
from .wrappers import DiscreteF1ActionWrapper
from .value_based import train_sarsa
from .reinforce_agent import train_reinforce


ALGOS = ["ppo", "a2c", "dqn", "sarsa", "reinforce"]
REGIMES = {
    "unconstrained": RaceRegime.UNCONSTRAINED,
    "rulebook": RaceRegime.RULEBOOK,
    "safe": RaceRegime.SAFE,
}


def make_env(regime: RaceRegime, discrete: bool = False, seed: int | None = None):
    if discrete:
        def _init():
            return DiscreteF1ActionWrapper(regime=regime, seed=seed)
        return _init
    else:
        def _init():
            return F1RaceEnv(regime=regime, seed=seed)
        return _init


def train_sb3_algo(
    algo: str,
    regime: RaceRegime,
    total_timesteps: int,
    seed: int,
    log_dir: Path,
    model_dir: Path,
):
    """Train a PPO/A2C/DQN agent on the F1RaceEnv.

    DQN uses the DiscreteF1ActionWrapper; PPO and A2C use the original
    continuous action space.
    """
    discrete = algo == "dqn"
    env = DummyVecEnv([make_env(regime, discrete=discrete, seed=seed)])

    tensorboard_log = log_dir / algo / regime.name.lower() / f"seed_{seed}"
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    if algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(tensorboard_log))
    elif algo == "a2c":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=str(tensorboard_log))
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=str(tensorboard_log))
    else:
        raise ValueError(f"Unsupported SB3 algo for this helper: {algo}")

    model.learn(total_timesteps=total_timesteps)

    model_dir = model_dir / algo / regime.name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algo}_regime={regime.name.lower()}_seed={seed}_steps={total_timesteps}.zip"
    model.save(str(model_path))

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Unified training entrypoint for RL architectures."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=ALGOS,
        required=True,
        help="RL algorithm: ppo, a2c, dqn, sarsa, reinforce",
    )
    parser.add_argument(
        "--regime",
        type=str,
        choices=list(REGIMES.keys()),
        required=True,
        help="Reward regime: unconstrained, rulebook, safe",
    )
    parser.add_argument(
        "--steps", type=int, default=50_000, help="Total environment timesteps."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Total episodes for REINFORCE (ignored for SB3/SARSA).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for environment and agent."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Base directory for TensorBoard logs.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Base directory for saved models.",
    )

    args = parser.parse_args()

    algo = args.algo.lower()
    regime = REGIMES[args.regime]
    total_timesteps = args.steps
    total_episodes = args.episodes
    seed = args.seed

    log_dir = Path(args.log_dir)
    model_dir = Path(args.model_dir)

    if algo in {"ppo", "a2c", "dqn"}:
        train_sb3_algo(algo, regime, total_timesteps, seed, log_dir, model_dir)
    elif algo == "sarsa":
        train_sarsa(regime, total_timesteps, seed, log_dir, model_dir)
    elif algo == "reinforce":
        train_reinforce(regime, total_episodes, seed, log_dir, model_dir)
    else:
        raise ValueError(f"Unsupported algo: {algo}")


if __name__ == "__main__":
    main()

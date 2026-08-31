import argparse
import random
from pathlib import Path

import numpy as np
import torch
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


def set_global_seeds(seed: int) -> None:
    """Set Python, NumPy and Torch seeds for reproducibility.

    This helps ensure that training runs are comparable across
    algorithms when using the same seed list.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    run_name: str,
    resume: bool = False,
    gamma: float = 0.999,
) -> Path:
    """Train a PPO/A2C/DQN agent on the F1RaceEnv.

    DQN uses the DiscreteF1ActionWrapper; PPO and A2C use the original
    continuous action space.

    The returned Path points to the saved model checkpoint.
    """

    set_global_seeds(seed)

    discrete = algo == "dqn"
    env = DummyVecEnv([make_env(regime, discrete=discrete, seed=seed)])

    # Logs: outputs/{mode}/logs/{run_name}/{algo}/{regime}/seed_{seed}
    tb_log_base = log_dir / run_name / algo / regime.name.lower() / f"seed_{seed}"
    tb_log_base.mkdir(parents=True, exist_ok=True)

    # Models: outputs/{mode}/models/{run_name}/{algo}/{regime}/...
    model_base = model_dir / run_name / algo / regime.name.lower()
    model_base.mkdir(parents=True, exist_ok=True)

    model_path = model_base / (
        f"{algo}_regime={regime.name.lower()}_seed={seed}_steps={total_timesteps}.zip"
    )

    if algo == "ppo":
        if resume and model_path.exists():
            # Explicit resume: load existing checkpoint and continue training,
            # retaining the checkpoint's own gamma rather than the CLI value.
            model = PPO.load(str(model_path), env=env)
        else:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=str(tb_log_base), seed=seed, gamma=gamma)
    elif algo == "a2c":
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=str(tb_log_base), seed=seed, gamma=gamma)
    elif algo == "dqn":
        model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=str(tb_log_base), seed=seed, gamma=gamma)
    else:
        raise ValueError(f"Unsupported SB3 algo for this helper: {algo}")

    model.learn(total_timesteps=total_timesteps)

    model.save(str(model_path))
    env.close()

    return model_path


def evaluate_ppo_smoke(
    model_path: Path,
    regime: RaceRegime,
    seed: int,
    eval_episodes: int,
    eval_dir: Path,
    run_name: str,
    total_timesteps: int,
) -> None:
    """Lightweight PPO-only evaluation for smoke tests.

    Runs a small number of deterministic evaluation episodes and
    exports two CSV files under eval_dir / run_name:

    - a single-row summary CSV with aggregate metrics
    - a crash-log CSV with one row per crash event (lap, segment/corner)
    """

    from stable_baselines3 import PPO as PPOCls

    eval_dir_run = eval_dir / run_name
    eval_dir_run.mkdir(parents=True, exist_ok=True)

    env = F1RaceEnv(regime=regime, seed=seed)
    model = PPOCls.load(str(model_path))

    episode_rows: list[dict] = []
    crash_rows: list[dict] = []

    for ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_risk: list[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            ep_risk.append(float(info["risk_level"]))

        # Per-episode summary
        episode_rows.append(
            {
                "algo": "ppo",
                "regime": regime.name.lower(),
                "run_name": run_name,
                "seed": seed,
                "timesteps": total_timesteps,
                "episode": ep,
                "total_reward": ep_reward,
                "finish_position": int(env.position),
                "race_time": float(env.race_time),
                "crashes": int(env.crashed),
                "catastrophic": int(env.catastrophic_event),
                "pitstops": int(env.pit_count),
                "mean_risk": float(np.mean(ep_risk)) if ep_risk else 0.0,
            }
        )

        # Per-crash logs for this episode (if any)
        for crash_event in env.crash_log:
            crash_rows.append(
                {
                    "algo": "ppo",
                    "regime": regime.name.lower(),
                    "run_name": run_name,
                    "seed": seed,
                    "timesteps": total_timesteps,
                    "episode": ep,
                    "lap": crash_event.get("lap"),
                    "segment_id": crash_event.get("segment_id"),
                    "corner_number": crash_event.get("corner_number"),
                    "corner_name": crash_event.get("corner_name"),
                }
            )

    import pandas as pd

    summary_path = eval_dir_run / (
        f"ppo_regime={regime.name.lower()}_seed={seed}_steps={total_timesteps}_"
        f"run={run_name}_summary.csv"
    )
    pd.DataFrame(episode_rows).to_csv(summary_path, index=False)

    if crash_rows:
        crash_path = eval_dir_run / (
            f"ppo_regime={regime.name.lower()}_seed={seed}_steps={total_timesteps}_"
            f"run={run_name}_crashes.csv"
        )
        pd.DataFrame(crash_rows).to_csv(crash_path, index=False)

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
        "--reward-regime",
        dest="regime",
        type=str,
        choices=list(REGIMES.keys()),
        required=True,
        help="Reward regime: unconstrained, rulebook, safe",
    )
    parser.add_argument(
        "--steps",
        "--timesteps",
        dest="steps",
        type=int,
        default=50_000,
        help="Total environment timesteps (for SB3/SARSA).",
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
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Base output directory for this run. When omitted, debug runs use "
            "'outputs/debug' and experiment runs use 'outputs/experiments'."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["debug", "experiment"],
        default="debug",
        help=(
            "Run mode: 'debug' for smoke tests (outputs/debug), 'experiment' for "
            "thesis runs (outputs/experiments)."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help=(
            "Short identifier for this run; used to namespace logs/models/eval "
            "within the chosen output directory."
        ),
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help=(
            "Discount factor shared by PPO, DQN, A2C, SARSA and REINFORCE. "
            "Must satisfy 0 < gamma <= 1. For --resume loads, the checkpoint's "
            "own gamma is kept instead of this value."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume PPO training from an existing checkpoint if present. "
            "Ignored for other algorithms."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help=(
            "Optional PPO-only smoke-test evaluation episodes to run immediately "
            "after training. If >0, results are written under outputs/.../eval."
        ),
    )

    args = parser.parse_args()

    if not (0.0 < args.gamma <= 1.0):
        parser.error(f"--gamma must be in (0, 1], got {args.gamma}")

    algo = args.algo.lower()
    regime = REGIMES[args.regime]
    total_timesteps = args.steps
    total_episodes = args.episodes
    seed = args.seed
    run_name = args.run_name
    mode = args.mode

    # Determine base output directory
    if args.output_dir is not None:
        base_out = Path(args.output_dir)
    else:
        base_out = Path("outputs") / ("debug" if mode == "debug" else "experiments")

    log_dir = base_out / "logs"
    model_dir = base_out / "models"
    eval_dir = base_out / "eval"

    # Ensure top-level output structure exists
    for d in (log_dir, model_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Set global seeds once per process
    set_global_seeds(seed)

    if algo in {"ppo", "a2c", "dqn"}:
        model_path = train_sb3_algo(
            algo,
            regime,
            total_timesteps,
            seed,
            log_dir,
            model_dir,
            run_name,
            resume=args.resume,
            gamma=args.gamma,
        )

        # Optional PPO-only smoke-test evaluation into outputs/.../eval
        if algo == "ppo" and args.eval_episodes > 0:
            evaluate_ppo_smoke(
                model_path=model_path,
                regime=regime,
                seed=seed,
                eval_episodes=args.eval_episodes,
                eval_dir=eval_dir,
                run_name=run_name,
                total_timesteps=total_timesteps,
            )

    elif algo == "sarsa":
        train_sarsa(regime, total_timesteps, seed, log_dir, model_dir, gamma=args.gamma)
    elif algo == "reinforce":
        train_reinforce(regime, total_episodes, seed, log_dir, model_dir, gamma=args.gamma)
    else:
        raise ValueError(f"Unsupported algo: {algo}")


if __name__ == "__main__":
    main()

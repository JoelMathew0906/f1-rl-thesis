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


def _unwrap_env(env):
    """Return underlying F1RaceEnv whether raw or wrapped."""
    if isinstance(env, F1RaceEnv):
        return env
    if hasattr(env, "_env") and isinstance(env._env, F1RaceEnv):
        return env._env
    return env


def _format_used_compounds(base_env: F1RaceEnv) -> str:
    """Convert used_compounds into a deterministic, CSV-safe string."""
    compounds = getattr(base_env, "used_compounds", None)
    if not compounds:
        return ""

    names = []
    idx_to_compound = getattr(base_env, "idx_to_compound", {})
    for idx in sorted(compounds):
        names.append(str(idx_to_compound.get(idx, idx)))
    return ";".join(names)


def _format_crash_site(crash_record: Dict[str, Any] | None) -> str:
    """Format crash site into a deterministic representation string."""
    if not crash_record:
        return ""

    seg_id = crash_record.get("segment_id", "")
    seg_type = crash_record.get("segment_type", "")
    corner_number = crash_record.get("corner_number") or ""
    corner_name = crash_record.get("corner_name") or ""

    return (
        f"segment={seg_id};"
        f"type={seg_type};"
        f"corner_number={corner_number};"
        f"corner_name={corner_name}"
    )


def _episode_rows(
    algo: str,
    regime: RaceRegime,
    seed: int,
    steps_or_episodes: int,
    base_env: F1RaceEnv,
    ep_risk: list[float],
    crash_log: list[dict[str, Any]],
    pitstop_distribution: Dict[int, int],
    evaluation_episode: int,
    evaluation_env_seed: int,
) -> list[Dict[str, Any]]:
    """Build CSV rows for a single evaluation episode.

    - No crash: one row with crash columns empty.
    - One or more crashes: one row per crash event.
    """
    completed_laps = getattr(base_env, "current_lap", 0)
    n_laps = getattr(base_env, "n_laps", None)
    crashed = bool(getattr(base_env, "crashed", False))
    pit_count = int(getattr(base_env, "pit_count", 0))
    cat_event = bool(getattr(base_env, "catastrophic_event", False))

    terminated_by_crash = bool(
        crashed and (n_laps is not None) and (completed_laps < n_laps)
    )

    used_compounds_str = _format_used_compounds(base_env)

    first_crash = crash_log[0] if crash_log else None
    final_crash = crash_log[-1] if crash_log else None

    first_crash_site = _format_crash_site(first_crash)
    final_crash_site = _format_crash_site(final_crash)

    first_crash_reason = first_crash.get("crash_reason") if first_crash else None
    final_crash_reason = final_crash.get("crash_reason") if final_crash else None

    mean_risk = float(np.mean(ep_risk)) if ep_risk else 0.0

    # reward component totals; older checkpoints may not have these attributes
    reward_time_total = getattr(base_env, "reward_time_total", None)
    reward_risk_total = getattr(base_env, "reward_risk_total", None)
    reward_crash_total = getattr(base_env, "reward_crash_total", None)
    reward_pit_total = getattr(base_env, "reward_pit_total", None)
    reward_compound_total = getattr(base_env, "reward_compound_total", None)
    reward_compliance_total = getattr(base_env, "reward_compliance_total", None)
    reward_lap_completion_total = getattr(base_env, "reward_lap_completion_total", None)
    episode_return = getattr(base_env, "episode_return", None)

    common = {
        # original aggregate-like fields (now per-episode)
        "algo": algo,
        "regime": regime.name.lower(),
        "seed": seed,
        "steps_or_episodes": steps_or_episodes,
        "evaluation_episode": evaluation_episode,
        "evaluation_env_seed": evaluation_env_seed,
        "finish_position": int(getattr(base_env, "position", 0)),
        "race_time": float(getattr(base_env, "race_time", 0.0)),
        "crashes": int(crashed),
        "catastrophic": int(cat_event),
        "pitstops": pit_count,
        "mean_risk": mean_risk,
        "pitstop_distribution": pitstop_distribution,
        # episode-level fields
        "algorithm": algo,
        "training_budget": steps_or_episodes,
        "completed_laps": completed_laps,
        "terminated_by_crash": terminated_by_crash,
        "pit_stops": pit_count,
        "used_compounds": used_compounds_str,
        "first_crash_site": first_crash_site or "",
        "first_crash_reason": first_crash_reason or "",
        "final_crash_site": final_crash_site or "",
        "final_crash_reason": final_crash_reason or "",
        # reward component totals (may be None for older checkpoints)
        "reward_time_total": reward_time_total,
        "reward_risk_total": reward_risk_total,
        "reward_crash_total": reward_crash_total,
        "reward_pit_total": reward_pit_total,
        "reward_compound_total": reward_compound_total,
        "reward_compliance_total": reward_compliance_total,
        "reward_lap_completion_total": reward_lap_completion_total,
        "episode_return": episode_return,
    }

    rows: list[Dict[str, Any]] = []

    if crash_log:
        for idx, crash in enumerate(crash_log):
            row = dict(common)
            row.update(
                {
                    "crash_index": idx,
                    "crash_lap": crash.get("lap"),
                    "crash_segment_id": crash.get("segment_id"),
                    "crash_segment_type": crash.get("segment_type"),
                    "crash_corner_number": crash.get("corner_number"),
                    "crash_corner_name": crash.get("corner_name"),
                    "crash_tyre_compound": crash.get("tyre_compound"),
                    "crash_tyre_age": crash.get("tyre_age"),
                    "crash_tyre_wear": crash.get("tyre_wear"),
                    "crash_risk_level": crash.get("risk_level"),
                    "crash_reason": crash.get("crash_reason"),
                }
            )
            rows.append(row)
    else:
        row = dict(common)
        row.update(
            {
                "crash_index": None,
                "crash_lap": None,
                "crash_segment_id": None,
                "crash_segment_type": None,
                "crash_corner_number": None,
                "crash_corner_name": None,
                "crash_tyre_compound": None,
                "crash_tyre_age": None,
                "crash_tyre_wear": None,
                "crash_risk_level": None,
                "crash_reason": None,
            }
        )
        rows.append(row)

    return rows


def _load_model(algo: str, regime: RaceRegime, seed: int, steps_or_episodes: int,
                model_dir: Path, run_name: str | None = None):
    if algo in {"ppo", "a2c", "dqn"}:
        sb3_base = (model_dir / run_name) if run_name else model_dir
        path = sb3_base / algo / regime.name.lower() / \
            f"{algo}_regime={regime.name.lower()}_seed={seed}_steps={steps_or_episodes}.zip"
        cls = {"ppo": PPO, "a2c": A2C, "dqn": DQN}[algo]

        # For PPO, fall back to top-level smoke-test models if detailed path
        # does not exist, without changing filenames.
        if not path.exists() and algo == "ppo":
            fallback_path = sb3_base / f"ppo_{regime.name.lower()}.zip"
            return cls.load(str(fallback_path))

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
        base = model_dir / "reinforce" / regime.name.lower()
        # New convention (steps-budgeted, the default since the REINFORCE
        # --steps support was added) takes priority; fall back to the
        # legacy episodes-budgeted filename for older checkpoints.
        steps_path = base / (
            f"reinforce_regime={regime.name.lower()}_seed={seed}_steps={steps_or_episodes}.pt"
        )
        episodes_path = base / (
            f"reinforce_regime={regime.name.lower()}_seed={seed}_episodes={steps_or_episodes}.pt"
        )
        path = steps_path if steps_path.exists() else episodes_path
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
        from .reinforce_agent import select_action_deterministic

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        pit_logits, tyre_logits, risk_mean, risk_log_std = model(obs_tensor)
        action_tensor = select_action_deterministic(
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
    run_name: str | None = None,
    eval_seed_base: int = 10000,
) -> list[Dict[str, Any]]:
    """Evaluate a trained model/agent over n_episodes.

    Returns a list of per-episode / per-crash rows combining
    aggregate summary fields with crash-level and episode-level
    information, including reward component totals when available.

    Evaluation-episode environment reset seeds are drawn from
    `eval_seed_base + ep` (ep = 0..n_episodes-1), independent of the
    training/checkpoint `seed`, so every evaluated model can share an
    identical evaluation-seed schedule.
    """

    env = _make_env(regime, algo, seed)
    model = _load_model(algo, regime, seed, steps_or_episodes, model_dir, run_name=run_name)

    stats = {
        "finish_position": [],
        "race_time": [],
        "crashes": [],
        "catastrophic": [],
        "pitstops": [],
        "mean_risk": [],
    }

    all_rows: list[Dict[str, Any]] = []

    for ep in range(n_episodes):
        evaluation_env_seed = eval_seed_base + ep
        obs, _ = env.reset(seed=evaluation_env_seed)
        done = False
        ep_risk: list[float] = []

        while not done:
            action = _select_action(algo, model, obs, env)
            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_risk.append(info["risk_level"])

        base_env = _unwrap_env(env)

        finish_pos = int(getattr(base_env, "position", 0))
        race_time = float(getattr(base_env, "race_time", 0.0))
        crashes = int(getattr(base_env, "crashed", False))
        catastrophic = int(getattr(base_env, "catastrophic_event", False))
        pitstops = int(getattr(base_env, "pit_count", 0))

        stats["finish_position"].append(finish_pos)
        stats["race_time"].append(race_time)
        stats["crashes"].append(crashes)
        stats["catastrophic"].append(catastrophic)
        stats["pitstops"].append(pitstops)
        stats["mean_risk"].append(float(np.mean(ep_risk)) if ep_risk else 0.0)

        crash_log = getattr(base_env, "crash_log", []) or []

        episode_rows = _episode_rows(
            algo=algo,
            regime=regime,
            seed=seed,
            steps_or_episodes=steps_or_episodes,
            base_env=base_env,
            ep_risk=ep_risk,
            crash_log=crash_log,
            pitstop_distribution={},  # placeholder; overwritten below
            evaluation_episode=ep,
            evaluation_env_seed=evaluation_env_seed,
        )
        all_rows.extend(episode_rows)

    pitstop_distribution = dict(Counter(stats["pitstops"]))
    for row in all_rows:
        row["pitstop_distribution"] = pitstop_distribution

    env.close()

    return all_rows


def evaluate_grid(
    algo: str,
    regime: RaceRegime,
    seeds: list[int],
    steps_or_episodes: int,
    n_episodes: int,
    model_dir: Path,
    output_csv: Path,
    run_name: str | None = None,
    eval_seed_base: int = 10000,
):
    """Evaluate multiple seeds for a given algo/regime and export CSV.

    The CSV contains both aggregate-style fields and detailed crash-level
    information, with one row per episode when no crash, and one row per
    crash when crashes occur.
    """

    rows: list[Dict[str, Any]] = []
    for seed in seeds:
        seed_rows = evaluate_model(
            algo=algo,
            regime=regime,
            seed=seed,
            steps_or_episodes=steps_or_episodes,
            n_episodes=n_episodes,
            model_dir=model_dir,
            run_name=run_name,
            eval_seed_base=eval_seed_base,
        )
        rows.extend(seed_rows)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

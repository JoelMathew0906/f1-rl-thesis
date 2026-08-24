"""Diagnostic script to audit environment control, crash hazard sensitivity,
reward ordering, and PPO action distributions in F1RaceEnv.

This script is read-only with respect to training/evaluation APIs and
models. It runs four diagnostics:

A. Forced action-transition matrix
B. Controlled crash-hazard sensitivity
C. Scripted-policy reward ordering
D. PPO action-distribution audit

Outputs human-readable console logs and CSV files under outputs/debug/diagnostics.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.f1_rl_safety.f1_env import F1RaceEnv, RaceRegime
from src.f1_rl_safety.wrappers import DiscreteF1ActionWrapper
from src.f1_rl_safety.evaluate_rl import _unwrap_env


BASE_OUT = Path("outputs") / "debug" / "diagnostics"
BASE_OUT.mkdir(parents=True, exist_ok=True)


@dataclass
class TransitionRecord:
    action_index: int
    action_vector: Tuple[float, float, float]
    regime: str
    pre_lap: int
    post_lap: int
    pre_segment_idx: int
    post_segment_idx: int
    pre_pit_count: int
    post_pit_count: int
    pre_used_compounds: str
    post_used_compounds: str
    pre_tyre_compound: int
    post_tyre_compound: int
    pre_tyre_age: int
    post_tyre_age: int
    pre_risk_level: float
    post_risk_level: float
    reward: float
    terminated: bool
    truncated: bool
    crash: bool


def format_used_compounds(env: F1RaceEnv) -> str:
    compounds = getattr(env, "used_compounds", set())
    idx_to_compound = getattr(env, "idx_to_compound", {})
    names = [str(idx_to_compound.get(idx, idx)) for idx in sorted(compounds)]
    return ";".join(names)


def forced_action_transition_matrix(regime: RaceRegime) -> Path:
    """Diagnostic A: force a grid of actions and log state deltas.

    Uses the underlying continuous F1RaceEnv action space:
    [pit_decision (0/1), tyre_choice (0-4), risk_level (-1..1)].
    """
    out_path = BASE_OUT / f"forced_transitions_{regime.name.lower()}.csv"

    records: List[TransitionRecord] = []
    env = F1RaceEnv(regime=regime, seed=0)

    # Define a compact set of diagnostic actions.
    pit_options = [0.0, 1.0]
    tyre_options = [0.0, 1.0, 2.0, 3.0, 4.0]
    risk_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

    action_index = 0
    for pit in pit_options:
        for tyre in tyre_options:
            for risk in risk_levels:
                # reset to a comparable state before each forced action
                obs, _ = env.reset(seed=0)

                pre_lap = env.current_lap
                pre_seg = env.current_segment_idx
                pre_pit = env.pit_count
                pre_used = format_used_compounds(env)
                pre_compound = env.tyre_compound
                pre_age = env.tyre_age
                pre_risk = env.last_risk_level

                action = np.array([pit, tyre, risk], dtype=np.float32)
                obs2, reward, terminated, truncated, info = env.step(action)

                post_lap = env.current_lap
                post_seg = env.current_segment_idx
                post_pit = env.pit_count
                post_used = format_used_compounds(env)
                post_compound = env.tyre_compound
                post_age = env.tyre_age
                post_risk = env.last_risk_level
                crash = bool(info.get("crash", False))

                records.append(
                    TransitionRecord(
                        action_index=action_index,
                        action_vector=(float(pit), float(tyre), float(risk)),
                        regime=regime.name.lower(),
                        pre_lap=pre_lap,
                        post_lap=post_lap,
                        pre_segment_idx=pre_seg,
                        post_segment_idx=post_seg,
                        pre_pit_count=pre_pit,
                        post_pit_count=post_pit,
                        pre_used_compounds=pre_used,
                        post_used_compounds=post_used,
                        pre_tyre_compound=pre_compound,
                        post_tyre_compound=post_compound,
                        pre_tyre_age=pre_age,
                        post_tyre_age=post_age,
                        pre_risk_level=pre_risk,
                        post_risk_level=post_risk,
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        crash=crash,
                    )
                )

                action_index += 1

    env.close()

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(out_path, index=False)
    print(f"[A] Forced action transitions written to {out_path}")
    return out_path


@dataclass
class HazardRecord:
    regime: str
    description: str
    tyre_compound: int
    tyre_age: int
    tyre_wear: float
    segment_type: str
    approx_radius: float | None
    risk_level: float
    crash_prob: float


def controlled_hazard_sensitivity(regime: RaceRegime) -> Path:
    """Diagnostic B: crash-probability sensitivity to state and action.

    Uses F1RaceEnv._segment_crash_prob on representative configurations.
    """
    out_path = BASE_OUT / f"hazard_sensitivity_{regime.name.lower()}.csv"

    env = F1RaceEnv(regime=regime, seed=0)
    segments = env.track_segments

    # pick first corner and first straight
    corner = next(s for s in segments if str(s.segment_type).lower() == "corner")
    straight = next(s for s in segments if str(s.segment_type).lower() == "straight")

    scenarios: List[Tuple[str, TrackSegment, int, int, float, float]] = []

    # description, segment, tyre_compound, tyre_age, tyre_wear, risk_level
    scenarios.append(("lowest risk, fresh tyre, corner", corner, 1, 0, 0.0, -1.0))
    scenarios.append(("highest risk, fresh tyre, corner", corner, 1, 0, 0.0, 1.0))
    scenarios.append(("lowest risk, worn tyre, corner", corner, 1, 24, 1.0, -1.0))
    scenarios.append(("highest risk, worn tyre, corner", corner, 1, 24, 1.0, 1.0))
    scenarios.append(("lowest risk, fresh tyre, straight", straight, 1, 0, 0.0, -1.0))
    scenarios.append(("highest risk, fresh tyre, straight", straight, 1, 0, 0.0, 1.0))

    records: List[HazardRecord] = []

    for desc, seg, compound, age, wear, risk in scenarios:
        env.tyre_compound = compound
        env.tyre_age = age
        env.tyre_wear = wear
        prob = env._segment_crash_prob(seg, risk)
        records.append(
            HazardRecord(
                regime=regime.name.lower(),
                description=desc,
                tyre_compound=compound,
                tyre_age=age,
                tyre_wear=wear,
                segment_type=str(seg.segment_type).lower(),
                approx_radius=getattr(seg, "approx_radius", None),
                risk_level=risk,
                crash_prob=prob,
            )
        )

    env.close()

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(out_path, index=False)
    print(f"[B] Hazard sensitivity written to {out_path}")
    return out_path


@dataclass
class EpisodeSummary:
    regime: str
    policy_name: str
    seed: int
    episode_index: int
    crashed: bool
    crash_lap: int | None
    crash_segment_id: int | None
    pitstops: int
    used_compounds: str
    race_time: float
    finish_position: int
    reward_time_total: float | None
    reward_risk_total: float | None
    reward_crash_total: float | None
    reward_pit_total: float | None
    reward_compound_total: float | None
    reward_compliance_total: float | None
    reward_lap_completion_total: float | None
    episode_return: float | None
    mean_risk: float


def run_scripted_policy(env: F1RaceEnv, policy_name: str, seed: int) -> EpisodeSummary:
    """Run a simple deterministic policy on a single episode.

    Policies:
    - "aggressive_zero_stop": high risk, never pit
    - "one_stop_compliant": pit once in mid-race, switch compound
    - "conservative_two_stop": low risk, two pits
    """
    obs, _ = env.reset(seed=seed)
    done = False
    ep_risk: List[float] = []
    crash_lap = None
    crash_segment_id = None

    while not done:
        # decode current lap to drive scripted logic
        lap = env.current_lap

        if policy_name == "aggressive_zero_stop":
            pit_decision = 0.0
            tyre_choice = float(env.tyre_compound)
            risk_level = 0.9
        elif policy_name == "one_stop_compliant":
            # pit once between 20-30 laps, switch compound
            if 20 <= lap <= 30 and env.pit_count == 0:
                pit_decision = 1.0
                # switch from MEDIUM to SOFT or HARD deterministically
                tyre_choice = float(0 if env.tyre_compound != 0 else 2)
            else:
                pit_decision = 0.0
                tyre_choice = float(env.tyre_compound)
            risk_level = 0.4
        elif policy_name == "conservative_two_stop":
            # pit twice: once ~15, once ~35, lower risk
            if (15 <= lap <= 20 and env.pit_count == 0) or (
                35 <= lap <= 40 and env.pit_count == 1
            ):
                pit_decision = 1.0
                tyre_choice = float(env.tyre_compound)
            else:
                pit_decision = 0.0
                tyre_choice = float(env.tyre_compound)
            risk_level = -0.2
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        action = np.array([pit_decision, tyre_choice, risk_level], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_risk.append(float(info["risk_level"]))

        if crash_lap is None and info.get("crash", False):
            crash_lap = env.current_lap
            crash_segment_id = info.get("segment_id")

    base_env = env

    used = format_used_compounds(base_env)
    summary = EpisodeSummary(
        regime=base_env.regime.name.lower(),
        policy_name=policy_name,
        seed=seed,
        episode_index=0,
        crashed=bool(base_env.crashed),
        crash_lap=crash_lap,
        crash_segment_id=crash_segment_id,
        pitstops=int(base_env.pit_count),
        used_compounds=used,
        race_time=float(base_env.race_time),
        finish_position=int(base_env.position),
        reward_time_total=getattr(base_env, "reward_time_total", None),
        reward_risk_total=getattr(base_env, "reward_risk_total", None),
        reward_crash_total=getattr(base_env, "reward_crash_total", None),
        reward_pit_total=getattr(base_env, "reward_pit_total", None),
        reward_compound_total=getattr(base_env, "reward_compound_total", None),
        reward_compliance_total=getattr(base_env, "reward_compliance_total", None),
        reward_lap_completion_total=getattr(base_env, "reward_lap_completion_total", None),
        episode_return=getattr(base_env, "episode_return", None),
        mean_risk=float(np.mean(ep_risk)) if ep_risk else 0.0,
    )
    return summary


def scripted_policy_reward_ordering(regime: RaceRegime) -> Path:
    """Diagnostic C: scripted policies and reward ordering per regime."""
    out_path = BASE_OUT / f"scripted_policies_{regime.name.lower()}.csv"

    seeds = list(range(20))
    policies = [
        "aggressive_zero_stop",
        "one_stop_compliant",
        "conservative_two_stop",
    ]

    summaries: List[EpisodeSummary] = []

    for policy_name in policies:
        for seed in seeds:
            env = F1RaceEnv(regime=regime, seed=seed)
            summary = run_scripted_policy(env, policy_name, seed)
            summaries.append(summary)
            env.close()

    df = pd.DataFrame([asdict(s) for s in summaries])
    df.to_csv(out_path, index=False)
    print(f"[C] Scripted policy summaries written to {out_path}")

    # Reward ordering checks
    rule = regime == RaceRegime.RULEBOOK
    safe = regime == RaceRegime.SAFE

    if rule:
        zero = df[df["policy_name"] == "aggressive_zero_stop"]["episode_return"].mean()
        comp = df[df["policy_name"] == "one_stop_compliant"]["episode_return"].mean()
        print(
            f"[C] RULEBOOK: mean return compliant one-stop={comp:.3f}, zero-stop={zero:.3f}"
        )
    if safe:
        zero = df[df["policy_name"] == "aggressive_zero_stop"]["episode_return"].mean()
        cons = df[df["policy_name"] == "conservative_two_stop"]["episode_return"].mean()
        print(
            f"[C] SAFE: mean return conservative={cons:.3f}, aggressive={zero:.3f}"
        )

    return out_path


@dataclass
class PPOActionRecord:
    regime: str
    deterministic: bool
    seed: int
    episode_index: int
    step_index: int
    pit_decision: float
    tyre_choice: float
    risk_level: float
    crash: bool
    terminated: bool
    truncated: bool
    lap: int
    segment_id: int | None


def load_ppo_model(run_name: str, regime: RaceRegime, steps: int, seed: int) -> PPO:
    model_dir = Path("outputs") / "experiments" / "models" / run_name / "ppo" / regime.name.lower()
    model_path = model_dir / (
        f"ppo_regime={regime.name.lower()}_seed={seed}_steps={steps}.zip"
    )
    return PPO.load(str(model_path))


def ppo_action_distribution_audit(regime: RaceRegime, steps: int, run_name: str) -> Path:
    """Diagnostic D: audit PPO actions under deterministic and stochastic modes."""
    out_path = BASE_OUT / f"ppo_actions_{regime.name.lower()}.csv"

    seeds = [0]
    det_flags = [True, False]

    records: List[PPOActionRecord] = []

    for det in det_flags:
        for seed in seeds:
            model = load_ppo_model(run_name, regime, steps, seed)
            env = F1RaceEnv(regime=regime, seed=seed)

            for ep in range(20):
                obs, _ = env.reset(seed=seed + ep)
                done = False
                step_idx = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=det)
                    pit_decision = float(action[0])
                    tyre_choice = float(action[1])
                    risk_level = float(action[2])
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    records.append(
                        PPOActionRecord(
                            regime=regime.name.lower(),
                            deterministic=det,
                            seed=seed,
                            episode_index=ep,
                            step_index=step_idx,
                            pit_decision=pit_decision,
                            tyre_choice=tyre_choice,
                            risk_level=risk_level,
                            crash=bool(info.get("crash", False)),
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            lap=int(env.current_lap),
                            segment_id=info.get("segment_id"),
                        )
                    )

                    step_idx += 1

            env.close()

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(out_path, index=False)
    print(f"[D] PPO action audit written to {out_path}")
    return out_path


def main() -> None:
    regimes = [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]

    for regime in regimes:
        forced_action_transition_matrix(regime)
        controlled_hazard_sensitivity(regime)
        scripted_policy_reward_ordering(regime)
        ppo_action_distribution_audit(regime, steps=200000, run_name="default")


if __name__ == "__main__":
    main()

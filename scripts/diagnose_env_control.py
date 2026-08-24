"""Diagnostic script to audit environment control, crash hazard sensitivity,
reward ordering, and PPO action distributions in F1RaceEnv.

This script is read-only with respect to training/evaluation APIs and
models. It runs five diagnostics:

A. Forced action-transition matrix
B. Controlled crash-hazard sensitivity
C. Valid lap-boundary pit control test
D. Scripted-policy reward ordering
E. PPO action-distribution audit

Outputs human-readable console logs and CSV files under outputs/debug/diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.f1_rl_safety.f1_env import F1RaceEnv, RaceRegime


BASE_OUT = Path("outputs") / "debug" / "diagnostics"
BASE_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def format_used_compounds(env: F1RaceEnv) -> str:
    compounds = getattr(env, "used_compounds", set())
    idx_to_compound = getattr(env, "idx_to_compound", {})
    names = [str(idx_to_compound.get(idx, idx)) for idx in sorted(compounds)]
    return ";".join(names)


# ---------------------------------------------------------------------------
# Diagnostic A: Forced action-transition matrix
# ---------------------------------------------------------------------------


@dataclass
class TransitionRecord:
    regime: str
    action_index: int
    pit_decision: float
    tyre_choice: float
    risk_level: float
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


def forced_action_transition_matrix(regime: RaceRegime) -> Path:
    """Diagnostic A: force a grid of actions and log state deltas.

    Uses the underlying continuous F1RaceEnv action space:
    [pit_decision (0/1), tyre_choice (0-4), risk_level (-1..1)].
    Output filename: forced_transitions_<regime>.csv
    """
    out_path = BASE_OUT / f"forced_transitions_{regime.name.lower()}.csv"

    records: List[TransitionRecord] = []
    env = F1RaceEnv(regime=regime, seed=0)

    pit_options = [0.0, 1.0]
    tyre_options = [0.0, 1.0, 2.0, 3.0, 4.0]
    risk_levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

    action_index = 0
    for pit in pit_options:
        for tyre in tyre_options:
            for risk in risk_levels:
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
                        regime=regime.name.lower(),
                        action_index=action_index,
                        pit_decision=float(pit),
                        tyre_choice=float(tyre),
                        risk_level=float(risk),
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


# ---------------------------------------------------------------------------
# Diagnostic B: Controlled crash-hazard sensitivity
# ---------------------------------------------------------------------------


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
    Output filename: hazard_sensitivity_<regime>.csv
    """
    out_path = BASE_OUT / f"hazard_sensitivity_{regime.name.lower()}.csv"

    env = F1RaceEnv(regime=regime, seed=0)
    segments = env.track_segments

    corner = next(s for s in segments if str(s.segment_type).lower() == "corner")
    straight = next(s for s in segments if str(s.segment_type).lower() == "straight")

    scenarios: List[Tuple[str, Any, int, int, float, float]] = []

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


# ---------------------------------------------------------------------------
# Diagnostic C: Valid lap-boundary pit control test
# ---------------------------------------------------------------------------


@dataclass
class PitBoundaryRecord:
    regime: str
    pre_lap: int
    post_lap: int
    pre_segment_idx: int
    post_segment_idx: int
    completed_lap_flag: bool
    pit_eligibility_condition: str
    requested_pit_decision: float
    requested_compound: float
    effective_pit_decision: float
    effective_compound: float
    pre_pit_count: int
    post_pit_count: int
    pre_tyre_compound: int
    post_tyre_compound: int
    pre_tyre_age: int
    post_tyre_age: int
    pre_tyre_wear: float
    post_tyre_wear: float
    pre_used_compounds: str
    post_used_compounds: str
    reward_time_total: float | None
    reward_lap_completion_total: float | None
    reward_risk_total: float | None
    reward_crash_total: float | None
    reward_pit_total: float | None
    reward_compound_total: float | None
    reward_compliance_total: float | None
    episode_return: float | None
    terminated: bool
    truncated: bool
    crash: bool
    crash_reason: str | None
    pit_pass: bool
    pit_test_status: str


def valid_pit_boundary_test(regime: RaceRegime) -> Path:
    """Diagnostic C: ensure pit control works at lap boundary.

    Progress deterministically with low risk and no pit to just before
    the end of lap 0, then issue a pit action at the next step with a
    different compound.

    Output filename: valid_pit_boundary_test_<regime>.csv
    """
    out_path = BASE_OUT / f"valid_pit_boundary_test_{regime.name.lower()}.csv"

    env = F1RaceEnv(regime=regime, seed=0)
    obs, _ = env.reset(seed=0)

    reached_boundary = False

    # Progress with low risk, no pit until just before lap completion.
    while True:
        if env.current_lap > 0:
            # terminated before reaching lap 0 boundary
            break
        # stop just before we would complete lap 0
        if env.current_lap == 0 and env.current_segment_idx == env.n_segments - 1:
            reached_boundary = True
            break
        action = np.array([0.0, float(env.tyre_compound), -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    pre_lap = env.current_lap
    pre_seg = env.current_segment_idx
    pre_pit = env.pit_count
    pre_compound = env.tyre_compound
    pre_age = env.tyre_age
    pre_wear = env.tyre_wear
    pre_used = format_used_compounds(env)

    requested_pit_decision = 1.0
    requested_compound = 2.0

    pit_test_status = ""

    if not reached_boundary:
        # Failed to reach a state where the next step would complete lap 0
        pit_pass = False
        crash_reason = None
        completed_lap_flag = False
        effective_pit_decision = requested_pit_decision
        effective_compound = requested_compound
        post_lap = env.current_lap
        post_seg = env.current_segment_idx
        post_pit = env.pit_count
        post_compound = env.tyre_compound
        post_age = env.tyre_age
        post_wear = env.tyre_wear
        post_used = format_used_compounds(env)
        terminated_flag = False
        truncated_flag = False
        crash_flag = False
        pit_test_status = "FAILED_TO_REACH_BOUNDARY"
    else:
        # Next step should complete lap 0; issue pit with different compound (HARD=2)
        pit_action = np.array([requested_pit_decision, requested_compound, -0.5], dtype=np.float32)
        obs, reward, terminated_flag, truncated_flag, info = env.step(pit_action)

        post_lap = env.current_lap
        post_seg = env.current_segment_idx
        post_pit = env.pit_count
        post_compound = env.tyre_compound
        post_age = env.tyre_age
        post_wear = env.tyre_wear
        post_used = format_used_compounds(env)

        # Environment's pit condition is: lap completion + pit_decision
        completed_lap_flag = bool(post_seg == 0 and post_lap == 1)
        effective_pit_decision = float(pit_action[0])
        effective_compound = float(np.clip(round(pit_action[1]), 0, 4))

        crash_flag = bool(info.get("crash", False))
        crash_reason = info.get("crash_reason")

        pit_pass = (post_pit == pre_pit + 1) and (post_compound != pre_compound)

        if completed_lap_flag and not pit_pass:
            pit_test_status = "VALID_REQUEST_NOT_APPLIED"
        elif completed_lap_flag and pit_pass:
            pit_test_status = "PASS"
        elif not completed_lap_flag:
            pit_test_status = "FAILED_TO_REACH_BOUNDARY"
        else:
            pit_test_status = "UNKNOWN"

    record = PitBoundaryRecord(
        regime=regime.name.lower(),
        pre_lap=pre_lap,
        post_lap=post_lap,
        pre_segment_idx=pre_seg,
        post_segment_idx=post_seg,
        completed_lap_flag=completed_lap_flag,
        pit_eligibility_condition="lap_completed_and_pit_decision",  # matches F1RaceEnv.step semantics
        requested_pit_decision=requested_pit_decision,
        requested_compound=requested_compound,
        effective_pit_decision=effective_pit_decision,
        effective_compound=effective_compound,
        pre_pit_count=pre_pit,
        post_pit_count=post_pit,
        pre_tyre_compound=pre_compound,
        post_tyre_compound=post_compound,
        pre_tyre_age=pre_age,
        post_tyre_age=post_age,
        pre_tyre_wear=float(pre_wear),
        post_tyre_wear=float(post_wear),
        pre_used_compounds=pre_used,
        post_used_compounds=post_used,
        reward_time_total=getattr(env, "reward_time_total", None),
        reward_lap_completion_total=getattr(env, "reward_lap_completion_total", None),
        reward_risk_total=getattr(env, "reward_risk_total", None),
        reward_crash_total=getattr(env, "reward_crash_total", None),
        reward_pit_total=getattr(env, "reward_pit_total", None),
        reward_compound_total=getattr(env, "reward_compound_total", None),
        reward_compliance_total=getattr(env, "reward_compliance_total", None),
        episode_return=getattr(env, "episode_return", None),
        terminated=bool(terminated_flag),
        truncated=bool(truncated_flag),
        crash=bool(crash_flag),
        crash_reason=crash_reason,
        pit_pass=pit_pass,
        pit_test_status=pit_test_status,
    )

    env.close()

    df = pd.DataFrame([asdict(record)])
    df.to_csv(out_path, index=False)
    print(
        f"[C] Valid pit boundary test ({regime.name.lower()}): {pit_test_status} -> {out_path}"
    )
    return out_path


# ---------------------------------------------------------------------------
# Diagnostic D: Scripted-policy reward ordering
# ---------------------------------------------------------------------------


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
    completed_laps: int


def scripted_action(env: F1RaceEnv, policy_name: str) -> np.ndarray:
    """Determine action at current state for a scripted policy.

    Policies:
    - aggressive_zero_stop: pit=0, risk=0.9
    - one_stop_compliant: pit at first valid boundary after lap>=1, MEDIUM->HARD, risk=-0.5 until then
    - conservative_two_stop: pits at first valid boundaries after lap>=1 and lap>=3, alternating compounds, risk=-0.8
    """
    lap = env.current_lap
    seg_idx = env.current_segment_idx
    n_segments = env.n_segments

    pit_decision = 0.0
    tyre_choice = float(env.tyre_compound)
    risk_level = 0.0

    if policy_name == "aggressive_zero_stop":
        pit_decision = 0.0
        tyre_choice = float(env.tyre_compound)
        risk_level = 0.9

    elif policy_name == "one_stop_compliant":
        if env.pit_count == 0 and lap >= 1 and seg_idx == n_segments - 1:
            pit_decision = 1.0
            tyre_choice = 2.0  # HARD
        else:
            pit_decision = 0.0
            tyre_choice = float(env.tyre_compound)
        risk_level = -0.5

    elif policy_name == "conservative_two_stop":
        if env.pit_count == 0 and lap >= 1 and seg_idx == n_segments - 1:
            pit_decision = 1.0
            tyre_choice = 2.0 if env.tyre_compound == 1 else 1.0
        elif env.pit_count == 1 and lap >= 3 and seg_idx == n_segments - 1:
            pit_decision = 1.0
            tyre_choice = 1.0 if env.tyre_compound == 2 else 2.0
        else:
            pit_decision = 0.0
            tyre_choice = float(env.tyre_compound)
        risk_level = -0.8

    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    return np.array([pit_decision, tyre_choice, risk_level], dtype=np.float32)


def run_scripted_policy(env: F1RaceEnv, policy_name: str, seed: int) -> EpisodeSummary:
    obs, _ = env.reset(seed=seed)
    done = False
    ep_risk: List[float] = []
    crash_lap = None
    crash_segment_id = None

    while not done:
        action = scripted_action(env, policy_name)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_risk.append(float(info["risk_level"]))

        if crash_lap is None and info.get("crash", False):
            crash_lap = env.current_lap
            crash_segment_id = info.get("segment_id")

    used = format_used_compounds(env)
    summary = EpisodeSummary(
        regime=env.regime.name.lower(),
        policy_name=policy_name,
        seed=seed,
        episode_index=0,
        crashed=bool(env.crashed),
        crash_lap=crash_lap,
        crash_segment_id=crash_segment_id,
        pitstops=int(env.pit_count),
        used_compounds=used,
        race_time=float(env.race_time),
        finish_position=int(env.position),
        reward_time_total=getattr(env, "reward_time_total", None),
        reward_risk_total=getattr(env, "reward_risk_total", None),
        reward_crash_total=getattr(env, "reward_crash_total", None),
        reward_pit_total=getattr(env, "reward_pit_total", None),
        reward_compound_total=getattr(env, "reward_compound_total", None),
        reward_compliance_total=getattr(env, "reward_compliance_total", None),
        reward_lap_completion_total=getattr(env, "reward_lap_completion_total", None),
        episode_return=getattr(env, "episode_return", None),
        mean_risk=float(np.mean(ep_risk)) if ep_risk else 0.0,
        completed_laps=int(env.current_lap),
    )
    return summary


def scripted_policy_reward_ordering(regime: RaceRegime) -> Path:
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
    print(f"[D] Scripted policy summaries written to {out_path}")
    return out_path


def aggregate_scripted_policy_results() -> Path:
    summary_path = BASE_OUT / "scripted_policy_summary_all_regimes.csv"

    all_frames: List[pd.DataFrame] = []
    for regime in [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]:
        path = BASE_OUT / f"scripted_policies_{regime.name.lower()}.csv"
        if path.exists():
            df = pd.read_csv(path)
            all_frames.append(df)

    if not all_frames:
        print("[D] No scripted policy CSVs found for aggregation.")
        return summary_path

    full = pd.concat(all_frames, ignore_index=True)

    def pct_two_compounds(g):
        return np.mean(g["used_compounds"].apply(lambda s: len(str(s).split(";")) >= 2))

    def finish_rate(g):
        return np.mean(~g["crashed"])

    def mean_crash_lap(g):
        vals = g["crash_lap"].dropna()
        return float(vals.mean()) if len(vals) > 0 else np.nan

    agg = full.groupby(["regime", "policy_name"]).agg(
        mean_episode_return=("episode_return", "mean"),
        std_episode_return=("episode_return", "std"),
        mean_reward_time_total=("reward_time_total", "mean"),
        mean_reward_lap_completion_total=("reward_lap_completion_total", "mean"),
        mean_reward_risk_total=("reward_risk_total", "mean"),
        mean_reward_crash_total=("reward_crash_total", "mean"),
        mean_reward_pit_total=("reward_pit_total", "mean"),
        mean_reward_compound_total=("reward_compound_total", "mean"),
        mean_reward_compliance_total=("reward_compliance_total", "mean"),
        finish_rate=("crashed", lambda x: np.mean(~x)),
        mean_completed_laps=("completed_laps", "mean"),
        mean_pitstops=("pitstops", "mean"),
        pct_two_or_more_compounds=("used_compounds", pct_two_compounds),
        mean_crash_lap=("crash_lap", mean_crash_lap),
        mean_actual_risk=("mean_risk", "mean"),
        conditional_mean_race_time=("race_time", "mean"),
    ).reset_index()

    agg.to_csv(summary_path, index=False)
    print(f"[D] Scripted policy aggregate summary written to {summary_path}")

    def check_rulebook(df: pd.DataFrame) -> None:
        rule_df = df[df["regime"] == "rulebook"]
        zs = rule_df[rule_df["policy_name"] == "aggressive_zero_stop"][
            "mean_episode_return"
        ].values
        os = rule_df[rule_df["policy_name"] == "one_stop_compliant"][
            "mean_episode_return"
        ].values
        if len(zs) and len(os):
            cond = os[0] > zs[0]
            status = "PASS" if cond else "FAIL"
            print(
                f"[D] RULEBOOK hypothesis (one_stop_compliant mean return > aggressive_zero_stop): {status}"
            )
        else:
            print("[D] RULEBOOK hypothesis: insufficient data for check.")

    def check_safe(df: pd.DataFrame) -> None:
        safe_df = df[df["regime"] == "safe"]
        zs = safe_df[safe_df["policy_name"] == "aggressive_zero_stop"][
            "mean_episode_return"
        ].values
        cs = safe_df[safe_df["policy_name"] == "conservative_two_stop"][
            "mean_episode_return"
        ].values
        if len(zs) and len(cs):
            cond = cs[0] > zs[0]
            status = "PASS" if cond else "FAIL"
            print(
                f"[D] SAFE hypothesis (conservative_two_stop mean return > aggressive_zero_stop): {status}"
            )
        else:
            print("[D] SAFE hypothesis: insufficient data for check.")

    def check_unconstrained(df: pd.DataFrame) -> None:
        u_df = df[df["regime"] == "unconstrained"]
        zs = u_df[u_df["policy_name"] == "aggressive_zero_stop"][
            "mean_actual_risk"
        ].values
        cs = u_df[u_df["policy_name"] == "conservative_two_stop"][
            "mean_actual_risk"
        ].values
        zt = u_df[u_df["policy_name"] == "aggressive_zero_stop"][
            "conditional_mean_race_time"
        ].values
        ct = u_df[u_df["policy_name"] == "conservative_two_stop"][
            "conditional_mean_race_time"
        ].values
        if len(zs) and len(cs) and len(zt) and len(ct):
            cond_risk = zs[0] > cs[0]
            cond_time = zt[0] < ct[0]
            status_risk = "PASS" if cond_risk else "FAIL"
            status_time = "PASS" if cond_time else "FAIL"
            print(
                f"[D] UNCONSTRAINED hypothesis (aggressive higher actual risk): {status_risk}"
            )
            print(
                f"[D] UNCONSTRAINED hypothesis (aggressive faster conditional race time): {status_time}"
            )
        else:
            print("[D] UNCONSTRAINED hypothesis: insufficient data for check.")

    check_rulebook(agg)
    check_safe(agg)
    check_unconstrained(agg)

    return summary_path


# ---------------------------------------------------------------------------
# Diagnostic E: PPO action-distribution audit
# ---------------------------------------------------------------------------


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
    model_load_status: str


def load_ppo_model(regime: RaceRegime) -> Tuple[PPO | None, str]:
    """Load the 200000-step seed-0 PPO model for the given regime.

    Uses models/ppo_<regime>.zip as committed in the repository.
    Returns (model, status) where status is "OK" or an error description.
    """
    model_path = Path("models") / f"ppo_{regime.name.lower()}.zip"
    if not model_path.exists():
        return None, f"MODEL_NOT_FOUND:{model_path}"
    try:
        model = PPO.load(str(model_path))
        return model, "OK"
    except Exception as e:
        return None, f"MODEL_LOAD_ERROR:{e}"


def ppo_action_distribution_audit(regime: RaceRegime) -> Path:
    """Diagnostic E: audit PPO actions under deterministic and stochastic modes.

    Output filename: ppo_actions_<regime>.csv
    Failures in model loading are logged per row and do not stop other diagnostics.
    """
    out_path = BASE_OUT / f"ppo_actions_{regime.name.lower()}.csv"

    seeds = [0]
    det_flags = [True, False]

    records: List[PPOActionRecord] = []

    model, status = load_ppo_model(regime)

    if model is None:
        # Log a single row indicating failure
        records.append(
            PPOActionRecord(
                regime=regime.name.lower(),
                deterministic=False,
                seed=0,
                episode_index=0,
                step_index=0,
                pit_decision=np.nan,
                tyre_choice=np.nan,
                risk_level=np.nan,
                crash=False,
                terminated=False,
                truncated=False,
                lap=0,
                segment_id=None,
                model_load_status=status,
            )
        )
    else:
        for det in det_flags:
            for seed in seeds:
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
                                model_load_status=status,
                            )
                        )

                        step_idx += 1

                env.close()

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_csv(out_path, index=False)
    print(f"[E] PPO action audit written to {out_path} (model status: {status})")
    return out_path


# ---------------------------------------------------------------------------
# Main entrypoint to run all diagnostics
# ---------------------------------------------------------------------------


def main() -> None:
    regimes = [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]

    for regime in regimes:
        forced_action_transition_matrix(regime)
        controlled_hazard_sensitivity(regime)
        valid_pit_boundary_test(regime)
        scripted_policy_reward_ordering(regime)
        ppo_action_distribution_audit(regime)

    aggregate_scripted_policy_results()


if __name__ == "__main__":
    main()

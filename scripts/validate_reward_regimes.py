"""Deterministic scripted-policy validation of the three reward regimes.

Runs fixed, hand-scripted policies through F1RaceEnv under each reward
regime (coefficients from configs/configs_silverstone.yaml, the single
source of truth) and checks the qualitative orderings required by the
regime hypotheses:

  H2 RULEBOOK:      compliant_one_stop > both zero-stop policies
  H1 SAFE:          conservative_one_stop > aggressive_zero_stop
  H3 UNCONSTRAINED: compliant_one_stop > conservative_one_stop (pit-matched
                    pace contrast), compliant_one_stop > aggressive_zero_stop
                    (no-early-crash: pit-managed pace-viable survival beats
                    envelope-exceeding early crashing; a zero-stop policy is
                    not a valid survival comparator under recalibrated tyre
                    degradation), compliant_one_stop > over_pitting, and
                    aggressive_zero_stop is not the top-ranked policy

Outputs (per-episode CSV, summary CSV, checks CSV) are written under
outputs/phase2-recalibration/reward_validation/ and never overwrite
historical artefacts. Exit code is non-zero if any check fails.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from f1_rl_safety.f1_env import F1RaceEnv, RaceRegime  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "phase2-recalibration" / "reward_validation"

REGIMES = {
    "unconstrained": RaceRegime.UNCONSTRAINED,
    "rulebook": RaceRegime.RULEBOOK,
    "safe": RaceRegime.SAFE,
}

SOFT, MEDIUM, HARD = 0, 1, 2


def _action(pit: int, tyre: int, risk: float) -> np.ndarray:
    return np.array([float(pit), float(tyre), float(risk)], dtype=np.float32)


def policy_action(name: str, env: F1RaceEnv) -> np.ndarray:
    """Deterministic scripted policies, expressible in the current action
    space. Pits are signalled only on the lap-final segment, where they
    actually take effect in the environment."""
    on_final_segment = env.current_segment_idx == env.n_segments - 1
    lap = env.current_lap

    if name == "aggressive_zero_stop":
        return _action(0, MEDIUM, 0.8)

    if name == "fast_zero_stop":
        # fast but at the corner risk-envelope edge, never beyond it
        return _action(0, MEDIUM, 0.3)

    if name == "compliant_one_stop":
        if on_final_segment and lap == 0 and env.pit_count == 0:
            return _action(1, HARD, 0.2)  # pit at end of lap 1: MEDIUM -> HARD
        return _action(0, MEDIUM, 0.2)

    if name == "conservative_one_stop":
        if on_final_segment and lap == 1 and env.pit_count == 0:
            return _action(1, HARD, -0.5)  # pit at end of lap 2: MEDIUM -> HARD
        return _action(0, MEDIUM, -0.5)

    if name == "over_pitting":
        if on_final_segment:
            tyre = SOFT if env.pit_count % 2 == 0 else HARD
            return _action(1, tyre, 0.0)
        return _action(0, MEDIUM, 0.0)

    raise ValueError(f"Unknown policy: {name}")


POLICIES = [
    "aggressive_zero_stop",
    "fast_zero_stop",
    "compliant_one_stop",
    "conservative_one_stop",
    "over_pitting",
]


def run_episode(env: F1RaceEnv, policy: str, seed: int) -> dict:
    env.reset(seed=seed)
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(policy_action(policy, env))
        done = terminated or truncated
        steps += 1
    return {
        "episode_seed": seed,
        "steps": steps,
        "completed_laps": env.current_lap,
        "crashed": int(env.crashed),
        "catastrophic": int(env.catastrophic_event),
        "pit_count": env.pit_count,
        "n_compounds_used": len(env.used_compounds),
        "race_time": float(env.race_time),
        "episode_return": float(env.episode_return),
        "reward_time_total": float(env.reward_time_total),
        "reward_lap_completion_total": float(env.reward_lap_completion_total),
        "reward_risk_total": float(env.reward_risk_total),
        "reward_crash_total": float(env.reward_crash_total),
        "reward_pit_total": float(env.reward_pit_total),
        "reward_compound_total": float(env.reward_compound_total),
        "reward_compliance_total": float(env.reward_compliance_total),
    }


def build_checks(summary: pd.DataFrame) -> pd.DataFrame:
    def mean_return(regime: str, policy: str) -> float:
        row = summary[(summary.regime == regime) & (summary.policy == policy)]
        return float(row["mean_return"].iloc[0])

    def top_policy(regime: str) -> str:
        block = summary[summary.regime == regime]
        return str(block.loc[block["mean_return"].idxmax(), "policy"])

    checks = [
        {
            "regime": "rulebook",
            "check": "compliant_one_stop > aggressive_zero_stop",
            "lhs": mean_return("rulebook", "compliant_one_stop"),
            "rhs": mean_return("rulebook", "aggressive_zero_stop"),
        },
        {
            "regime": "rulebook",
            "check": "compliant_one_stop > fast_zero_stop",
            "lhs": mean_return("rulebook", "compliant_one_stop"),
            "rhs": mean_return("rulebook", "fast_zero_stop"),
        },
        {
            "regime": "safe",
            "check": "conservative_one_stop > aggressive_zero_stop",
            "lhs": mean_return("safe", "conservative_one_stop"),
            "rhs": mean_return("safe", "aggressive_zero_stop"),
        },
        {
            # pit-matched pace contrast: both policies one-stop, so the
            # known free-pit environment defect does not confound the
            # fast-vs-deliberately-slow comparison
            "regime": "unconstrained",
            "check": "compliant_one_stop > conservative_one_stop",
            "lhs": mean_return("unconstrained", "compliant_one_stop"),
            "rhs": mean_return("unconstrained", "conservative_one_stop"),
        },
        {
            # No-early-crash check. Under recalibrated tyre degradation and
            # correctly priced pit loss, the no-early-crash condition is
            # tested using a pit-managed, pace-viable policy versus an
            # envelope-exceeding early-crash policy. A zero-stop policy is
            # no longer a valid survival comparator because its long-stint
            # tyre degradation is itself strategically pathological.
            "regime": "unconstrained",
            "check": "compliant_one_stop > aggressive_zero_stop",
            "lhs": mean_return("unconstrained", "compliant_one_stop"),
            "rhs": mean_return("unconstrained", "aggressive_zero_stop"),
        },
        {
            # over-pitting must remain strongly disfavoured now that every
            # effective pit pays the full pit-lane time loss
            "regime": "unconstrained",
            "check": "compliant_one_stop > over_pitting",
            "lhs": mean_return("unconstrained", "compliant_one_stop"),
            "rhs": mean_return("unconstrained", "over_pitting"),
        },
    ]
    for c in checks:
        c["passed"] = bool(c["lhs"] > c["rhs"])

    top_unconstrained = top_policy("unconstrained")
    checks.append(
        {
            "regime": "unconstrained",
            "check": "aggressive_zero_stop is not top-ranked",
            "lhs": float("nan"),
            "rhs": float("nan"),
            "passed": top_unconstrained != "aggressive_zero_stop",
        }
    )
    return pd.DataFrame(checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for regime_name, regime in REGIMES.items():
        env = F1RaceEnv(regime=regime, seed=args.seed)
        for policy in POLICIES:
            for ep in range(args.episodes):
                # identical episode seeds across policies and regimes so
                # every policy faces the same hazard realisations
                row = run_episode(env, policy, seed=args.seed + ep)
                row.update({"regime": regime_name, "policy": policy})
                rows.append(row)
        env.close()

    episodes_df = pd.DataFrame(rows)

    summary = (
        episodes_df.groupby(["regime", "policy"], sort=False)
        .agg(
            mean_return=("episode_return", "mean"),
            std_return=("episode_return", "std"),
            mean_laps=("completed_laps", "mean"),
            crash_rate=("crashed", "mean"),
            catastrophic_rate=("catastrophic", "mean"),
            mean_pits=("pit_count", "mean"),
            mean_time=("reward_time_total", "mean"),
            mean_lap_comp=("reward_lap_completion_total", "mean"),
            mean_risk_comp=("reward_risk_total", "mean"),
            mean_crash_comp=("reward_crash_total", "mean"),
            mean_pit_comp=("reward_pit_total", "mean"),
            mean_compound_comp=("reward_compound_total", "mean"),
            mean_compliance_comp=("reward_compliance_total", "mean"),
        )
        .reset_index()
    )

    checks = build_checks(summary)

    stem = f"scripted_policy_seed{args.seed}_n{args.episodes}_{run_id}"
    episodes_path = out_dir / f"{stem}_episodes.csv"
    summary_path = out_dir / f"{stem}_summary.csv"
    checks_path = out_dir / f"{stem}_checks.csv"
    episodes_df.to_csv(episodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    checks.to_csv(checks_path, index=False)

    pd.set_option("display.width", 200)
    print("\n=== Scripted-policy summary (mean over episodes) ===")
    print(
        summary[
            [
                "regime",
                "policy",
                "mean_return",
                "std_return",
                "mean_laps",
                "mean_pits",
                "catastrophic_rate",
            ]
        ].to_string(index=False)
    )
    print("\n=== Qualitative ordering checks ===")
    print(checks.to_string(index=False))
    print(f"\nOutputs written to: {episodes_path}, {summary_path}, {checks_path}")

    all_passed = bool(checks["passed"].all())
    print(f"\nOVERALL: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

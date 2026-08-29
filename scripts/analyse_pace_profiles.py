"""Pace diagnostics: local per-lap pace versus race-level outcome.

Read-only analysis. This script does not train, and does not modify the
environment, rewards, wrappers, evaluator or any existing artefact. It
replays policies through the unchanged F1RaceEnv and records the value the
environment already emits in ``info["lap_time"]``.

TIMING DEFINITION (verified against src/f1_rl_safety/f1_env.py)
---------------------------------------------------------------
``info["lap_time"]`` is a MISNOMER in the environment: it is the *segment*
time returned by ``_simulate_segment``, expressed in seconds, emitted once
per ``step()`` call, i.e. once per track segment (1/n_segments of a lap;
n_segments = 36 for the committed Silverstone segmentation).

Its components are

    seg_time = base_lap/N + compound_offset/N + (deg * tyre_age)/N
             + (2.5 * fuel_level)/N + (-1.8 * max(0, risk))/N
             + (0.8 * max(0, -risk))/N + pit_loss + noise

where N = n_segments, ``pit_loss`` is the FULL 21.5 s pit-lane loss charged
on the single step where a pit takes effect, and noise ~ N(0, 0.2) per
segment. Because every term except ``pit_loss`` and ``noise`` is already
divided by N, summing the N segment times belonging to one lap reconstructs
a physically coherent lap time:

    lap_time = base_lap + compound_offset + deg*tyre_age + 2.5*fuel_level
             + risk terms + (pit_loss if a pit executed that lap)
             + sum of N noise draws  [~ N(0, 0.2*sqrt(N)) = N(0, 1.2) s]

AGGREGATION METHOD
------------------
A lap record is the sum of segment times over the segments belonging to one
lap index. Lap membership is determined by reading ``env.current_lap``
*before* each ``env.step()`` call, because the environment increments the
lap counter before ``_simulate_segment`` is invoked on the lap-final step.
A lap is treated as COMPLETE only when exactly n_segments segments were
recorded for it; the truncated final lap of a crashed episode is therefore
excluded from pace statistics but retained in progress/attrition statistics.

``tyre_compound``, ``tyre_age``, ``tyre_wear`` and ``fuel_level`` are
recorded on entry to the lap's first segment. These quantities update only
at lap completion, so they are constant across a lap and are well-defined
per-lap attributes.

FUEL CONFOUND
-------------
The fuel term contributes exactly ``2.5 * fuel_level`` seconds per lap and
declines from 2.5 s to 0 s across a race. It is identical for all policies
at the same lap number, so same-lap comparisons are unconfounded by fuel;
comparisons across lap numbers within a policy conflate fuel with tyre age.
A documented derived column ``lap_time_fuel_adj_s = lap_time_s -
2.5 * fuel_level_entry`` removes the fuel term exactly.

RACE_TIME CAVEAT
----------------
``env.race_time`` additionally receives a +300 s classification penalty for
RULEBOOK episodes that complete the race with zero pit stops
(f1_env.py, terminal block). That penalty is applied to state, not to any
segment time, so the per-lap pace metric here is clean of it while
``race_time`` is not. Both are reported, and finisher views are broken down
by pit count so the artefact is visible.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from stable_baselines3 import PPO  # noqa: E402
from f1_rl_safety.f1_env import F1RaceEnv, RaceRegime  # noqa: E402

REGIMES = {
    "unconstrained": RaceRegime.UNCONSTRAINED,
    "rulebook": RaceRegime.RULEBOOK,
    "safe": RaceRegime.SAFE,
}
SEEDS = (0, 1, 2)
GAMMA = 0.999
TRAIN_STEPS = 100_000

CHECKPOINT_ROOT = (
    REPO / "outputs" / "phase2-recalibration"
    / "ppo_gamma_ablation_0999_20260826T162818" / "models" / "gamma0999"
)

# Fixed-policy counterfactual conditions (zero-pit, constant demanded risk).
FIXED_RISKS = (-0.5, 0.0, 0.3, 0.68, 0.8)
FIXED_SEED_BASE = 9000  # disjoint from training (0-2) and evaluation seeds
MEDIUM = 1

# Pre-registered matched window.
MATCHED_LAP_LO, MATCHED_LAP_HI = 5, 15
FUEL_COEF = 2.5  # documented constant from _simulate_segment
RULEBOOK_NONCOMPLIANCE_TIME_PENALTY = 300.0


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def _lap_records(env, action_fn, episode_seed, episode_id, labels):
    """Replay one episode; return per-complete-lap records plus an episode summary.

    action_fn(env, obs) -> np.ndarray action. Nothing in the environment is
    modified; only emitted info and public state attributes are read.
    """
    obs, _ = env.reset(seed=episode_seed)
    n_seg = env.n_segments

    laps = {}          # lap_index -> accumulator
    order = []
    done = False
    steps = 0
    crashed_lap = None

    while not done:
        # lap the segment about to be simulated belongs to (counter increments
        # inside step() before _simulate_segment is called on the wrap step)
        lap_idx = int(env.current_lap)
        seg_idx = int(env.current_segment_idx)

        if lap_idx not in laps:
            laps[lap_idx] = {
                "segments": 0,
                "lap_time_s": 0.0,
                "tyre_compound": env.idx_to_compound.get(
                    env.tyre_compound, str(env.tyre_compound)
                ),
                "tyre_age_entry": int(env.tyre_age),
                "tyre_wear_entry": float(env.tyre_wear),
                "fuel_level_entry": float(env.fuel_level),
                "risk_sum": 0.0,
                "pit_executed": 0,
                "crashed_in_lap": 0,
            }
            order.append(lap_idx)

        action = action_fn(env, obs)
        pit_before = int(env.pit_count)

        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        done = terminated or truncated

        rec = laps[lap_idx]
        rec["segments"] += 1
        rec["lap_time_s"] += float(info["lap_time"])   # segment time, seconds
        rec["risk_sum"] += float(info["risk_level"])
        if int(env.pit_count) > pit_before:
            rec["pit_executed"] = 1
        if bool(info["crash"]):
            rec["crashed_in_lap"] = 1
            crashed_lap = lap_idx

    crashed = bool(env.crashed)
    completed_laps = int(env.current_lap)
    finished = (not crashed) and completed_laps >= env.n_laps
    pit_count_total = int(env.pit_count)

    rows = []
    segsum_total = 0.0
    for lap_idx in order:
        rec = laps[lap_idx]
        segsum_total += rec["lap_time_s"]
        if rec["segments"] != n_seg:
            continue  # truncated lap (crash) -> excluded from pace statistics
        rows.append({
            **labels,
            "episode": episode_id,
            "episode_seed": episode_seed,
            "lap_index": lap_idx,
            "lap_number": lap_idx + 1,          # 1-based for reporting
            "segments_in_lap": rec["segments"],
            "lap_time_s": rec["lap_time_s"],
            "lap_time_fuel_adj_s": rec["lap_time_s"] - FUEL_COEF * rec["fuel_level_entry"],
            "tyre_compound": rec["tyre_compound"],
            "tyre_age_entry": rec["tyre_age_entry"],
            "tyre_wear_entry": rec["tyre_wear_entry"],
            "fuel_level_entry": rec["fuel_level_entry"],
            "mean_risk_in_lap": rec["risk_sum"] / rec["segments"],
            "pit_executed_in_lap": rec["pit_executed"],
            "episode_pit_count": pit_count_total,
            "episode_crashed": int(crashed),
            "episode_finished": int(finished),
            "episode_completed_laps": completed_laps,
        })

    summary = {
        **labels,
        "episode": episode_id,
        "episode_seed": episode_seed,
        "steps": steps,
        "completed_laps": completed_laps,
        "crashed": int(crashed),
        "catastrophic": int(env.catastrophic_event),
        "finished": int(finished),
        "pit_count": pit_count_total,
        "n_compounds_used": len(env.used_compounds),
        "race_time_s": float(env.race_time),
        "segment_sum_time_s": segsum_total,
        "crash_lap": crashed_lap if crashed else None,
        "mean_risk_episode": (
            float(np.mean([r["mean_risk_in_lap"] for r in rows])) if rows else np.nan
        ),
    }
    return rows, summary


def replay_fixed_policies(n_episodes):
    """Zero-pit, constant-risk counterfactuals. Dynamics are regime-independent
    (self.regime affects only reward weights and the RULEBOOK terminal
    race_time adjustment), so a single regime instance suffices for pace."""
    lap_rows, ep_rows = [], []
    for risk in FIXED_RISKS:
        env = F1RaceEnv(regime=RaceRegime.UNCONSTRAINED, seed=FIXED_SEED_BASE)
        action = np.array([0.0, float(MEDIUM), float(risk)], dtype=np.float32)

        def action_fn(_env, _obs, _a=action):
            return _a

        labels = {
            "source": "fixed_policy",
            "policy": f"constant_risk_{risk:+.2f}_zero_pit",
            "risk_condition": risk,
            "regime": "n/a_dynamics_regime_independent",
            "seed": "n/a",
        }
        for ep in range(n_episodes):
            rows, summ = _lap_records(
                env, action_fn, FIXED_SEED_BASE + ep, ep, labels
            )
            lap_rows.extend(rows)
            ep_rows.append(summ)
        env.close()
    return pd.DataFrame(lap_rows), pd.DataFrame(ep_rows)


def replay_learned_policies(n_episodes):
    """Deterministic replay of the nine committed gamma=0.999 checkpoints."""
    lap_rows, ep_rows, used = [], [], []
    for regime_name, regime in REGIMES.items():
        for seed in SEEDS:
            ckpt = (
                CHECKPOINT_ROOT / "ppo" / regime_name
                / f"ppo_regime={regime_name}_seed={seed}_steps={TRAIN_STEPS}.zip"
            )
            if not ckpt.exists():
                raise FileNotFoundError(f"checkpoint missing: {ckpt}")
            model = PPO.load(str(ckpt))
            env = F1RaceEnv(regime=regime, seed=seed)

            def action_fn(_env, obs, _m=model):
                action, _ = _m.predict(obs, deterministic=True)
                return action

            labels = {
                "source": "learned_policy",
                "policy": f"ppo_gamma{GAMMA}_{regime_name}_seed{seed}",
                "risk_condition": np.nan,
                "regime": regime_name,
                "seed": seed,
            }
            for ep in range(n_episodes):
                rows, summ = _lap_records(env, action_fn, seed + ep, ep, labels)
                lap_rows.extend(rows)
                ep_rows.append(summ)
            env.close()
            used.append({
                "regime": regime_name, "seed": seed, "gamma": GAMMA,
                "training_steps": TRAIN_STEPS,
                "checkpoint": str(ckpt.relative_to(REPO)),
                "episodes_replayed": n_episodes,
                "episode_seeds": f"{seed}..{seed + n_episodes - 1}",
            })
    return pd.DataFrame(lap_rows), pd.DataFrame(ep_rows), pd.DataFrame(used)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def _boot_ci(values, n_boot, rng, alpha=0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def _stat_row(view, group_kind, label, regime, seed, risk, bucket, metric,
              values, n_boot, rng):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lo, hi = _boot_ci(v, n_boot, rng)
    return {
        "view": view, "group_kind": group_kind, "group_label": label,
        "regime": regime, "seed": seed, "risk_condition": risk, "bucket": bucket,
        "metric": metric, "n": int(v.size),
        "mean": float(v.mean()) if v.size else np.nan,
        "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
        "ci95_lo": lo, "ci95_hi": hi,
    }


def _groups(lap_df, ep_df):
    """Yield (group_kind, label, regime, seed, risk, lap_subset, ep_subset)."""
    for pol, sub in lap_df[lap_df.source == "fixed_policy"].groupby("policy"):
        risk = float(sub.risk_condition.iloc[0])
        yield ("fixed_policy", pol, "n/a", "n/a", risk, sub,
               ep_df[(ep_df.source == "fixed_policy") & (ep_df.policy == pol)])
    learned = lap_df[lap_df.source == "learned_policy"]
    for regime, sub in learned.groupby("regime"):
        yield ("learned_regime_pooled", f"ppo_gamma{GAMMA}_{regime}_seeds012",
               regime, "pooled", np.nan, sub,
               ep_df[(ep_df.source == "learned_policy") & (ep_df.regime == regime)])
    for (regime, seed), sub in learned.groupby(["regime", "seed"]):
        yield ("learned_model", f"ppo_gamma{GAMMA}_{regime}_seed{seed}",
               regime, seed, np.nan, sub,
               ep_df[(ep_df.source == "learned_policy")
                     & (ep_df.regime == regime) & (ep_df.seed == seed)])


def build_summary(lap_df, ep_df, n_boot, rng):
    out = []
    for kind, label, regime, seed, risk, laps, eps in _groups(lap_df, ep_df):
        # (1) pre-registered matched window: laps 5-15, zero-pit episodes
        mw = laps[(laps.lap_number.between(MATCHED_LAP_LO, MATCHED_LAP_HI))
                  & (laps.episode_pit_count == 0)]
        # lap_time metrics plus balance metrics, so the artefact documents
        # whether the window is matched on tyre age, compound and risk
        for metric in ("lap_time_s", "lap_time_fuel_adj_s", "tyre_age_entry",
                       "mean_risk_in_lap", "is_medium_compound"):
            vals = (mw.tyre_compound.eq("MEDIUM").astype(float)
                    if metric == "is_medium_compound" else mw[metric])
            out.append(_stat_row(
                f"matched_window_laps{MATCHED_LAP_LO}_{MATCHED_LAP_HI}_zeropit",
                kind, label, regime, seed, risk,
                f"laps{MATCHED_LAP_LO}-{MATCHED_LAP_HI}", metric,
                vals, n_boot, rng))

        # (1b) robustness: same window, non-pit laps from any episode
        mw2 = laps[(laps.lap_number.between(MATCHED_LAP_LO, MATCHED_LAP_HI))
                   & (laps.pit_executed_in_lap == 0)]
        out.append(_stat_row(
            f"matched_window_laps{MATCHED_LAP_LO}_{MATCHED_LAP_HI}_nonpitlaps",
            kind, label, regime, seed, risk,
            f"laps{MATCHED_LAP_LO}-{MATCHED_LAP_HI}", "lap_time_s",
            mw2["lap_time_s"], n_boot, rng))

        # (2) local pace by lap number (non-pit laps)
        npl = laps[laps.pit_executed_in_lap == 0]
        for lap_no, g in npl.groupby("lap_number"):
            if len(g) < 3:
                continue
            out.append(_stat_row("by_lap_number", kind, label, regime, seed,
                                 risk, int(lap_no), "lap_time_s",
                                 g["lap_time_s"], n_boot, rng))

        # (3) local pace by tyre age (non-pit laps; fuel-adjusted isolates wear)
        for age, g in npl.groupby("tyre_age_entry"):
            if len(g) < 3:
                continue
            out.append(_stat_row("by_tyre_age", kind, label, regime, seed, risk,
                                 int(age), "lap_time_fuel_adj_s",
                                 g["lap_time_fuel_adj_s"], n_boot, rng))

        # (3b) compound-controlled: MEDIUM only (the common start compound).
        # Required because the calibrated SOFT compound is both faster
        # (offset -1.270 s) and far more durable (deg 0.044 vs 0.202 s per lap
        # of age) than MEDIUM, so any regime that pits onto SOFT would
        # otherwise appear faster at matched tyre age for reasons unrelated to
        # demanded risk.
        med = npl[npl.tyre_compound == "MEDIUM"]
        for age, g in med.groupby("tyre_age_entry"):
            if len(g) < 3:
                continue
            out.append(_stat_row("by_tyre_age_medium_only", kind, label, regime,
                                 seed, risk, int(age), "lap_time_fuel_adj_s",
                                 g["lap_time_fuel_adj_s"], n_boot, rng))

        # (4) conditional completed-race time (finishers only)
        fin = eps[eps.finished == 1]
        for metric in ("race_time_s", "segment_sum_time_s"):
            out.append(_stat_row("race_outcome_conditional_finishers", kind,
                                 label, regime, seed, risk, "finishers_all",
                                 metric, fin[metric], n_boot, rng))
        for pc_label, sub in (("finishers_zero_pit", fin[fin.pit_count == 0]),
                              ("finishers_with_pit", fin[fin.pit_count >= 1])):
            out.append(_stat_row("race_outcome_conditional_finishers", kind,
                                 label, regime, seed, risk, pc_label,
                                 "race_time_s", sub["race_time_s"], n_boot, rng))

        # (5) all-episode progress / survival
        out.append(_stat_row("all_episode_progress", kind, label, regime, seed,
                             risk, "all_episodes", "completed_laps",
                             eps["completed_laps"], n_boot, rng))

        # (6) crash / attrition effects
        for metric, col in (("crash_rate", "crashed"),
                            ("catastrophic_rate", "catastrophic"),
                            ("finish_rate", "finished")):
            out.append(_stat_row("attrition", kind, label, regime, seed, risk,
                                 "all_episodes", metric, eps[col], n_boot, rng))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------

README_TEMPLATE = """# Pace diagnostics — local pace versus race-level outcome

Generated by `scripts/analyse_pace_profiles.py` (read-only analysis; no
training, and no modification of the environment, rewards, wrappers,
evaluator or any existing artefact).

Run: `{run_id}` | fixed-policy episodes per risk: {n_fixed} | learned-policy
episodes per checkpoint: {n_learned} | bootstrap resamples: {n_boot}
(percentile bootstrap of the mean, RNG seed {boot_seed}).

## Timing definition and aggregation

`info["lap_time"]` emitted by `F1RaceEnv.step()` is a misnomer: it is the
**segment** time in **seconds**, produced once per `step()` call, i.e. once
per track segment ({n_segments} segments per lap). Every component except the
pit-lane loss and the per-segment noise is already divided by the segment
count, so **summing the {n_segments} segment times of one lap reconstructs a
coherent lap time** (residual noise on a lap total is ~N(0, 1.2) s).

Lap membership is determined by reading `env.current_lap` **before** each
`step()`, because the environment increments the lap counter before
`_simulate_segment` runs on the lap-final step. A lap enters the pace
statistics only if exactly {n_segments} segments were recorded for it, so the
truncated final lap of a crashed episode is excluded from pace but retained
in progress and attrition statistics. `tyre_compound`, `tyre_age`,
`tyre_wear` and `fuel_level` are recorded on entry to the lap and are
constant within it.

`lap_time_fuel_adj_s = lap_time_s - 2.5 * fuel_level_entry` removes the fuel
term exactly; it is used for tyre-age profiles so that wear is not conflated
with fuel burn. Same-lap-number comparisons between policies are unaffected
by fuel either way.

**`race_time` caveat:** RULEBOOK episodes that finish with zero pit stops
receive a +{penalty:.0f} s classification penalty added to `env.race_time`
(not to any segment time). Per-lap pace here is clean of it; `race_time` is
not. Finisher views are therefore also broken down by pit count, and
`segment_sum_time_s` (the sum of segment times) is reported alongside
`race_time_s`.

## Files

- `manifest.csv` — provenance: checkpoints replayed, episode counts, seeds, timings.
- `fixed_policy_pace_profiles.csv` — per-complete-lap records, zero-pit constant-risk counterfactuals.
- `learned_policy_pace_profiles.csv` — per-complete-lap records, nine deterministic gamma=0.999 checkpoints.
- `matched_window_summary.csv` — all aggregated views (see `view` column).

Per-segment records are aggregated in memory and not persisted, to keep
artefacts small; the aggregation is fully specified above and reproducible.
No PNG figures are produced: the repository has no established plotting
convention for analysis scripts, so tabular output only.

## Views in `matched_window_summary.csv`

| `view` | Meaning |
|---|---|
| `matched_window_laps{lo}_{hi}_zeropit` | Pre-registered primary: laps {lo}-{hi}, episodes with zero pit stops. |
| `matched_window_laps{lo}_{hi}_nonpitlaps` | Robustness: laps {lo}-{hi}, excluding only laps on which a pit executed (retains post-pit stints). |
| `by_lap_number` | Local pace by lap number (non-pit laps), buckets with n>=3. |
| `by_tyre_age` | Fuel-adjusted pace by tyre age (non-pit laps), buckets with n>=3. |
| `by_tyre_age_medium_only` | As above, restricted to the MEDIUM start compound. Required because calibrated SOFT is faster (offset -1.270 s) and far more durable (deg 0.044 vs 0.202 s per lap of age), so a regime that pits onto SOFT appears faster at matched tyre age for reasons unrelated to demanded risk. |
| `race_outcome_conditional_finishers` | Completed-race time, finishers only, split all / zero-pit / with-pit. |
| `all_episode_progress` | Completed laps over all episodes (progress/survival proxy). |
| `attrition` | Crash rate, catastrophic rate, finish rate over all episodes. |

Columns: `group_kind` (`fixed_policy`, `learned_regime_pooled`,
`learned_model`), `group_label`, `regime`, `seed`, `risk_condition`,
`bucket`, `metric`, `n`, `mean`, `sd`, `ci95_lo`, `ci95_hi`.

## Interpretation guardrail

Conditional completed-race time, local per-lap pace, all-episode progress
and attrition are distinct quantities and are reported separately. A
difference in local pace between two learned regimes should be asserted only
where the bootstrap confidence intervals support it.
"""


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-fixed", type=int, default=200)
    p.add_argument("--episodes-learned", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=12345)
    p.add_argument("--label", type=str, default=None,
                   help="Optional output subdirectory label (default: timestamp).")
    args = p.parse_args()

    run_id = args.label or datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = (REPO / "outputs" / "phase2-recalibration" / "pace_diagnostics" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = datetime.now()
    print(f"[{t_start:%H:%M:%S}] fixed-policy replays "
          f"({len(FIXED_RISKS)} risks x {args.episodes_fixed} episodes)...", flush=True)
    fixed_laps, fixed_eps = replay_fixed_policies(args.episodes_fixed)

    t_fixed = datetime.now()
    print(f"[{t_fixed:%H:%M:%S}] learned-policy replays "
          f"(9 checkpoints x {args.episodes_learned} episodes)...", flush=True)
    learned_laps, learned_eps, used = replay_learned_policies(args.episodes_learned)
    t_learned = datetime.now()

    lap_df = pd.concat([fixed_laps, learned_laps], ignore_index=True)
    ep_df = pd.concat([fixed_eps, learned_eps], ignore_index=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    summary = build_summary(lap_df, ep_df, args.bootstrap, rng)
    t_end = datetime.now()

    fixed_laps.to_csv(out_dir / "fixed_policy_pace_profiles.csv", index=False)
    learned_laps.to_csv(out_dir / "learned_policy_pace_profiles.csv", index=False)
    summary.to_csv(out_dir / "matched_window_summary.csv", index=False)

    n_segments = int(F1RaceEnv(regime=RaceRegime.SAFE, seed=0).n_segments)
    manifest = used.copy()
    manifest["run_id"] = run_id
    manifest["replay_kind"] = "learned_policy_deterministic"
    extra = pd.DataFrame([{
        "run_id": run_id, "replay_kind": "fixed_policy_zero_pit",
        "regime": "n/a_dynamics_regime_independent", "seed": "n/a",
        "gamma": np.nan, "training_steps": np.nan,
        "checkpoint": f"risks={list(FIXED_RISKS)}",
        "episodes_replayed": args.episodes_fixed,
        "episode_seeds": f"{FIXED_SEED_BASE}..{FIXED_SEED_BASE + args.episodes_fixed - 1}",
    }])
    manifest = pd.concat([manifest, extra], ignore_index=True)
    manifest["n_segments_per_lap"] = n_segments
    manifest["bootstrap_resamples"] = args.bootstrap
    manifest["bootstrap_seed"] = args.bootstrap_seed
    manifest["script"] = "scripts/analyse_pace_profiles.py"
    manifest["fixed_replay_seconds"] = round((t_fixed - t_start).total_seconds(), 1)
    manifest["learned_replay_seconds"] = round((t_learned - t_fixed).total_seconds(), 1)
    manifest["aggregation_seconds"] = round((t_end - t_learned).total_seconds(), 1)
    manifest["generated_at"] = t_end.isoformat(timespec="seconds")
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    (out_dir / "README.md").write_text(README_TEMPLATE.format(
        run_id=run_id, n_fixed=args.episodes_fixed, n_learned=args.episodes_learned,
        n_boot=args.bootstrap, boot_seed=args.bootstrap_seed,
        n_segments=n_segments, penalty=RULEBOOK_NONCOMPLIANCE_TIME_PENALTY,
        lo=MATCHED_LAP_LO, hi=MATCHED_LAP_HI,
    ))

    print(f"[{t_end:%H:%M:%S}] wrote {out_dir.relative_to(REPO)}: "
          f"{len(fixed_laps)} fixed laps, {len(learned_laps)} learned laps, "
          f"{len(summary)} summary rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Final-v1 evaluation batch driver.

Evaluates the completed final-v1 checkpoint matrix (5 algorithms x 3
regimes x 3 training seeds = 45 checkpoints) directly from their
existing, unmodified on-disk locations:

  - PPO/A2C/DQN:  outputs/experiments/models/final-v1/<algo>/<regime>/
                  <algo>_regime=<regime>_seed=<seed>_steps=200000.zip
  - SARSA:        outputs/experiments/models/sarsa/<regime>/
                  sarsa_regime=<regime>_seed=<seed>_steps=200000.pt
  - REINFORCE:    outputs/experiments/models/reinforce/<regime>/
                  reinforce_regime=<regime>_seed=<seed>_steps=200000.pt

For each algorithm x regime pair, `evaluate_grid` is called once across
the three training seeds, using a shared, training-seed-independent
evaluation-seed schedule (eval_seed_base=10000, one reset seed per
evaluation episode), producing one final CSV:

  outputs/experiments/eval/final-v1/<algo>_<regime>_steps=200000_eval=100.csv

This script only reads checkpoints and writes new CSVs under
outputs/experiments/eval/final-v1/. It never moves, copies, renames,
overwrites, or deletes any checkpoint or training log, and never
trains anything.

Before the full batch, a single smoke check (ppo/rulebook/seed=0,
2 evaluation episodes) is run and written only to
outputs/experiments/eval/final-v1/_smoke/, never merged into the final
CSVs. The full batch only proceeds if the smoke check passes.

Re-running this script is safe: a final CSV that already exists and is
verified complete (right algo/regime, exactly training seeds 0/1/2,
exactly 100 distinct evaluation episodes per seed, exactly evaluation
env seeds 10000..10099 per seed) is left untouched and reported as
skipped-complete. A CSV that exists but fails that check causes the
script to stop with an error rather than overwrite or append to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from f1_rl_safety.f1_env import RaceRegime  # noqa: E402
from f1_rl_safety.evaluate_rl import evaluate_grid, evaluate_model  # noqa: E402

ALGOS = ["ppo", "a2c", "dqn", "sarsa", "reinforce"]
REGIMES = [RaceRegime.UNCONSTRAINED, RaceRegime.RULEBOOK, RaceRegime.SAFE]
SEEDS = [0, 1, 2]
STEPS = 200000
N_EPISODES = 100
EVAL_SEED_BASE = 10000
RUN_NAME = "final-v1"

MODEL_DIR = REPO_ROOT / "outputs" / "experiments" / "models"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "experiments" / "eval" / "final-v1"
SMOKE_DIR = OUTPUT_ROOT / "_smoke"


class IncompleteCsvError(RuntimeError):
    """Raised when a pre-existing final CSV fails the completeness check."""


def final_csv_path(algo: str, regime: RaceRegime) -> Path:
    return OUTPUT_ROOT / f"{algo}_{regime.name.lower()}_steps={STEPS}_eval={N_EPISODES}.csv"


def validate_complete_csv(
    path: Path,
    algo: str,
    regime: RaceRegime,
    seeds: list[int],
    n_episodes: int,
    eval_seed_base: int,
) -> None:
    """Raise IncompleteCsvError if `path` does not satisfy the completeness
    criteria for this algo/regime; otherwise return silently.
    """
    df = pd.read_csv(path)

    required_cols = {"algo", "regime", "seed", "evaluation_episode", "evaluation_env_seed"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise IncompleteCsvError(f"{path}: missing required columns {sorted(missing_cols)}")

    algo_values = set(df["algo"].unique())
    if algo_values != {algo}:
        raise IncompleteCsvError(f"{path}: expected algo={{{algo!r}}}, found {algo_values}")

    regime_values = set(df["regime"].unique())
    expected_regime = {regime.name.lower()}
    if regime_values != expected_regime:
        raise IncompleteCsvError(
            f"{path}: expected regime={expected_regime}, found {regime_values}"
        )

    seed_values = set(int(s) for s in df["seed"].unique())
    if seed_values != set(seeds):
        raise IncompleteCsvError(f"{path}: expected seeds={set(seeds)}, found {seed_values}")

    expected_eval_seeds = set(range(eval_seed_base, eval_seed_base + n_episodes))
    for seed in seeds:
        seed_df = df[df["seed"] == seed]

        episodes = set(int(e) for e in seed_df["evaluation_episode"].unique())
        expected_episodes = set(range(n_episodes))
        if episodes != expected_episodes:
            raise IncompleteCsvError(
                f"{path}: seed={seed} expected {n_episodes} distinct evaluation_episode "
                f"values 0..{n_episodes - 1}, found {len(episodes)} distinct values"
            )

        env_seeds = set(int(s) for s in seed_df["evaluation_env_seed"].unique())
        if env_seeds != expected_eval_seeds:
            raise IncompleteCsvError(
                f"{path}: seed={seed} expected evaluation_env_seed values "
                f"{eval_seed_base}..{eval_seed_base + n_episodes - 1}, found {sorted(env_seeds)}"
            )


def run_smoke_check() -> Path:
    """Run the ppo/rulebook/seed=0/2-episode smoke check.

    Writes to SMOKE_DIR only; never touches the final 15 CSVs. Raises on
    any failure so the caller can stop before evaluating the full batch.
    """
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    smoke_path = SMOKE_DIR / f"ppo_rulebook_seed=0_steps={STEPS}_eval=2.csv"

    print(f"[smoke] evaluating ppo/rulebook/seed=0/steps={STEPS}/episodes=2 ...")
    rows = evaluate_model(
        algo="ppo",
        regime=RaceRegime.RULEBOOK,
        seed=0,
        steps_or_episodes=STEPS,
        n_episodes=2,
        model_dir=MODEL_DIR,
        run_name=RUN_NAME,
        eval_seed_base=EVAL_SEED_BASE,
    )
    df = pd.DataFrame(rows)

    observed_episodes = set(int(e) for e in df["evaluation_episode"].unique())
    observed_env_seeds = set(int(s) for s in df["evaluation_env_seed"].unique())
    if observed_episodes != {0, 1}:
        raise IncompleteCsvError(
            f"[smoke] expected evaluation_episode values {{0, 1}}, found {observed_episodes}"
        )
    if observed_env_seeds != {EVAL_SEED_BASE, EVAL_SEED_BASE + 1}:
        raise IncompleteCsvError(
            f"[smoke] expected evaluation_env_seed values "
            f"{{{EVAL_SEED_BASE}, {EVAL_SEED_BASE + 1}}}, found {observed_env_seeds}"
        )

    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(smoke_path, index=False)
    print(f"[smoke] PASS -> {smoke_path}")
    return smoke_path


def run_full_batch() -> tuple[list[Path], list[Path]]:
    """Evaluate all 5 algos x 3 regimes (each across 3 training seeds).

    Returns (written_paths, skipped_complete_paths).
    """
    written: list[Path] = []
    skipped: list[Path] = []

    for algo in ALGOS:
        for regime in REGIMES:
            out_path = final_csv_path(algo, regime)

            if out_path.exists():
                validate_complete_csv(
                    out_path, algo, regime, SEEDS, N_EPISODES, EVAL_SEED_BASE
                )
                print(f"[skip-complete] {algo}/{regime.name.lower()} -> {out_path}")
                skipped.append(out_path)
                continue

            print(
                f"[evaluate] {algo}/{regime.name.lower()} "
                f"seeds={SEEDS} steps={STEPS} episodes={N_EPISODES} ..."
            )
            evaluate_grid(
                algo=algo,
                regime=regime,
                seeds=SEEDS,
                steps_or_episodes=STEPS,
                n_episodes=N_EPISODES,
                model_dir=MODEL_DIR,
                output_csv=out_path,
                run_name=RUN_NAME,
                eval_seed_base=EVAL_SEED_BASE,
            )
            # Immediately re-validate what we just wrote before trusting it.
            validate_complete_csv(out_path, algo, regime, SEEDS, N_EPISODES, EVAL_SEED_BASE)
            print(f"[done] {algo}/{regime.name.lower()} -> {out_path}")
            written.append(out_path)

    return written, skipped


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        run_smoke_check()
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAIL: {exc}")
        print("Stopping before evaluating any final model.")
        return 1

    try:
        written, skipped = run_full_batch()
    except IncompleteCsvError as exc:
        print(f"[error] pre-existing final CSV failed completeness check: {exc}")
        print("Stopping. No final CSV was overwritten, appended to, or mixed.")
        return 1

    total_final_csvs = len(list(OUTPUT_ROOT.glob("*.csv")))
    total_rollouts_completed = (len(written) + len(skipped)) * len(SEEDS) * N_EPISODES

    print()
    print("=== final-v1 evaluation batch summary ===")
    print(f"newly written CSVs: {len(written)}")
    for p in written:
        print(f"  {p}")
    print(f"skipped-complete CSVs: {len(skipped)}")
    for p in skipped:
        print(f"  {p}")
    print(f"total final CSVs present under {OUTPUT_ROOT}: {total_final_csvs}")
    print(f"expected total episode rollouts (45 models x {N_EPISODES}): "
          f"{len(ALGOS) * len(REGIMES) * len(SEEDS) * N_EPISODES}")
    print(f"accounted-for episode rollouts (written + skipped-complete): "
          f"{total_rollouts_completed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

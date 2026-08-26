"""Evaluation + pit-opportunity diagnostics for the PPO baseline study.

Uses the existing, unmodified evaluator (f1_rl_safety.evaluate_rl.evaluate_grid,
deterministic policy) for the canonical per-episode CSVs, then replays the
identical deterministic episodes (same model, same episode seeds) to record
diagnostic-only statistics of the pit action channel at lap-final segments
(the only steps where a pit can take effect). No policy, reward, wrapper,
timing, or threshold is altered — measurement only.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
sys.path.insert(0, str(REPO / "src"))

from stable_baselines3 import PPO  # noqa: E402
from f1_rl_safety.evaluate_rl import evaluate_grid  # noqa: E402
from f1_rl_safety.f1_env import F1RaceEnv, RaceRegime  # noqa: E402

REGIMES = {
    "unconstrained": RaceRegime.UNCONSTRAINED,
    "rulebook": RaceRegime.RULEBOOK,
    "safe": RaceRegime.SAFE,
}
SEEDS = [0, 1, 2]
STEPS = 100_000
N_EPISODES = 50
MODEL_DIR = OUT / "models" / "gamma0999"
GAMMA = 0.999

manifest_rows = []

for regime_name, regime in REGIMES.items():
    for seed in SEEDS:
        model_path = (
            MODEL_DIR / "ppo" / regime_name /
            f"ppo_regime={regime_name}_seed={seed}_steps={STEPS}.zip"
        )
        eval_csv = OUT / "eval" / f"ppo_{regime_name}_seed{seed}_steps{STEPS}_eval{N_EPISODES}.csv"
        diag_csv = OUT / "diagnostics" / f"pit_opportunities_{regime_name}_seed{seed}.csv"

        t0 = datetime.now().isoformat(timespec="seconds")
        evaluate_grid(
            algo="ppo", regime=regime, seeds=[seed],
            steps_or_episodes=STEPS, n_episodes=N_EPISODES,
            model_dir=MODEL_DIR, output_csv=eval_csv,
        )
        t1 = datetime.now().isoformat(timespec="seconds")

        # --- diagnostic replay: identical seeds + deterministic policy ---
        model = PPO.load(str(model_path))
        env = F1RaceEnv(regime=regime, seed=seed)
        rows = []
        for ep in range(N_EPISODES):
            obs, _ = env.reset(seed=seed + ep)
            done = False
            while not done:
                at_pit_opportunity = (
                    env.current_segment_idx == env.n_segments - 1
                )
                action, _ = model.predict(obs, deterministic=True)
                if at_pit_opportunity:
                    rows.append({
                        "episode": ep,
                        "lap": env.current_lap,
                        "action0_pit": float(action[0]),
                        "pit_intent": bool(action[0] > 0.5),
                        "decoded_tyre_choice": int(np.clip(round(float(action[1])), 0, 4)),
                        "action2_risk": float(np.clip(action[2], -1.0, 1.0)),
                        "tyre_age": int(env.tyre_age),
                        "tyre_wear": float(env.tyre_wear),
                    })
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
        env.close()
        pd.DataFrame(rows).to_csv(diag_csv, index=False)
        t2 = datetime.now().isoformat(timespec="seconds")

        manifest_rows.append({
            "regime": regime_name,
            "seed": seed,
            "training_steps": STEPS,
            "gamma": GAMMA,
            "eval_episodes": N_EPISODES,
            "model_path": str(model_path.relative_to(REPO)),
            "eval_csv": str(eval_csv.relative_to(REPO)),
            "diagnostics_csv": str(diag_csv.relative_to(REPO)),
            "train_command": (
                f"PYTHONPATH=src .venv_f1/bin/python -m f1_rl_safety.train_rl "
                f"--algo ppo --regime {regime_name} --steps {STEPS} --seed {seed} "
                f"--mode experiment --output-dir {OUT.relative_to(REPO)} --gamma 0.999 --run-name gamma0999"
            ),
            "eval_invocation": (
                f"evaluate_grid(algo='ppo', regime=RaceRegime.{regime.name}, "
                f"seeds=[{seed}], steps_or_episodes={STEPS}, n_episodes={N_EPISODES}, "
                f"model_dir='{MODEL_DIR.relative_to(REPO)}', output_csv='{eval_csv.relative_to(REPO)}')"
            ),
            "eval_started": t0,
            "eval_finished": t1,
            "diagnostics_finished": t2,
        })
        print(f"done: {regime_name} seed {seed}", flush=True)

pd.DataFrame(manifest_rows).to_csv(OUT / "manifest.csv", index=False)
print("manifest written")

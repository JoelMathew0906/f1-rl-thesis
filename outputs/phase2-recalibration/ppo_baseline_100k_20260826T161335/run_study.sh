#!/bin/bash
# PPO baseline study: 3 regimes x 3 seeds x 100k steps, then eval+diagnostics.
set -e
cd "$(dirname "$0")/../../.."
OUT=outputs/phase2-recalibration/ppo_baseline_100k_20260826T161335
for regime in unconstrained rulebook safe; do
  for seed in 0 1 2; do
    echo "[$(date +%T)] training $regime seed $seed"
    PYTHONPATH=src .venv_f1/bin/python -m f1_rl_safety.train_rl \
      --algo ppo --regime $regime --steps 100000 --seed $seed \
      --mode experiment --output-dir "$OUT" \
      > "$OUT/trainlogs/train_${regime}_seed${seed}.log" 2>&1
  done
done
echo "[$(date +%T)] training complete; starting eval + diagnostics"
PYTHONPATH=src .venv_f1/bin/python "$OUT/eval_and_diagnose.py"
touch "$OUT/DONE"
echo "[$(date +%T)] study complete"

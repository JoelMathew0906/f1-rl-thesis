#!/bin/bash
# PPO gamma ablation: 3 regimes x 3 seeds x 100k steps at gamma=0.999.
set -e
cd "$(dirname "$0")/../../.."
OUT=outputs/phase2-recalibration/ppo_gamma_ablation_0999_20260826T162818
for regime in unconstrained rulebook safe; do
  for seed in 0 1 2; do
    echo "[$(date +%T)] training $regime seed $seed gamma=0.999"
    PYTHONPATH=src .venv_f1/bin/python -m f1_rl_safety.train_rl \
      --algo ppo --regime $regime --steps 100000 --seed $seed \
      --mode experiment --output-dir "$OUT" --gamma 0.999 --run-name gamma0999 \
      > "$OUT/trainlogs/train_${regime}_seed${seed}_gamma0999.log" 2>&1
  done
done
echo "[$(date +%T)] training complete; starting eval + diagnostics"
PYTHONPATH=src .venv_f1/bin/python "$OUT/eval_and_diagnose.py"
touch "$OUT/DONE"
echo "[$(date +%T)] study complete"

# Phase 2: Recalibration Workflow

This branch (`phase2-recalibration`) is for controlled reward-regime validation
first, followed by Silverstone 2024 recalibration and algorithm extension —
each only after separate approval.

## Output convention

All new outputs must be written under:

```
outputs/phase2-recalibration/{logs,models,eval}/
```

This directory tree is not pre-created (git does not track empty directories);
scripts should create it on first run via `--output-dir outputs/phase2-recalibration`
or an equivalent `--run-name phase2-recalibration`.

## Baseline artefacts

Existing historical `V1`/`V2`/`V3`-labelled artefacts (`models/`, `models_v2/`,
`models_v2_eval/`, `logs/`, `outputs/reward_v2/`, `outputs/debug/`) remain
baseline references and must not be overwritten, renamed, or migrated.

## Cache path

`data/fastf1_cache/` is the intended future FastF1 cache location. The code
still reads/writes `data/cache/` until a separately approved migration.

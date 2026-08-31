# Phase 05 — segment-identity consumer inventory

The compatibility surface for segment identity in this repository. Deliverable for
`docs/phase-05-plan.md`, scope item 2 and implementation step 2. Companion to
`docs/phase-05-geometry-schema.md`.

**Verdict: compatibility PRESERVED. No migration plan required.**

## Purpose

Before any geometry schema could be proposed, every consumer of segment identity had
to be enumerated so the blast radius of a change was known rather than assumed. This
document is that enumeration, and the evidence behind the compatibility verdict.

## Method

Grep for `segment`, `segment_id`, `corner_number`, `corner_name`, `segment_type`,
`approx_radius`, `n_segments` and `sector` across `src/`, `scripts/`, `notebooks/`,
`configs/` and committed `outputs/`, followed by reading each hit in context. The
stray `src/f1_rl_safety/.venv/` directory (a gitignored Python 3.9 virtualenv,
unrelated to the project interpreter `.venv_f1`) was excluded from results.

## The `TrackSegment` field surface

Defined at `src/f1_rl_safety/track.py:13-35`; the CSV columns mirror it exactly.

| Field | Consumed by code? |
|---|---|
| `id` | **Yes** |
| `segment_type` | **Yes** |
| `corner_number` | **Yes** |
| `corner_name` | **Yes** |
| `approx_radius` | **Yes** |
| `length` | **No — read by nothing** |
| `sector` | **No — read by nothing** |
| `start_x`, `start_y`, `end_x`, `end_y` | **No — read by nothing** |

This is the central finding of the inventory. Six of the eleven fields in the
existing segments CSV are inert metadata. It is why the two documented upstream
defects (segment 1's zero `length`, the scalar `sector` under-assignment) are
**non-behavioural**, and why they can be documented and corrected in a separate
additive artefact without any risk to simulator behaviour.

## Consumer inventory

Risk is the consequence *if the field's definition or values were changed* — not the
risk of this phase, which changes none of them.

### Environment — `src/f1_rl_safety/f1_env.py`

| Line(s) | Field | How consumed | Risk if changed | Verdict |
|---|---|---|---|---|
| 150-151 | — | `track_segments` loaded; `n_segments = len(...)` = 36 | **CRITICAL** | Preserved |
| 262, 349-354 | index | `current_segment_idx` advance and wrap; the wrap sets `completed_lap`, which gates tyre age, wear, fuel **and pit execution** (372-378) | **CRITICAL** — pit mechanics depend on it | Preserved |
| 345-346 | — | Segment for the current step, stored as `last_segment` | HIGH | Preserved |
| 445-455, 615 | `n_segments` | Uniform divisor for base, compound, wear, fuel, risk-gain and overcaution terms. **No per-segment `length` weighting anywhere** | **CRITICAL** | Preserved |
| 500-506 | `segment_type` | `is_corner` → `base *= 8.0`; else `base *= 0.5` | **CRITICAL** — the test is `seg_type == "corner"` with no else-branch error, so a rename or recase fails **silently** | Preserved |
| 510-512 | `approx_radius` | `radius_term = 0.015 / max(radius, 1.0)`, corners only (straights are `None`) | **LOW** — numerically inert, 0.015%–0.086% of the corner base rate | Preserved |
| 517 | `segment_type` | Risk-envelope tolerance `safe_risk` = 0.3 corner / 0.6 straight | **CRITICAL** — same silent-failure mode | Preserved |
| 572-574 | `segment_type` | `_crash_reason` → `"combined_load_exceedance"` | HIGH | Preserved |
| 386-389 | `id`, `corner_number`, `corner_name`, `segment_type` | Crash-log payload dict | **HIGH** — public dict contract | Preserved |
| 423-427 | same, plus `segment_risk` | `info` dict returned from every `step()` | **HIGH** — public dict contract | Preserved |

### Training — `src/f1_rl_safety/train_rl.py`

| Line(s) | Field | How consumed | Risk | Verdict |
|---|---|---|---|---|
| 179-181 | `segment_id`, `corner_number`, `corner_name` | Crash-log CSV columns | **HIGH** — persisted output schema | Preserved |

### Evaluation — `src/f1_rl_safety/evaluate_rl.py`

| Line(s) | Field | How consumed | Risk | Verdict |
|---|---|---|---|---|
| 48-62 | all four | `_format_crash_site` builds `"segment=…;type=…;corner_number=…;corner_name=…"` | **HIGH** — this exact string is persisted in 11 committed CSVs | Preserved |
| 96-97, 134-136 | — | `first_crash_site` / `final_crash_site` columns | **HIGH** | Preserved |
| 158-161, 176-179 | `segment_id`, `segment_type`, `corner_number`, `corner_name` | The four `crash_*` evaluation CSV columns, NULL-filled on no-crash rows | **HIGH** — evaluation CSV schema | Preserved |

### Analysis scripts

| File | Line(s) | Field | How consumed | Risk | Verdict |
|---|---|---|---|---|---|
| `scripts/analyse_pace_profiles.py` | 115, 127, 153, 172 | `n_segments`, `current_segment_idx` | Aggregates per-segment times into laps; **admits a lap only if exactly `n_segments` segments were recorded** (172) | **CRITICAL** — the direct regression guard for the frozen evidence | Preserved |
| `scripts/analyse_pace_profiles.py` | 532, 545 | `n_segments` | Records `n_segments_per_lap` in the manifest | **CRITICAL** | Preserved |
| `scripts/diagnose_env_control.py` | 176-177, 191-194, 211-220 | `segment_type`, `approx_radius` | Selects the first corner and first straight via `next(s for s in segments if str(s.segment_type).lower() == …)`; probes `_segment_crash_prob` | **HIGH** — `next(...)` raises `StopIteration` if either type name disappears | Preserved |
| `scripts/diagnose_env_control.py` | 66-67, 244-245, 300, 429, 448-473, 492-512, 682, 734, 768 | `current_segment_idx`, `n_segments`, `segment_id` | Lap-final-segment pit targeting (`seg_idx == n_segments - 1`); records `crash_segment_id` | **HIGH** — depends on both count and index semantics | Preserved |
| `scripts/validate_reward_regimes.py` | 53-76 | `current_segment_idx`, `n_segments` | Same lap-final pit targeting: `on_final_segment = env.current_segment_idx == env.n_segments - 1` | **HIGH** | Preserved |

### Notebooks

| File | Result |
|---|---|
| `notebooks/01_environment_and_data.ipynb` | **Zero** references |
| `notebooks/02_training_and_logs.ipynb` | **Zero** references |
| `notebooks/03_evaluation_and_plots.ipynb` | **Zero** references |

No notebook reads any segment or corner field. Verified by grep for `segment`,
`corner_number`, `corner_name` and `approx_radius` across all three files.

## Committed output compatibility surface

Eleven tracked CSVs persist segment identity. All eleven carry the same four-column
block `crash_segment_id`, `crash_segment_type`, `crash_corner_number`,
`crash_corner_name`, plus the two `_format_crash_site` string columns
`first_crash_site` and `final_crash_site`:

- `outputs/phase2-recalibration/reward_validation/` — 6 files
  (`ppo_{rulebook,safe,unconstrained}_seed0_steps25000_eval20_*.csv`, both the
  original and `_recal_` variants)
- `outputs/reward_v2/` — 3 files
  (`ppo_{rulebook,safe,unconstrained}_reward_v2_200k_seed0.csv`)

Additionally:

- `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/manifest.csv` carries
  `n_segments_per_lap`. **This is the frozen-evidence dependency.** Changing the
  segment count would silently invalidate the committed
  `matched_window_summary.csv` (2,732 rows) and the corresponding evidence section in
  `docs/drafts/reward_and_environment_recalibration_evidence.md`.

The Phase 03 and Phase 04 output manifests contain no per-segment identity;
`circuit_corners_rows` in the Phase 03 manifest is a corner *count*, not an identity
column.

## Why the schema is safe

**1. It is additive and lives in a separate file.** All new columns are written to
`outputs/phase-05-geometry/20260830T112139/segment_geometry.csv`, keyed on the
existing `segment_id`. `data/silverstone_2024_track_segments.csv` is byte-unchanged
(sha256 `059983bafcef957fc0954fae89cbff167c1572bfd81e66e36104d4b029c4126e`).

**2. No consumed field is touched.** `id`, `segment_type`, `corner_number`,
`corner_name` and `approx_radius` retain their existing names, types and values. The
schema republishes `approx_radius` verbatim as `approx_radius_legacy` for
documentation rather than substituting a corrected value.

**3. The structure is unchanged.** 36 segments, 18 `Straight` + 18 `Corner`,
alternating, `n_segments = 36`. Validation check `segment_type_matches_existing_csv`
confirms all 36 rows join to the existing CSV with identical `segment_type`.

**4. No dict contract changes.** No key was added to or removed from the environment
`info` dict or the crash-log payload, so `train_rl.py`, `evaluate_rl.py` and
`diagnose_env_control.py` are untouched.

**5. Even the loader's tolerance is not relied upon.** For the record,
`load_or_build_silverstone_segments` (`track.py:152-187`) constructs `TrackSegment`
from *named* attributes (`row.id`, `row.segment_type`, …) with no positional or
`**row` unpacking, so it would ignore unknown columns anyway. Because the new columns
live in a separate file, that tolerance is never exercised.

**6. Both documented defects are non-behavioural.** `length` and `sector` are the
fields the defects affect, and both are consumed by nothing.

## Regression evidence

The plan nominates `scripts/analyse_pace_profiles.py` as the direct regression guard,
because its lap aggregation admits a lap only when exactly `n_segments` segments were
recorded.

```
.venv_f1/bin/python scripts/analyse_pace_profiles.py --episodes-fixed 3 \
    --episodes-learned 2 --bootstrap 100 --label phase05_regression_check
```

Result: 243 fixed laps, 361 learned laps, 550 summary rows, manifest
`n_segments_per_lap = [36]` across all 10 manifest rows. The 36-segment assertion
holds. The temporary labelled output directory
`outputs/phase2-recalibration/pace_diagnostics/phase05_regression_check/` was removed
after the run.

Also confirmed unchanged in `git status`:
`outputs/phase2-recalibration/pace_diagnostics/20260827T124054/` and
`data/silverstone_2024_track_segments.csv`.

## Migration plan

**None required.** No consumer in this inventory needs any change. No output schema,
log schema or committed CSV is affected.

## If a future phase does change segment identity

Recorded here so the cost is known in advance, not discovered later.

- **Changing `n_segments` or the alternating structure** invalidates
  `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/` and the frozen
  evidence section that cites it. This is a plan stop condition and requires an
  approved migration plan.
- **Renaming or recasing `segment_type` values** fails **silently** at
  `f1_env.py:500-506` and `517` — a non-`"corner"` value is treated as a straight
  with no error — and raises `StopIteration` at
  `scripts/diagnose_env_control.py:193-194`. Any such change needs both call sites
  updated together.
- **Changing `approx_radius` semantics** is a dynamics change to
  `_segment_crash_prob` and requires separate approval. Note it is currently
  numerically inert, so a "fix" that made it physical would change hazard behaviour
  materially rather than marginally.
- **Adding or removing keys** in the `info` dict or crash-log payload propagates to
  `train_rl.py:179-181`, `evaluate_rl.py:158-161` and the 11 committed CSVs.

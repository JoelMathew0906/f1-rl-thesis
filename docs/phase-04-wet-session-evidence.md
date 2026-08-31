# Phase 04 — Wet-session calibration evidence

Deliverable for `docs/phase-04-plan.md`. **No environment, reward, hazard,
tyre-calibration, training, model or evaluator code was changed to produce this
document.** The only new code is a read-only measurement script,
`scripts/measure_wet_calibration.py`, which does not import `F1RaceEnv`.

`M/` below is shorthand for the curated measurement directory
`outputs/phase-04-wet-calibration/20260830T114500/`.

**Outcome: documented no-calibration decision.** No parameter change is
proposed, so no `docs/phase-04-proposed-calibration-diff.md` exists. Two
pre-registered stop conditions fired. Details in
[§9 Decision](#9-decision).

---

## 1. Gate check — Phase 03

Phase 03 recommendation, quoted verbatim from
`docs/phase-03-fastf1-feasibility-audit.md` §Recommendation:

> **Proceed.**

Committed at `2ddcee1` (`docs: FastF1 feasibility audit for 2024 British Grand
Prix sessions`), which contains the audit report, the audit script and the
curated audit outputs. The gate permits Phase 04 to begin.

Phase 03 constraints honoured here:

| Constraint | How honoured |
|---|---|
| Phase 04 not blocked; `Rainfall`, `Compound`, `TrackTemp`, lap timing confirmed clean | all four used; no material nulls encountered |
| No session asserted as canonically "the wet session" — classification is Phase 04's decision | classification performed per-lap from `Rainfall`, §3 |
| Session-level condition labelling is an **assumption**, not an observation | session treated as time-varying; no session-level label asserted |
| `pit_loss` not re-derived in Phase 03 | not re-derived here either; out of scope |
| Do not migrate the cache path | `data/cache` used; `data/fastf1_cache` untouched |
| Audit covered one event only | no generalisation claimed beyond the 2024 British GP |

**Correction to a stale record.** The `## Completion record` in
`docs/phase-03-plan.md` still reads `**Commit SHA:** _pending — not committed_`
and describes the curated outputs as "not yet committed". Both are now
out of date: the commit is `2ddcee1` and all five curated files are tracked.
This document does not edit that file; the discrepancy is recorded here and
flagged for the user.

## 2. Data source and provenance

**Primary calibration source:** `data/silverstone_2024_laps.csv`, 850 rows — the
exact file `F1RaceEnv._load_calibration_2024` reads. Produced by
`data_loader.extract_stint_and_lap_data(2024)`, which loads the 2024 British
Grand Prix **Race** session and applies a single filter,
`laps[laps["IsAccurate"] == True]` (`src/f1_rl_safety/data_loader.py:44`),
keeping 12 columns.

**Auxiliary columns** (protocol P1, Variant B, user-approved): re-read from the
*same already-cached Race session*, read-only — `LapStartTime`, `Deleted`,
`PitInTime`/`PitOutTime`, `TrackStatus`, `Position`, and
`session.weather_data` (`Rainfall`, `TrackTemp`, `AirTemp`). **No other session
was loaded**; FP1, FP2, FP3 and Qualifying laps were never touched, and no lap
from any other session entered any estimate.

**Provenance checks** [SOURCE: `M/manifest.csv`]:

| Check | Result |
|---|---|
| CSV rows unmatched against the cached session on `(Driver, LapNumber)` | **0** |
| `Compound` mismatches, CSV vs session | **0** |
| `TyreLife` mismatches, CSV vs session | **0** |
| `Stint` mismatches, CSV vs session | **0** |
| Session lap rows / weather rows | 960 / 147 |
| `Deleted` column available | yes |
| `data/cache` growth during the full run | **0 KB** |
| Reproduced `base_lap_time` | **92.4335 s** — identical to the live runtime value |

The measurement reproduces the live calibration path exactly. Every
`compound_offsets` and `deg_per_lap` value in `M/clip_test.csv` equals the
runtime value recorded in §4, to six decimals. This is the strongest available
evidence that the audit measures the same quantity the simulator uses.

**Cache note, stated honestly:** the full run caused **0 KB** growth. An earlier
reduced-scope smoke run bumped `fastf1_http_cache.sqlite` by 4,096 bytes
(58,859,520 → 58,863,616) — a single SQLite page allocation from a cache
revalidation, not new session data. All five session directories are
byte-identical in size before and after, and `data/fastf1_cache` is unchanged.

## 3. Session identification and lap classification

**The session is treated as time-varying, never as a single condition.** Phase 03
labels whole-session labelling an assumption; the Race was 51/147 rainfall-True,
so a single label would be indefensible.

Classification is from `Rainfall` alone (the directly observed boolean).
`TrackTemp` and `Compound` corroborate and are recorded, and never override.
Laps within 2 weather samples of a `Rainfall` state change are classed `mixed`.
Weather sampling is ~1/minute: median join gap 30.75 s, maximum 60.01 s
[SOURCE: `M/weather_classification.csv`].

| Scope | Laps | wet | mixed | dry | median `TrackTemp` |
|---|---|---|---|---|---|
| All | 850 | 276 | 174 | 400 | 24.37 °C |
| SOFT | 158 | 0 | 57 | 101 | 24.95 °C |
| MEDIUM | 465 | 103 | 88 | 274 | 24.53 °C |
| HARD | 39 | 0 | 14 | 25 | 24.93 °C |
| **INTERMEDIATE** | **188** | **173** | **15** | **0** | **21.00 °C** |

[SOURCE: `M/weather_classification.csv`] — all **observed**.

**Rainfall-vs-compound agreement.** Zero intermediate laps were run on a lap
classified `dry`: 173/188 wet, 15/188 mixed. The observed weather signal and the
teams' tyre choice agree completely for the compound under audit, and the
intermediate median `TrackTemp` is 3.5–4.0 °C below every dry compound's. Lap
classification therefore rests on an observation, not an assumption, and
**stop condition 7 does not fire.**

In the other direction, 103 of 662 dry-compound laps ran while
`Rainfall == True` — expected in a mixed race as slicks are run either side of
the rain. This does not affect intermediate classification, but it does
contaminate the dry baseline, which is exactly why baseline `B_dry_green` was
pre-registered (§5).

**Conclusion:** a verifiable wet condition exists within the 2024 British Grand
Prix Race, localised to a window in the middle of the race, and the intermediate
laps within it are cleanly identified. Session identification **succeeds**.

## 4. Live calibration values, and the P8 clip test

Runtime values from a single default `F1RaceEnv()` construction, and the raw
unclipped slopes the same data produces [SOURCE: `M/clip_test.csv`]:

| idx | compound | laps | live `deg_per_lap` | **raw unclipped slope** | clip active | live offset | live `typical_stint` | stint clip |
|---|---|---|---|---|---|---|---|---|
| 0 | SOFT | 158 | 0.044105 | 0.044105 | no | −1.2705 | 13 | no |
| 1 | MEDIUM | 465 | 0.201839 | 0.201839 | no | +0.2635 | 24 | no |
| 2 | HARD | 39 | 0.168399 | 0.168399 | no | −2.5615 | 13 | no |
| 3 | **INTERMEDIATE** | **188** | **0.010000** | **−0.071881** | **FLOOR** | **+10.7135** | 9 | no |
| 4 | WET | **0** | 0.250000 | — | fallback | +11.0000 | 10 | fallback |

`base_lap_time = 92.4335 s`; `pit_loss = 21.5 s` (hardcoded in both the data and
fallback paths, never derived).

**Finding P8-1 — the live INTERMEDIATE degradation value is an artefact of the
clip floor, not an estimate.** The raw OLS slope of lap time on `TyreLife` over
all 188 intermediate laps is **−0.071881 s per lap**: negative. Tyres appear to
get *faster* with age. `np.clip(slope, 0.01, 0.40)` at
`src/f1_rl_safety/f1_env.py:235` silently converts this to `+0.010000`, the
floor. The simulator therefore models intermediate tyres as very slightly
degrading on the strength of data that shows the opposite sign.
**Labelled: observed** (the raw slope) and **non-comparable** (the live value,
which corresponds to no estimate).

**Finding P8-2 — the clip is not systemic.** No other compound is clipped, and
no `typical_stint` is clipped. INTERMEDIATE is the only affected parameter.

**Finding P8-3 — INTERMEDIATE and WET differ in kind.** INTERMEDIATE is
data-derived from 188 laps. WET has **zero** laps in the calibration source, so
all three of its parameters are fallback constants and always have been.

## 5. Estimates, with uncertainty and sample counts

Sample after the pre-registered P3 exclusion cascade
[SOURCE: `M/attrition.csv`]:

| Step | Rule | Rows | Dropped |
|---|---|---|---|
| 0 | CSV as delivered | 850 | 0 |
| 1 | `IsAccurate == True` (already applied upstream) | 850 | 0 |
| 2 | drop null `LapTime`/`Compound`/`TyreLife`, non-finite seconds | 850 | 0 |
| 3 | drop `Deleted == True` — the one exclusion `IsAccurate` does not apply | 841 | **9** |
| 4 | drop `TrackStatus != "1"` (asserted, not assumed) | 841 | 0 |
| 5 | drop first surviving lap after any gap left by a non-green lap | 790 | **51** |
| 6 | drop `(Driver, Stint)` clusters with < 3 surviving laps | 787 | 3 |

In-scope INTERMEDIATE: **158 laps, 22 `(Driver, Stint)` clusters, 19 drivers**;
`TyreLife` 3–12 (span 9); cluster sizes 3–10, median 8.

All intervals are cluster bootstraps over `(Driver, Stint)`, resampled with
replacement, B = 10,000, percentile 95%, seed 20260830.

### 5.1 Pace offset — passes every threshold

| Baseline | Offset (s) | 95% CI | CI halfwidth | n laps | n clusters | n drivers |
|---|---|---|---|---|---|---|
| `B_current` — median of all dry laps (reproduces live code) | **+10.858** | [10.316, 11.755] | 0.720 | 158 | 22 | 19 |
| `B_dry_green` — dry laps classified `dry` by §3 | +10.966 | [10.442, 11.790] | 0.674 | 158 | 22 | 19 |
| `B_medium` — MEDIUM laps only | +10.613 | [10.118, 11.412] | 0.647 | 158 | 22 | 19 |

[SOURCE: `M/estimates.csv`] — **derived**. Baseline uncertainty is propagated:
both the target and the baseline cluster sets are resampled.

The three baselines agree within 0.35 s, and location-estimator sensitivity is
small — median 10.858, 10 % trimmed mean 11.086, mean 10.745 (R2,
[SOURCE: `M/robustness.csv`]). **The live value +10.7135 lies inside all three
confidence intervals.**

### 5.2 Degradation slope — fails, and the three fits disagree in sign

| Fit | Raw lap time | 95% CI | CI excludes 0 | Fuel-adjusted | 95% CI |
|---|---|---|---|---|---|
| Pooled OLS (reproduces `f1_env.py:233`) | **−0.227861** | [−0.366, −0.084] | yes | −0.174785 | [−0.312, −0.036] |
| Within-stint fixed effects | **−0.116270** | [−0.210, **+0.000449**] | **no** | −0.068193 | [−0.161, +0.049] |
| Condition-controlled (+`LapNumber`, +wet class) | **+0.541856** | [+0.123, +0.761] | yes | +0.541856 | [+0.125, +0.762] |

[SOURCE: `M/estimates.csv`] — **derived**. Pooled fit diagnostics: R² = 0.0673
(raw), 0.0430 (fuel-adjusted); max Cook's distance 0.0875.

Three estimators of the same quantity return −0.23, −0.12 and +0.54 s per lap.
Two are negative, one is strongly positive, and the sign of the answer is
determined by the specification rather than by the data.

### 5.3 Why the slope is not identifiable

The decisive diagnostic:

| Diagnostic | Value |
|---|---|
| Pearson r(`TyreLife`, `LapNumber`), pooled | 0.7826 |
| Pearson r(`TyreLife`, `LapNumber`), **within-stint demeaned** | **1.0000** |
| VIF, condition-controlled design | `TyreLife` 2.90, `LapNumber` 2.58, `is_mixed` 1.35 |
| `LapNumber` range of in-scope intermediate laps | 22–38 (145 of 158 laps in laps 29–38) |

**Within a stint, tyre age and lap number are the same variable up to a
constant shift** — the correlation is exactly 1. Every intermediate stint began
at `TyreLife` 3 or 4 after in-lap and out-lap removal, and all 22 stints are
compressed into a 17-lap window of a drying race. Consequently:

- The within-stint estimator cannot separate tyre degradation from any
  time-varying track effect, because within a stint the two are numerically
  indistinguishable.
- The condition-controlled estimator's `TyreLife` coefficient is therefore
  identified almost entirely from **between-stint** differences in when stints
  began — precisely the mechanism that produced the documented SOFT artefact.
  Its +0.542 s per lap is not a credible degradation rate.
- Track condition was genuinely non-monotonic across the window: median
  `TrackTemp` 21.97 → 20.70 → 20.80 → 21.30 °C and median lap time 106.96 →
  102.74 → 104.05 → 102.52 s over laps 22-26 / 27-30 / 31-34 / 35-38, while
  median `TyreLife` moved 4.5 → 3.0 → 6.0 → 9.0.

**Labelled: observed** (the correlations and ranges), **derived** (the VIFs),
**non-comparable** (any degradation rate purporting to separate the two).

### 5.4 Per-stint instability

Of 22 within-stint slopes, **14 are negative and 8 are positive**
[SOURCE: `M/stint_summary.csv`]. Individual values span −0.722 (ZHO stint 3) to
+0.948 (PER stint 2). The sign is close to a coin flip across stints, on stints
of 3–10 laps.

### 5.5 WET

**Zero laps in the calibration source.** No estimate was attempted and none will
be fabricated. All three WET parameters — offset 11.0 s, `deg_per_lap` 0.25,
`typical_stint` 10 — are fallback constants from
`src/f1_rl_safety/f1_env.py:173-197`. **Labelled: assumption, unevidenced.**

## 6. Robustness checks

[SOURCE: `M/robustness.csv`]

| ID | Perturbation | n laps | pooled | within | cond. | within 95% CI | signs agree |
|---|---|---|---|---|---|---|---|
| R0 | primary specification | 158 | −0.2279 | −0.1163 | +0.5419 | [−0.207, +0.002] | yes |
| R1 | P3 rules 4-6 removed (live-path filtering) | 188 | −0.0719 | **+0.0207** | +0.6851 | [−0.072, +0.141] | **no** |
| R3 | exclude used sets (`FreshTyre == False`) | 158 | −0.2279 | −0.1163 | +0.5419 | [−0.211, +0.000] | yes |
| R4 | leave-one-driver-out, 19 refits | 158 | [−0.248, −0.195] | [−0.139, −0.092] | — | — | yes |
| R4 | leave-one-stint-out, 22 refits | 158 | [−0.248, −0.179] | [−0.139, −0.092] | — | — | yes |
| R5 | wet-classified laps only (mixed excluded) | 144 | −0.0363 | **+0.0975** | +0.5762 | [+0.006, +0.206] | **no** |
| R6 | fuel-adjusted lap time | 158 | −0.1748 | −0.0682 | +0.5419 | [−0.162, +0.055] | yes |

**R1 and R5 flip the sign of the within-stint slope.** Removing just 15 mixed
laps (R5) moves it from −0.116 to +0.098 with a CI that excludes zero on the
positive side. The estimate's sign is a function of the exclusion rules.

R3 is numerically identical to R0: no in-scope intermediate lap is on a used
set, so used-set contamination is not a factor here.

R4 shows no single driver or stint dominates — the instability is distributional,
not one outlier.

### 6.1 R7 — falsification test on the protocol itself

Does the protocol reproduce the *known* artefact it was designed to catch?

| Compound | laps | clusters | pooled | within | cond. | within 95% CI | signs agree |
|---|---|---|---|---|---|---|---|
| SOFT | 145 | 12 | 0.0844 | 0.1251 | 0.1992 | [0.047, 0.193] | yes |
| MEDIUM | 447 | 23 | 0.2052 | 0.1782 | 0.2261 | [0.158, 0.199] | yes |
| HARD | 37 | 3 | 0.1742 | 0.0708 | 0.1541 | [−0.070, 0.116] | yes |

The protocol behaves as intended. For MEDIUM — the best-sampled compound, 447
laps across 23 stints — all three estimators agree within 0.05 s per lap and the
CI is tight, which is what an identifiable parameter looks like. For SOFT the
within-stint slope (0.125) is nearly **three times** the live value
(0.044105), confirming that the documented SOFT artefact is a pooling artefact
and that the within-stint estimator detects it. For HARD, 3 clusters produce a
CI spanning zero, consistent with the −2.5615 s offset anomaly noted in §4 being
an undersampling artefact.

This contrast is the strongest argument that the INTERMEDIATE failure is a
property of the data, not of the method: the same code, on the same race,
produces a clean identifiable answer for MEDIUM and an unidentifiable one for
INTERMEDIATE.

## 7. Threshold gate (P9, pre-registered and approved before measurement)

[SOURCE: `M/thresholds.csv`] — 9 of 11 criteria passed.

| Group | Criterion | Required | Observed | Pass |
|---|---|---|---|---|
| offset | n laps ≥ 30 | 30 | 158 | ✅ |
| offset | n clusters ≥ 6 | 6 | 22 | ✅ |
| offset | n drivers ≥ 3 | 3 | 19 | ✅ |
| offset | CI halfwidth ≤ 2.0 s | ≤ 2.0 | 0.7195 | ✅ |
| slope | n laps ≥ 40 | 40 | 158 | ✅ |
| slope | n clusters ≥ 6 | 6 | 22 | ✅ |
| slope | clusters with ≥ 5 laps ≥ 3 | 3 | 19 | ✅ |
| slope | `TyreLife` span ≥ 8 | 8 | 9 | ✅ |
| slope | within-stint CI excludes 0 | true | **false** ([−0.210, +0.000449]) | ❌ |
| slope | pooled and within-stint agree in sign | true | −0.2279 / −0.1163 | ✅ |
| slope | raw unclipped slope > 0 and CI excludes 0 | true | **raw −0.071881** | ❌ |

**Offset: 4/4 passed. Slope: 6/8 passed, and the two failures are the decisive
ones.** Under the pre-registered rule "no parameter is proposed unless every
applicable criterion is met", no degradation parameter may be proposed.

## 8. Limitations

- **Single event, single session.** One mixed-condition race at one circuit. No
  claim generalises beyond it. No real-world meteorological or regulatory
  validity is claimed.
- **Tyre age is confounded with elapsed time by construction** (§5.3), and
  within a stint the confounding is exact. This is not a limitation that more
  careful modelling of this sample can remove.
- **Fuel burn-off** acts opposite to degradation. The adjustment uses the
  environment's own model (`fuel_level = max(0, 1 − (LapNumber−1)/52)`,
  coefficient 2.5 s), so the linear burn-off is an **assumption**. It changes
  the slope magnitude but not the identifiability problem.
- **Traffic is uncontrolled.** No gap-to-car-ahead exists in the source; even
  `Position` gives running order, not proximity. In a mixed race, intermediate
  runners are frequently out of position. Declared an uncontrolled confound.
- **Minimum observable tyre age is 3.** `IsAccurate` removes out-laps, so the
  first two laps of every stint — where degradation is most identifiable — are
  absent.
- **`typical_stint` is biased low, and no threshold was pre-registered for it.**
  The live INTERMEDIATE value of 9 is the median of *`IsAccurate`-surviving* laps
  per stint. Because in-laps and out-laps are removed, this understates true
  stint length by roughly 2 laps. Since `typical_stint` feeds `old_tyre_term` in
  `_segment_crash_prob` (`f1_env.py:521-526`), the hazard model likely begins
  penalising intermediate tyre age about 2 laps too early. **No change is
  proposed**, because the protocol pre-registered no threshold for this
  quantity; it is recorded as an open finding.
- **Deleted laps are not caught by `IsAccurate`.** Nine were removed here
  (attrition step 3). The live calibration path does **not** remove them.
- **A second dry-compound anomaly exists and was not addressed.** HARD is
  calibrated 2.5615 s *faster* than the dry baseline and faster than SOFT, from
  39 laps across 3 stints. Modifying dry parameters is a plan non-goal, so this
  is reported only; it does contaminate `B_current`, which is why two
  alternative baselines were pre-registered.
- **`pit_loss` was not re-derived.** Out of scope, as in Phase 03.

## 9. Decision

**Documented no-calibration decision.** Two pre-registered stop conditions
fired:

1. *"The raw INTERMEDIATE slope is negative or indistinguishable from zero."*
   The raw unclipped slope is **−0.071881 s per lap**; the live value
   `0.010000` exists only because of the clip floor.
2. *"The degradation estimate appears driven by short or unequal stints."*
   Confirmed and strengthened: within-stint tyre age and lap number are
   perfectly collinear (r = 1.0000), 14 of 22 stint slopes are negative and 8
   positive, and removing 15 mixed laps flips the pooled sign.

Per-parameter outcomes:

| Parameter | Live value | Finding | Decision |
|---|---|---|---|
| `compound_offsets[3]` INTERMEDIATE | +10.7135 s | Measured +10.858 s, 95% CI [10.316, 11.755], 158 laps / 22 stints / 19 drivers. Live value **inside** all three baseline CIs. Robust to baseline, location estimator and exclusion rules. | **Corroborated. No change proposed** — the 0.145 s difference is a fifth of the CI halfwidth and not material. |
| `deg_per_lap[3]` INTERMEDIATE | 0.010000 | Clip-floor artefact. Raw slope −0.0719. Three estimators disagree in sign; within-stint CI includes zero; not identifiable from this sample. | **No-calibration decision.** Value retained; **relabelled as an unevidenced assumption**, not an estimate. |
| `typical_stint[3]` INTERMEDIATE | 9 | Data-derived, unclipped, but biased low by ~2 laps through out-lap removal. | **No change proposed** — no threshold pre-registered. Open finding. |
| `compound_offsets[4]`, `deg_per_lap[4]`, `typical_stint[4]` WET | 11.0 / 0.25 / 10 | Zero WET laps in the calibration source. | **No-calibration decision.** Fallback constants retained, **labelled unevidenced assumptions**. |

Because no parameter change is proposed, `docs/phase-04-proposed-calibration-diff.md`
is **not** created. The plan states a documented no-calibration decision is an
acceptable and complete outcome.

**What would be required for a defensible intermediate degradation estimate**
(each needs separate approval, and none is undertaken here): intermediate stints
spanning a wider and overlapping range of tyre ages so that age and elapsed time
are not collinear; more than one wet or mixed event, so between-event variation
can break the confound; or lap-level data retaining out-laps, to recover the
low-age end of the age axis. FP3 of this same weekend is 54/82 rainfall-True and
near-exclusively INTERMEDIATE, and is already cached — a candidate source, but
adding it is explicitly outside this phase's approved scope and would be a
separate proposal.

## 10. Separation from frozen dry-condition evidence

Nothing in this phase disturbs the evidence frozen at `e5f0d9e`, and no
INTERMEDIATE or WET parameter value *can* disturb it. Reasoning, not assertion:

- **No code changed.** The live calibration path is byte-for-byte unmodified;
  §2 shows this measurement reproduces its outputs exactly rather than altering
  them.
- **No parameter changed.** The decision is to change nothing.
- **The wet-exemption predicate is independent of parameter values.**
  `used_wet` at `src/f1_rl_safety/f1_env.py:670` is
  `any(c not in dry_indices for c in self.used_compounds)` — a test of compound
  *index* membership. It reads no offset, slope or stint value, so no wet
  parameter can alter the compound-diversity milestone
  (`f1_env.py:671-675`) or terminal compliance settlement (`f1_env.py:682`).
- **Dry episodes never read wet parameters.** `compound_offsets`, `deg_per_lap`
  and `typical_stint` are looked up by `self.tyre_compound`
  (`f1_env.py:447-451`, `:521-526`, `:566-570`). An episode that never fits
  index 3 or 4 never reads their values.
- **The frozen documents are untouched.**
  `docs/drafts/reward_and_environment_recalibration_evidence.md` is not edited.
  The two locked wording decisions and all five `[VERIFY]` markers stand
  unchanged; none concerns wet compounds.
- **Future wet results will be reported separately.** Any later wet calibration
  is documented in a new, separately dated artefact, never by rewriting the
  frozen pace–survival section.

## 11. Reproducibility

```
.venv_f1/bin/python -m compileall -q src scripts
.venv_f1/bin/python scripts/measure_wet_calibration.py --smoke
.venv_f1/bin/python scripts/measure_wet_calibration.py --label 20260830T114500
```

Full run: 5.4 s, seed 20260830, B = 10,000, 0 KB cache growth, 36 KB of curated
output across nine files. No record-level CSV is written, so no new
`.gitignore` pattern is required.

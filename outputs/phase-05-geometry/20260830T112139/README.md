# Phase 05 — Silverstone additive geometry artefact

Generated 2026-08-30T11:25:55Z by `scripts/build_geometry_schema.py`.

## What this is, and what it is not

This is a **geometry-aware strategic simulator, not a validated digital twin.** Nothing here is validated against vehicle dynamics.

This artefact is **additive and read-only**. It adds supplementary geometry columns keyed on the *existing* `segment_id` values. It does not modify `data/silverstone_2024_track_segments.csv`, the 36-segment structure, `n_segments`, the hazard model, or any code.

**Explicitly not represented:**

- elevation (telemetry Z exists but is not incorporated; its unit is only inferred by analogy with X/Y and is untested here)
- camber and banking
- track width
- kerb, run-off and barrier geometry
- surface state, grip or friction model
- racing-line model
- corner entry/exit or apex speeds
- fitted corner radii (the legacy approx_radius is a chord proxy, not a radius)
- any vehicle-dynamics validation

## Structure

36 segments: 18 `Straight` at odd `segment_id`, 18 zero-length `Corner` at even `segment_id`, strictly alternating. `segment_id` is the sole primary key and is unchanged.

`region_key` is the **authoritative stable identifier** for a named region (`T9`, `T14_to_T15`). It is built only from `corners.Number` and `corners.Letter`, so it is fully observed. `region_name_external` is **ASSUMED, external, and nullable** — downstream consumers must key on `region_key`, never on the name.

## Provenance

Single source session: **2024 Silverstone R**, loaded from the warm `data/cache` with `fastf1.Cache.offline_mode(True)` so a cache miss raises rather than fetching. No cache written; no other session loaded; `data/fastf1_cache` never referenced.

Per-column `observed` / `derived` / `assumed` labels with their derivation rules are in `column_provenance.csv`. Phase 03 (`docs/phase-03-fastf1-feasibility-audit.md`) established that `circuit_info.corners` returns 18 null-free rows identically across all five sessions of this event.

## The coordinate-unit question, resolved

Phase 03 recorded the unit of FastF1 `X`/`Y`/`Z` as an **open assumption**: the coordinate range is far too large for metres, and the documentation does not state a unit. This artefact resolves it by measurement.

Fitting each reference lap's cumulative `X`/`Y` polyline length against its metre-denominated `Distance` channel over **19 laps**:

- **9.994137 – 10.053723 raw units per metre** (mean **10.016642**, sd 0.018501)
- i.e. 1 raw unit = **0.099834 m**

`X`/`Y`/`Z` are therefore in **decimetres (1/10 m)**, with a fitted deviation from exactly 10.0 of 0.1664% and a lap-to-lap spread of 0.5949%. This is **derived with a stated residual**, not assumed.

The origin of the sub-percent deviation from exactly 10.0 is **not established here**. Candidate causes, none tested: the straight-line polyline between samples spaced ~7.4 m apart under-measures true curved arc; and the `Distance` channel is itself a FastF1-derived quantity rather than a coordinate measurement, so the two sides of the fit are not independent measurements of one path. Do not cite a mechanism for the residual without testing it.

The scale used for all `*_m` conversions in `segment_geometry.csv` is the multi-lap mean, **10.016642**.

`corners.Distance` was independently confirmed to be in metres on the same origin as the telemetry channel: see `corner_distance_check.csv`. 12/18 corners agree to 0.00 m with the nearest telemetry sample; the largest disagreement is 5.04 m, below one ~7.4 m sample interval.

## Distance boundaries

Region boundaries are **observed**, taken from `corners.Distance` in metres. Segment 1 is wrap-aware: it runs from the last corner, across the start/finish line, to Turn 1.

**Validation:** the 18 derived straight `arc_length_m` values sum to **5837.69 m** against a measured lap length of **5837.69 m** — closing to 0.000000 m.

**Caveat — two different reference paths.** `arc_length_m` follows the *racing line* (`corners.Distance` is snapped to the telemetry grid of a driven lap); `chord_length_m` is a *centreline map* chord between corner `X`/`Y` points. They are not measurements of the same path. The consequence is visible: **4 of 18 chords exceed their own arc length**, which is impossible on a single path. Never mix the two in one calculation.

The measured lap length (5837.69 m) is a racing-line distance and is **not** the published circuit length. Do not present it as such.

## Sector boundaries and their uncertainty

Derived by interpolating each reference lap's `SectorNSessionTime` onto that lap's `Distance` channel. Per-lap values are in `sector_boundaries.csv`; the summary is in `sector_boundary_summary.csv`.

Over 19 reference laps:

| Boundary | Absolute (m) | Lap fraction |
|---|---|---|
| End of S1 | 1804.99 – 1822.26 (mean 1811.75, range 17.27) | 0.31030 – 0.31276 (mean 0.31139, range 0.00246) |
| End of S2 | 4218.50 – 4251.93 (mean 4237.44, range 33.43) | 0.72729 – 0.72935 (mean 0.72829, range 0.00206) |
| End of S3 (lap) | 5799.12 – 5837.72 (mean 5818.50, range 38.60) | 1.00000 – 1.00009 (mean 1.00003, range 0.00009) |

**Absolute boundaries are NOT stable and are published as intervals, not point estimates.** End-of-S1 varies by 17.27 m and end-of-S2 by 33.43 m across laps, because total racing-line length itself varies by 38.60 m between drivers.

**Lap fractions are more stable, but by a margin that varies sharply by boundary.** Comparing each quantity's range against its own mean (range / mean, so absolute and fractional forms are comparable):

| Boundary | Absolute spread | Fractional spread | Fraction tighter by |
|---|---|---|---|
| End of S1 | 0.953% | 0.790% | 1.2x |
| End of S2 | 0.789% | 0.283% | 2.8x |
| End of S3 (lap) | 0.663% | 0.009% | 71.5x |

So the lap boundary is essentially exact in fractional terms, S2 is materially tighter, and **S1 is only marginally tighter** — the fractional form is not a uniform improvement, and the S1 boundary carries real uncertainty either way. **Use lap fraction, but treat the S1 boundary as uncertain to ~0.25% of a lap.** `sector_start`/`sector_end` in `segment_geometry.csv` are assigned from the mean fraction boundaries (0.31139, 0.72829); no segment boundary in this schema falls close enough to either sector boundary for that uncertainty to change its assignment.

3 straights cross a sector boundary, which is why sector is published as a `sector_start`/`sector_end` pair rather than the existing scalar. The existing scalar `sector` column is republished unchanged as `legacy_sector`.

## Radius proxy

`approx_radius_legacy` reproduces the existing `approx_radius` verbatim, so that the value the hazard model actually consumes is documented. It is computed by `track.py` as the **mean chord distance to the previous and next corner points** — a crude proxy, **not a fitted radius**, and not a physical quantity.

Converted at the fitted scale it spans 109.3 – 639.6 m. Two properties make it unfit for physical use: it is **larger** in fast open sections, inverting the intended "tighter corner is riskier" semantics; and in the hazard model it contributes only 0.015%–0.086% of the corner base rate, so corners are effectively identical to the hazard model.

**No corrected radius is published and no hazard change is proposed.** Rewiring the radius term would be a dynamics change requiring separate approval and would invalidate frozen evidence.

## Known upstream defects — documented, not repaired

Both are reproduced verbatim in the `legacy_*` columns and corrected only in the new columns. `data/silverstone_2024_track_segments.csv` is byte-unchanged.

1. **Segment 1's `length` is `0.0`, but its true extent is 615.93 m of arc (577.92 m of chord)** — the longest straight on the lap. `track.py:87` computes the wrap-around straight as `total_len - s_coords[-1]`, which is identically zero for any circuit. The segment's coordinates are correct; only `length` is wrong.
2. **The existing scalar `sector` under-assigns sector 1.** `track.py:90-91` applies the proportion rule to the *corner*, not to the straight preceding it, and Turn 1's cumulative distance is 0, so segment 1 is labelled sector 1 despite beginning in sector 3.

Neither defect is behavioural: `length` and `sector` are read by **no code** in this repository — verified by grep across `src/`, `scripts/` and `notebooks/`. Only `id`, `segment_type`, `corner_number`, `corner_name` and `approx_radius` are consumed.

## Assumed external naming

`region_name_external` is populated on 21 of 36 rows and NULL on the rest. FastF1's `corners` table has **no `Name` column** (its columns are exactly `X`, `Y`, `Number`, `Letter`, `Angle`, `Distance`), and the `corner_name` values in the existing CSV are `track.py` fallbacks (`"Turn 1"`…), not source data. These names are therefore **not derivable from the dataset** and are labelled `assumed`.

They are corroborated indirectly by the derived arc lengths: the two longest straights on the lap fall exactly between the corner pairs that published sources place at either end of the two named straights.

| Segment | region_key | Assumed name | Derived arc (m) | Rank |
|---|---|---|---|---|
| 11 | `T5_to_T6` | Wellington Straight | 727.54 | 2 of 18 |
| 29 | `T14_to_T15` | Hangar Straight | 866.44 | 1 of 18 |
| 1 | `T18_to_T1` | start/finish straight | 615.93 | 3 of 18 |

This is **corroboration, not derivation**. The names remain assumed.

`corner_angle_raw` carries `corners.Angle` verbatim. Its values are observed and null-free, but its **unit and reference direction are undocumented**, and two values look like angle-wrap artefacts (T18 = -178.57, T10 = -159.33). It is carried for future work and **consumed by nothing**.

## Backwards compatibility

Additive only, in a new file. No existing column is renamed, retyped or repurposed; no key is added to or removed from the environment `info` dict or the crash log; `n_segments` remains 36. The frozen pace-diagnostics evidence at `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/`, whose aggregation asserts exactly 36 segments per lap, is unaffected. Validation results are in `validation.csv`.

## Files

| File | Role |
|---|---|
| `segment_geometry.csv` | THE SCHEMA: 36 rows, one per existing segment_id, additive geometry columns. |
| `column_provenance.csv` | Per-column observed/derived/assumed label and derivation rule. |
| `sector_boundaries.csv` | Per-lap sector boundaries and coordinate-scale fit, one row per reference lap. |
| `sector_boundary_summary.csv` | Sector boundaries as INTERVALS (min/max/mean/sd/range), absolute metres and lap fractions. |
| `scale_fit_summary.csv` | Coordinate-scale fit summary establishing the decimetre unit. |
| `corner_distance_check.csv` | corners.Distance vs telemetry Distance at the nearest X/Y sample, per corner. Evidence the two share unit and origin. |
| `validation.csv` | Schema validation checks and results. |
| `README.md` | Provenance, scale fit, sector uncertainty, defects, compatibility, and the not-a-digital-twin statement. |
| `manifest.csv` | This file listing. |


# Phase 05 — Silverstone geometry schema

Deliverable for `docs/phase-05-plan.md`. `GA/` below is shorthand for the generated
artefact directory `outputs/phase-05-geometry/20260830T112139/`.

## Scope statement — read this first

**This is a geometry-aware strategic simulator, not a validated digital twin.**
Nothing in this schema is validated against vehicle dynamics. The schema records
where the existing 36 track segments sit along a lap and what they are called. It
does not model how a car behaves in them.

No environment, reward, hazard, tyre-calibration, training, model or evaluator code
was changed to produce this document. `data/silverstone_2024_track_segments.csv` is
byte-unchanged (sha256 `059983bafcef957fc0954fae89cbff167c1572bfd81e66e36104d4b029c4126e`).
The only new code is a read-only generation script, `scripts/build_geometry_schema.py`,
which imports nothing from `f1_env`, constructs no environment, and reads the existing
segments CSV for comparison only.

## Objective

Express Silverstone circuit geometry as a traceable, additive schema over the
existing 36-segment model, mapping every `segment_id` to a named track region with
documented provenance, without changing the segment count, the alternating
structure, or any dependent output, log schema or analysis.

## Provenance and data access

Single source session: **2024 British Grand Prix Race**, loaded from the existing
warm `data/cache` with `fastf1.Cache.offline_mode(True)` set, so a cache miss raises
rather than silently fetching. Verified afterwards: **0 KB `data/cache` growth**
(339,912 KB before and after), `data/fastf1_cache` never referenced, no session
other than the Race loaded.

Phase 03 (`docs/phase-03-fastf1-feasibility-audit.md`, section "Geometry readiness")
established the gate this phase depends on: `circuit_info.corners` returns 18
null-free rows with columns `X`, `Y`, `Number`, `Letter`, `Angle`, `Distance`,
identically across all five sessions of this event. Phase 03 also imposed two
constraints, both honoured here: position rows are filtered `Status != "OffTrack"`,
and the `X`/`Y`/`Z` coordinate unit was treated as unresolved until independently
established (see below).

## Structure — unchanged

36 segments: 18 `Straight` at odd `segment_id`, 18 zero-length `Corner` at even
`segment_id`, strictly alternating. `segment_id` remains the **sole primary key**.
`n_segments` remains 36. Every column in this schema is supplementary and lives in a
separate file.

`region_key` is the **authoritative stable identifier** — `T9` for a corner,
`T14_to_T15` for a straight. It is built only from `corners.Number` and
`corners.Letter`, so it is fully observed. Any downstream consumer needing a stable
region identifier must use `region_key`.

## The coordinate-unit question, resolved

Phase 03 recorded the unit of FastF1 `X`/`Y`/`Z` as an **open assumption**: the
coordinate range is far too large for metres and the documentation states no unit.
Phase 05's plan made resolving it a precondition for treating those fields as
physical distances.

Resolved by measurement, not assumption. Fitting each reference lap's cumulative
`X`/`Y` polyline length against its metre-denominated `Distance` channel, over 19
laps (the fastest accurate dry lap per driver, plus the fastest per dry compound):

| Quantity | Value |
|---|---|
| Fitted scale | **9.994137 – 10.053723 raw units per metre** |
| Mean (used for all `*_m` conversions) | **10.016642** (sd 0.018501) |
| Equivalently | 1 raw unit = **0.099834 m** |
| Deviation from exactly 10.0 | 0.1664% |
| Lap-to-lap spread | 0.5949% |

**`X`/`Y`/`Z` are in decimetres (1/10 m).** This is `derived` with a stated
residual. The *origin* of the sub-percent deviation is deliberately **not** asserted:
candidate causes are that the straight-line polyline between ~7.4 m-spaced samples
under-measures true curved arc, and that `Distance` is itself a FastF1-derived
channel rather than a coordinate measurement, so the two sides of the fit are not
independent measurements of one path. Neither was tested. Do not cite a mechanism
without testing it.

**`corners.Distance` is in metres, on the same origin as the telemetry channel.**
Evidence in `GA/corner_distance_check.csv`: comparing each corner's `Distance`
against the `Distance` of the nearest telemetry sample to its `X`/`Y` point, **12 of
18 corners agree exactly (0.00 m)** and the largest disagreement is **5.04 m**, below
one ~7.4 m sample interval. `corners.Distance` values are snapped to the telemetry
grid.

## Schema columns

Full per-column labels and derivation rules in `GA/column_provenance.csv`:
**9 observed, 10 derived, 2 assumed**.

| Column | Provenance | Definition |
|---|---|---|
| `segment_id` | observed | Join key; identical to `id` in the existing segments CSV |
| `segment_type` | observed | `Straight`/`Corner`, copied unchanged |
| `region_key` | observed | From `corners.Number`/`Letter` only. **Authoritative identifier** |
| `region_description` | derived | Human-readable rendering of `region_key`; no external input |
| `region_name_external` | **assumed** | Published circuit naming. **Nullable. Never use as a key** |
| `corner_number` | observed | `corners.Number`; NULL on straights, matching existing convention |
| `d_start_m`, `d_end_m` | observed | `corners.Distance` in metres |
| `arc_length_m` | derived | `d_end_m − d_start_m`, wrap-aware for segment 1. Racing-line distance |
| `chord_length_m` | derived | `X`/`Y` chord ÷ fitted scale. Centreline chord |
| `lap_fraction_start`, `lap_fraction_end` | derived | `d_*_m` ÷ lap length |
| `sector_start`, `sector_end` | derived | From the multi-lap mean fraction boundaries |
| `crosses_sector_boundary` | derived | True where `sector_start != sector_end` |
| `corner_angle_raw` | **assumed** | `corners.Angle` verbatim. Values observed; **semantics undocumented**. Carried, consumed by nothing |
| `approx_radius_legacy` | observed | Existing `approx_radius` verbatim, so hazard's actual input is documented |
| `approx_radius_legacy_m` | derived | `approx_radius_legacy` ÷ fitted scale. **Not a physical radius** |
| `legacy_length_raw` | observed | Existing `length` verbatim, **including the segment-1 defect** |
| `legacy_length_m` | derived | `legacy_length_raw` ÷ fitted scale |
| `legacy_sector` | observed | Existing scalar `sector` verbatim, **including its under-assignment** |

## Derivation rules

**Straights.** Straight segment `2i+1` spans from corner `i−1` to corner `i`. Segment
1 is wrap-aware: it runs from the last corner, across the start/finish line, to Turn
1. Its `arc_length_m` is `(lap_length − corners.Distance[17]) + corners.Distance[0]`.

**Corners.** Corner segment `2i+2` is a zero-length point at `corners.Distance[i]`,
matching the existing model's convention (`length = 0.0`, `start == end`). All 18
carry `arc_length_m = 0.0`.

**Distance boundaries.** Observed, from `corners.Distance` in metres. The validating
property: the 18 derived straight arc lengths sum to **5837.6948 m** against a
measured lap length of **5837.6948 m** — residual **0.000000 m**.

**Radius proxy.** No new radius is derived. `approx_radius_legacy` reproduces the
existing value verbatim so the quantity the hazard model actually consumes is
documented. `track.py:108-116` computes it as the **mean chord distance to the
previous and next corner points** — a crude proxy, not a fitted radius. At the fitted
scale it spans 109.3–639.6 m. Two properties make it unfit for physical use: it is
*larger* in fast open sections, inverting the intended "tighter corner is riskier"
semantics; and in `_segment_crash_prob` it contributes only **0.015%–0.086% of the
corner base rate**, so all 18 corners are effectively identical to the hazard model.
**No hazard change is proposed.** Rewiring the radius term would be a dynamics change
requiring separate approval and would invalidate frozen evidence.

## The 36-row mapping

`region_name_external` values below are **assumed**; `region_key` and the distances
are observed.

| id | type | region_key | region_name_external *(assumed)* | d_start_m | d_end_m | arc_m | chord_m | sector |
|---|---|---|---|---|---|---|---|---|
| 1 | Straight | `T18_to_T1` | start/finish straight | 5684.40 | 462.64 | **615.93** | 577.92 | 3→1 |
| 2 | Corner | `T1` | Abbey | 462.64 | 462.64 | 0.00 | 0.00 | 1 |
| 3 | Straight | `T1_to_T2` | — | 462.64 | 609.62 | 146.98 | 157.57 | 1 |
| 4 | Corner | `T2` | Farm Curve | 609.62 | 609.62 | 0.00 | 0.00 | 1 |
| 5 | Straight | `T2_to_T3` | — | 609.62 | 862.31 | 252.69 | 251.71 | 1 |
| 6 | Corner | `T3` | Village | 862.31 | 862.31 | 0.00 | 0.00 | 1 |
| 7 | Straight | `T3_to_T4` | — | 862.31 | 1034.44 | 172.13 | 150.28 | 1 |
| 8 | Corner | `T4` | The Loop | 1034.44 | 1034.44 | 0.00 | 0.00 | 1 |
| 9 | Straight | `T4_to_T5` | — | 1034.44 | 1230.14 | 195.71 | 177.52 | 1 |
| 10 | Corner | `T5` | Aintree | 1230.14 | 1230.14 | 0.00 | 0.00 | 1 |
| 11 | Straight | `T5_to_T6` | **Wellington Straight** | 1230.14 | 1957.68 | **727.54** | 714.24 | 1→2 |
| 12 | Corner | `T6` | Brooklands | 1957.68 | 1957.68 | 0.00 | 0.00 | 2 |
| 13 | Straight | `T6_to_T7` | — | 1957.68 | 2183.44 | 225.76 | 181.65 | 2 |
| 14 | Corner | `T7` | Luffield | 2183.44 | 2183.44 | 0.00 | 0.00 | 2 |
| 15 | Straight | `T7_to_T8` | — | 2183.44 | 2534.97 | 351.53 | 311.28 | 2 |
| 16 | Corner | `T8` | Woodcote | 2534.97 | 2534.97 | 0.00 | 0.00 | 2 |
| 17 | Straight | `T8_to_T9` | — | 2534.97 | 3056.98 | 522.01 | 515.73 | 2 |
| 18 | Corner | `T9` | Copse | 3056.98 | 3056.98 | 0.00 | 0.00 | 2 |
| 19 | Straight | `T9_to_T10` | — | 3056.98 | 3612.97 | 555.98 | 534.47 | 2 |
| 20 | Corner | `T10` | Maggotts | 3612.97 | 3612.97 | 0.00 | 0.00 | 2 |
| 21 | Straight | `T10_to_T11` | — | 3612.97 | 3699.61 | 86.65 | 90.49 | 2 |
| 22 | Corner | `T11` | Becketts | 3699.61 | 3699.61 | 0.00 | 0.00 | 2 |
| 23 | Straight | `T11_to_T12` | — | 3699.61 | 3840.87 | 141.26 | 144.27 | 2 |
| 24 | Corner | `T12` | Becketts | 3840.87 | 3840.87 | 0.00 | 0.00 | 2 |
| 25 | Straight | `T12_to_T13` | — | 3840.87 | 3985.54 | 144.67 | 138.07 | 2 |
| 26 | Corner | `T13` | Becketts | 3985.54 | 3985.54 | 0.00 | 0.00 | 2 |
| 27 | Straight | `T13_to_T14` | — | 3985.54 | 4131.83 | 146.29 | 140.27 | 2 |
| 28 | Corner | `T14` | Chapel | 4131.83 | 4131.83 | 0.00 | 0.00 | 2 |
| 29 | Straight | `T14_to_T15` | **Hangar Straight** | 4131.83 | 4998.27 | **866.44** | 846.72 | 2→3 |
| 30 | Corner | `T15` | Stowe | 4998.27 | 4998.27 | 0.00 | 0.00 | 3 |
| 31 | Straight | `T15_to_T16` | — | 4998.27 | 5454.80 | 456.53 | 432.42 | 3 |
| 32 | Corner | `T16` | Vale | 5454.80 | 5454.80 | 0.00 | 0.00 | 3 |
| 33 | Straight | `T16_to_T17` | — | 5454.80 | 5536.94 | 82.14 | 83.04 | 3 |
| 34 | Corner | `T17` | Club | 5536.94 | 5536.94 | 0.00 | 0.00 | 3 |
| 35 | Straight | `T17_to_T18` | — | 5536.94 | 5684.40 | 147.47 | 135.53 | 3 |
| 36 | Corner | `T18` | Club exit | 5684.40 | 5684.40 | 0.00 | 0.00 | 3 |

## Sector-boundary uncertainty

Derived by interpolating each reference lap's `SectorNSessionTime` onto that lap's
`Distance` channel. Per-lap values in `GA/sector_boundaries.csv`, summary in
`GA/sector_boundary_summary.csv`. Over 19 reference laps:

| Boundary | Absolute (m) | Lap fraction |
|---|---|---|
| End of S1 | 1804.99 – 1822.26 (mean 1811.75, range **17.27**) | 0.31030 – 0.31276 (mean 0.31139, range 0.00246) |
| End of S2 | 4218.50 – 4251.93 (mean 4237.44, range **33.43**) | 0.72729 – 0.72935 (mean 0.72829, range 0.00206) |
| End of S3 (lap) | 5799.12 – 5837.72 (mean 5818.50, range **38.60**) | 1.00000 – 1.00009 (mean 1.00003, range 0.00009) |

**Absolute boundaries are not stable and are published as intervals, not point
estimates**, because total racing-line length itself varies 38.60 m between drivers.

Lap fractions are more stable — but **not uniformly so**, and the margin is stated
honestly rather than oversold:

| Boundary | Absolute spread (range/mean) | Fractional spread | Fraction tighter by |
|---|---|---|---|
| End of S1 | 0.953% | 0.790% | **1.2x** |
| End of S2 | 0.789% | 0.283% | 2.8x |
| End of S3 (lap) | 0.663% | 0.009% | 71.5x |

The lap boundary is essentially exact in fractional terms and S2 is materially
tighter, but **S1 is only marginally tighter and carries real uncertainty either
way** — treat it as uncertain to ~0.25% of a lap. Use lap fraction, not absolute
metres.

`sector_start`/`sector_end` are assigned from the mean fraction boundaries (0.31139,
0.72829). **That assignment is verified robust, not assumed robust:** validation check
`sector_assignment_robust_to_boundary_uncertainty` confirms the nearest segment
endpoint to the S1 boundary is 139.9 m away against a 14.4 m boundary range — a
**9.7x margin** (S2: 119.7 m against 12.0 m, 10.0x).

**3 straights cross a sector boundary** (segments 1, 11, 29). That is why sector is
published as a `sector_start`/`sector_end` pair rather than a scalar, and why the
existing scalar column is republished unchanged as `legacy_sector` rather than
redefined.

## Two documented upstream defects — recorded, not repaired

Both are reproduced verbatim in the `legacy_*` columns and corrected only in the new
columns. `data/silverstone_2024_track_segments.csv` is byte-unchanged.

**1. Segment 1's `length` is `0.0`; its true extent is 615.93 m of arc (577.92 m of
chord)** — the longest straight on the lap. `src/f1_rl_safety/track.py:87` computes the
wrap-around straight's length as `total_len - corners.iloc[-1]["s_coord"]`. Since
`total_len` *is* `s_coords[-1]`, this expression is identically zero for any circuit.
The segment's coordinates are correct — it genuinely runs Turn 18 → Turn 1; only
`length` is wrong.

**2. The existing scalar `sector` under-assigns sector 1.** `track.py:90-91` applies
the proportion rule to the *corner*, not to the straight preceding it, and Turn 1's
cumulative distance is 0, so segment 1 is labelled sector 1 despite beginning in
sector 3. The last corner's raw index is `1 + 3×1.0 = 4`, clipped down to 3.

**Neither defect is behavioural.** `length` and `sector` are read by **no code** in
this repository — verified by grep across `src/`, `scripts/` and `notebooks/`. Only
`id`, `segment_type`, `corner_number`, `corner_name` and `approx_radius` are consumed.
See `docs/phase-05-segment-consumer-inventory.md`.

## Assumed external naming

`region_name_external` is populated on 21 of 36 rows and NULL on 15. FastF1's
`corners` table has **no `Name` column** — its columns are exactly `X`, `Y`, `Number`,
`Letter`, `Angle`, `Distance` — and the `corner_name` values in the existing CSV
(`"Turn 1"`…`"Turn 18"`) are `track.py:121-123` fallback strings, not source data.
These names are therefore **not derivable from the dataset**.

They are corroborated indirectly by the derived arc lengths: the two longest
straights on the lap fall exactly between the corner pairs that published sources
place at either end of the two named straights.

| Segment | region_key | Assumed name | Derived arc (m) | Rank of 18 |
|---|---|---|---|---|
| 29 | `T14_to_T15` | Hangar Straight | 866.44 | 1 |
| 11 | `T5_to_T6` | Wellington Straight | 727.54 | 2 |
| 1 | `T18_to_T1` | start/finish straight | 615.93 | 3 |

Additional consistency: segments 21–27 (86.65, 141.26, 144.67, 146.29 m) form the
only run of four short straights on the lap, matching the Maggotts/Becketts/Chapel
complex, and their corners carry the three shallowest `Angle` magnitudes (−0.75,
6.13, −11.18). Turn 1 at 462.64 m after the start/finish line matches Abbey's
published position.

**This is corroboration, not derivation. The names remain `assumed`.** An arbitrary
assignment would not have reproduced both longest-straight positions and the esses
run, but that falls short of observation.

`corner_angle_raw` carries `corners.Angle` verbatim. Values are observed and
null-free, but the **unit and reference direction are undocumented**, and two values
look like angle-wrap artefacts (T18 = −178.57, T10 = −159.33, both real corners). It
is carried for future work and **consumed by nothing**.

## Assumption register

| # | Element | Status | Basis | What would falsify it |
|---|---|---|---|---|
| A1 | `X`/`Y`/`Z` = decimetres | derived, 0.1664% from 10.0 | 19-lap fit, 9.994137–10.053723 units/m | A materially different or non-constant scale on other laps, drivers, sessions or circuits |
| A2 | `corners.Distance` in metres, telemetry origin | observed | 12/18 exact, max diff 5.04 m < one 7.4 m sample interval | Any corner diverging by more than one sample interval |
| A3 | Lap length 5837.69 m | observed, this lap | Telemetry `Distance` span | Materially different span on other laps. **Differs from the published circuit length; it is a racing-line distance and must not be presented as the official length** |
| A4 | 18 straight arcs sum to lap length | derived, verified | Residual 0.000000 m | Non-closure |
| A5 | Sector boundaries | derived, **bounded over 19 laps** | Intervals above; fraction preferred to metres | Boundaries outside the published intervals on other laps or sessions |
| A6 | `region_key` | observed | `corners.Number`/`Letter` only | Nothing external; it is a relabelling of observed data |
| A7 | `region_name_external` | **assumed, external** | Published naming; FastF1 supplies no `Name` column | A source disagreeing with any name |
| A8 | `corners.Angle` semantics | **assumed, unresolved** | Values observed; unit and reference undocumented; two apparent wrap artefacts | Establishing the definition from FastF1 source. Until then: carried, never consumed |
| A9 | `approx_radius` is not a physical radius | observed defect | Mean chord to neighbours; 109.3–639.6 m; 0.015%–0.086% of corner base hazard; larger in fast sections | Nothing — a property of the formula |
| A10 | Segment 1 `length = 0.0` | observed defect | `track.py:87`; true arc 615.93 m. Consumed by no code | Nothing — reproducible by inspection |
| A11 | Scalar `sector` under-assigns S1 | observed defect | `track.py:90-91`. Consumed by no code | Nothing |
| A12 | Arc and chord measure different paths | observed | `Distance` follows the racing line; `X`/`Y` chords are centreline. **4 of 18 chords exceed their own arc** (segments 3, 21, 23, 33) — impossible on one path | Nothing. Means the two must never be mixed in one calculation |

## What this geometry does NOT represent

- **elevation** — telemetry `Z` exists but is not incorporated; its unit is only
  inferred by analogy with `X`/`Y` and is untested here
- **camber and banking**
- **track width**
- **kerb, run-off and barrier geometry**
- **surface state, grip or friction model**
- **racing-line model**
- **corner entry/exit or apex speeds**
- **fitted corner radii** — the legacy `approx_radius` is a chord proxy, not a radius
- **any vehicle-dynamics validation**

## Compatibility verdict

**Preserved. No migration plan required.** Full analysis in
`docs/phase-05-segment-consumer-inventory.md`.

The schema is additive-only and lives in a new file. No existing column is renamed,
retyped or repurposed; no key is added to or removed from the environment `info` dict
or the crash log; `segment_type` values are unchanged; `n_segments` remains 36. The
frozen pace-diagnostics evidence at
`outputs/phase2-recalibration/pace_diagnostics/20260827T124054/`, whose aggregation
asserts exactly 36 segments per lap, is unaffected — re-verified by running the
documented reduced-scope check, which reported `n_segments_per_lap = 36`.

## Validation

All 12 checks in `GA/validation.csv` PASS:

| Check | Result |
|---|---|
| `row_count_is_36` | PASS — 36 rows |
| `segment_ids_are_1_to_36` | PASS — min 1, max 36, 36 unique |
| `alternating_18_straight_18_corner` | PASS — `{'Straight': 18, 'Corner': 18}` |
| `segment_type_matches_existing_csv` | PASS — 36 rows joined, all types equal |
| `every_segment_maps_to_exactly_one_region` | PASS — 36 distinct `region_key` over 36 rows |
| `straight_arcs_sum_to_lap_length` | PASS — residual 0.000000 m |
| `all_corner_arc_lengths_zero` | PASS — 18 corners at 0.0 |
| `corner_numbering_matches_source` | PASS — 18 vs 18, equal |
| `corner_numbers_null_on_straights` | PASS — all 18 straights NULL |
| `region_name_external_is_nullable_and_labelled_assumed` | PASS — 21 named, 15 NULL by design |
| `sector_assignment_robust_to_boundary_uncertainty` | PASS — tightest margin 9.7x |
| `segment1_legacy_defect_reproduced_and_corrected_separately` | PASS — defect verbatim, correction separate |

## What this foundation supports, and what it does not license

**Supports:** locating any segment along the lap in metres and lap fraction; grouping
segments into named regions for reporting and plots; region-level aggregation of
existing crash and pace diagnostics; a documented, unit-resolved coordinate basis for
future geometry work; and an auditable record of what the current `approx_radius` and
`length` values actually are.

**Does not license:** any claim of physical or digital-twin fidelity; any
segment-weighted or geometry-derived change to hazard, tyre or pit dynamics; treating
`arc_length_m` as a centreline distance or `chord_length_m` as a racing-line distance;
treating `corner_angle_raw` as a known quantity; presenting 5837.69 m as Silverstone's
official length; or presenting `region_name_external` as observed data.

## Commands run

```
.venv_f1/bin/python -m compileall -q src scripts
.venv_f1/bin/python scripts/build_geometry_schema.py --label smoke_test --smoke
.venv_f1/bin/python scripts/build_geometry_schema.py --label 20260830T112139
.venv_f1/bin/python scripts/analyse_pace_profiles.py --episodes-fixed 3 \
    --episodes-learned 2 --bootstrap 100 --label phase05_regression_check
git diff --check
git status --short
```

The smoke-test and regression-check output directories were both removed after use.

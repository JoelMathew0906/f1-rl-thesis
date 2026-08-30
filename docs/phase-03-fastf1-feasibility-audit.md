# Phase 03 — FastF1 feasibility audit for the 2024 British Grand Prix

Audit-only deliverable for `docs/phase-03-plan.md`. No environment, reward, hazard,
tyre-calibration, training, model or evaluator code was changed to produce this
document. `AA/` below is shorthand for the curated audit-artefact directory
`outputs/phase-03-fastf1-audit/20260830T092331/`.

## Objective

Establish, with reproducible evidence, whether FastF1 can reliably and repeatably
supply the data this project needs for Phase 04 (wet-session calibration) and
Phase 05 (circuit-geometry / digital-twin work), and record precisely which fields
are available, which are derived, and which are absent or unreliable.

## Environment provenance

Interpreter: `.venv_f1` (Python 3.13.5). Installed versions match `requirements.txt`
pins exactly; no drift, no installation performed.

| Package | Installed | Pinned in `requirements.txt` |
|---|---|---|
| fastf1 | 3.8.3 | 3.8.3 |
| pandas | 2.3.3 | 2.3.3 |
| numpy | 2.4.6 | 2.4.6 |

Commands run:
```
.venv_f1/bin/python --version
.venv_f1/bin/python -c "import fastf1, pandas, numpy; print(...)"
grep -E "^(fastf1|pandas|numpy)==" requirements.txt
```

Network access was confirmed available in this session: `fastf1.get_event_schedule(2023)`
(a year not present in the pre-existing warm cache) completed in 0.4 s with 23 rows,
demonstrating a genuine network fetch rather than a cache hit.

## Repository state at audit time

- Branch: `phase2-recalibration`; working tree clean.
- HEAD `4874384` (`docs: add phased FastF1, wet calibration and geometry handoffs`),
  confirmed a descendant of the freeze commit `e5f0d9e` via
  `git merge-base --is-ancestor e5f0d9e HEAD`.
- `src/f1_rl_safety/track.py` and `src/f1_rl_safety/data_loader.py` read in full; both
  use `fastf1.Cache.enable_cache("data/cache")`, `fastf1.get_session(year, "Silverstone",
  "R")` and `session.load()`. `track.py` additionally calls
  `session.get_circuit_info()` and reads `circuit_info.corners` columns `X`, `Y`,
  `Number`, `Letter` (and `Name` if present).
- `F1RaceEnv._load_calibration_2024` (`src/f1_rl_safety/f1_env.py`) reads
  `data/silverstone_2024_laps.csv`, consumes `LapTime`, `Compound`, `TyreLife`,
  `Stint`, and derives `base_lap_time` (median dry-compound lap time), per-compound
  `compound_offsets`, `deg_per_lap` (linear fit against `TyreLife`), `typical_stint`,
  and the fixed `pit_loss = 21.5` s. It falls back to hardcoded constants only if the
  CSV is missing.
- Pre-existing on-disk cache before this audit: `data/cache/2024/2024-07-07_British_Grand_Prix/2024-07-07_Race/`
  (11 `.ff1pkl` groups) plus `fastf1_http_cache.sqlite`, ~99 MB total. `data/fastf1_cache`
  exists (~99 MB, an older/duplicate snapshot) but is referenced by no code and was left
  untouched by this audit (`du -sh` unchanged before/after).
- `.gitignore` already excludes `data/cache/`, `data/fastf1_cache/` and (via
  `outputs/**/{models,logs,trainlogs,eval,diagnostics}/`, `outputs/**/DONE`,
  `outputs/**/*_pace_profiles.csv`) the bulky Phase 2 study outputs. No existing
  pattern excludes `outputs/phase-03-fastf1-audit/`, so this audit's curated,
  small (28 KB total) CSV/README outputs are trackable as-is; no new `.gitignore`
  pattern was needed.

## Session discovery

`fastf1.get_event_schedule(2024)`, filtered to `EventName` containing "British",
returns exactly one 2024 British Grand Prix event (`OfficialEventName`: "FORMULA 1
QATAR AIRWAYS BRITISH GRAND PRIX 2024", round 12, `EventFormat = conventional`,
`EventDate = 2024-07-07`). It reports five sessions, all present as named
`SessionN` / `SessionNDate` / `SessionNDateUtc` fields — no session codes were
assumed in advance:

| # | Name | Local date/time | UTC date/time |
|---|---|---|---|
| 1 | Practice 1 | 2024-07-05 12:30 +01:00 | 2024-07-05 11:30 |
| 2 | Practice 2 | 2024-07-05 16:00 +01:00 | 2024-07-05 15:00 |
| 3 | Practice 3 | 2024-07-06 11:30 +01:00 | 2024-07-06 10:30 |
| 4 | Qualifying | 2024-07-06 15:00 +01:00 | 2024-07-06 14:00 |
| 5 | Race | 2024-07-07 15:00 +01:00 | 2024-07-07 14:00 |

Conventional format — no Sprint session exists for this event, so there is nothing
untested in that category; it is simply absent from the schedule.

## Load matrix

All five sessions were loaded with `fastf1.get_session(2024, "Silverstone", CODE)` /
`session.load()` for `CODE` in `{FP1, FP2, FP3, Q, R}`. Full results in `AA/manifest.csv`.

| Session | Load result | Wall-clock | Cache growth | Laps rows | Drivers |
|---|---|---|---|---|---|
| FP1 | success | 0.51 s | 0 B (already cached) | 459 | 20 |
| FP2 | success | 4.66 s | 67.4 MB (fetched) | 477 | 20 |
| FP3 | success | 4.59 s | 50.4 MB (fetched) | 438 | 20 |
| Q | success | 5.02 s | 53.7 MB (fetched) | 368 | 20 |
| R | success | 0.78 s | 0 B (already cached) | 960 | 19 |

5/5 sessions loaded successfully with no untested or unavailable sessions. FP1 and R
were already warm from a pre-existing cache and from earlier steps of this same
audit session; FP2, FP3 and Q were cold-fetched from the network and are now warm
for any future run in this environment.

## Per-group field audit

Full per-(session, data group, column) dtype and null-count table in
`AA/field_audit.csv` (52 rows for FP1/R alone in the initial smoke check; 256 rows in
the full five-session run). Summary by group, observed once across all five sessions
(column sets were identical across sessions in every group):

**Lap timing / tyre / stint (`session.laps`)** — 31 columns. Columns the simulator's
calibration path depends on: `LapTime` (`timedelta64[ns]`, 0 nulls across all
sessions), `Compound` (`object`, 0 nulls), `TyreLife` (`float64`, 0 nulls), `Stint`
(`float64`, 0 nulls), `IsAccurate` (`bool`, 0 nulls). Additionally present and
audited: `PitInTime`/`PitOutTime` (`timedelta64[ns]`, null wherever a lap is not a
pit lap — e.g. 914/960 null in the Race — non-null exactly on in/out laps),
`TrackStatus` (`object`, per-lap string of status codes active during that lap, 0
nulls). Compounds observed varied by session and by track condition: FP1
`HARD,MEDIUM,SOFT`; FP2 `HARD,INTERMEDIATE,MEDIUM,SOFT`; FP3
`INTERMEDIATE,None` (drivers ran almost exclusively intermediates in a wet session,
with `None` appearing for installation/outlaps); Q `INTERMEDIATE,SOFT`; R
`HARD,INTERMEDIATE,MEDIUM,SOFT`.

**Weather (`session.weather_data`)** — 8 columns: `Time`, `AirTemp`, `Humidity`,
`Pressure`, `Rainfall`, `TrackTemp`, `WindDirection`, `WindSpeed`. No nulls observed
in any column in any session. `Rainfall` is boolean.

**Track status (`session.track_status`)** — 3 columns: `Time`, `Status`, `Message`.
Row counts (i.e. number of status-change events) ranged from 1 (FP2, Q — clean
sessions) to 8 (FP1).

**Circuit information (`session.get_circuit_info().corners`)** — 6 columns: `X`,
`Y`, `Number`, `Letter`, `Angle`, `Distance`. 18 rows (18 corners) in every session,
0 nulls. This matches the 18 `Straight` + 18 `Corner` = 36-segment track model
already built by `track.py`/`load_or_build_silverstone_segments`. Note:
`_build_silverstone_track_segments_from_fastf1` currently reads only `X`, `Y`,
`Number`, `Letter` (and `Name` if present) — it does not read the also-available
`Angle` or `Distance` columns; those are additional geometry inputs available for
Phase 05 but not yet consumed.

**Telemetry (`session.laps.pick_fastest().get_telemetry()`)** — 18 columns:
`Date`, `SessionTime`, `DriverAhead`, `DistanceToDriverAhead`, `Time`, `RPM`,
`Speed`, `nGear`, `Throttle`, `Brake`, `DRS`, `Source`, `Distance`,
`RelativeDistance`, `Status`, `X`, `Y`, `Z`. Median sample interval on the fastest
lap of each session ranged 0.125–0.14 s (≈7.1–8.0 Hz); this channel is the
car-data/position merge FastF1 produces per lap, not raw position data.

**Raw position data (`session.pos_data`)** — per-driver dict, one DataFrame per
driver keyed by driver number; 20 drivers present in every session (Race laps report
19 distinct `Driver` values in `session.laps`, but `pos_data` itself still has 20
keys). Columns: `Date`, `Status`, `X`, `Y`, `Z`, `Source`, `Time`, `SessionTime`. No
nulls in the columns sampled. Median raw sample interval measured directly on driver
44 in the Race: 0.241 s (≈4.15 Hz) — markedly coarser than the merged telemetry rate
above.

## Weather and condition evidence

Full per-session weather summary in `AA/weather_summary.csv`. `Rainfall` is a usable,
directly observed boolean condition indicator, corroborated by `AirTemp`/`TrackTemp`
swings:

| Session | Samples | Rainfall True / False | Any rain | AirTemp range | TrackTemp range |
|---|---|---|---|---|---|
| FP1 | 81 | 0 / 81 | No | 15.6–17.2 °C | 21.2–28.1 °C |
| FP2 | 81 | 14 / 67 | Yes | 17.5–18.9 °C | 25.0–32.7 °C |
| FP3 | 82 | 54 / 28 | Yes (majority wet) | 10.7–11.6 °C | 15.7–17.7 °C |
| Q | 85 | 6 / 79 | Yes (brief) | 12.4–14.6 °C | 18.3–24.0 °C |
| R | 147 | 51 / 96 | Yes (sustained) | 14.5–16.8 °C | 20.7–37.9 °C |

This is consistent with the well-documented outcome of the actual 2024 British
Grand Prix weekend (a wet FP3, a mixed race). For Phase 04, a session or
sub-window could be empirically classified wet/dry/mixed directly from the
`Rainfall` boolean time series (e.g. fraction-of-samples-True over a rolling
window), cross-checked against the compound choices already visible in
`session.laps["Compound"]` (`INTERMEDIATE`/`WET` usage) and against `TrackTemp`
drops. No session's wetness is asserted here beyond what these two independently
observed fields show.

## Geometry readiness

Available for Phase 05:
- `circuit_info.corners`: 18 corners, columns `X`, `Y`, `Number`, `Letter`, `Angle`,
  `Distance` — currently only `X`, `Y`, `Number`, `Letter` are consumed by
  `track.py`; `Angle` and `Distance` are additional, currently-unused geometry
  inputs.
- Telemetry `Distance`/`RelativeDistance`: `Distance` on the session's fastest lap
  reaches a maximum of 5837.7 m, consistent with Silverstone's known ~5.891 km lap
  length — this channel is in metres.
- Telemetry and raw position `X`/`Y`/`Z`: span roughly ±2,300 to ±13,000 in the same
  arbitrary units, and are **not** confirmed to be metres — the coordinate range is
  far larger than the ~5.8 km circuit length would suggest if the units were metres,
  and FastF1's public documentation does not state a unit for this field. This is
  recorded as an **assumption/open question**, not a fact: any geometry work must
  independently establish the coordinate unit and scale (e.g. by fitting `X`,`Y`
  arc length against the metre-denominated `Distance` channel) before treating `X`,
  `Y` as physical distances.
- Sampling rate: merged telemetry ≈7.1–8.0 Hz (fastest lap, all sessions); raw
  `pos_data` ≈4.15 Hz (measured on Race, driver 44).
- Minor, quantified data-completeness artefact: FastF1 logs
  `"Driver N: Position data is incomplete!"` for roughly half the drivers in FP1.
  Direct inspection shows this refers to `fastf1.api.position_data`'s zero-fill/
  `Status="OffTrack"` padding used to align all drivers' position traces to the same
  length. Measured directly: at most 4 padded rows out of ~18,522 per affected
  driver (~0.02%), not a material gap. Practical implication for Phase 05: filter
  `Status != "OffTrack"` before treating `X`/`Y`/`Z` as valid, but do not expect
  large gaps in this event.

## Cache behaviour

- Cold loads (FP2, FP3, Q — none previously cached): 4.6–5.0 s each; wrote
  50–67 MB per session under `data/cache/2024/2024-07-07_British_Grand_Prix/`.
- Warm loads (FP1, R — pre-existing or already-fetched-this-session): 0.5–0.8 s,
  zero cache growth.
- Reproducibility check (`AA/reproducibility_check.csv`): the Race session was
  loaded twice in immediate succession against the warm cache. Both loads returned
  identical shapes (960 lap rows, 147 weather rows) and near-identical timing
  (0.782 s, then 0.778 s) — the load is reproducible from a warm cache.
- As an additional, non-destructive reproducibility check, `data/silverstone_2024_laps.csv`
  (the file the simulator's calibration reads) was compared cell-by-cell against a
  fresh call to the existing `extract_stint_and_lap_data(2024)` function in
  `src/f1_rl_safety/data_loader.py`: identical shape (850 rows × 12 columns) and
  **zero differing cells** across all 10,200 compared values. The existing
  calibration input is exactly reproducible from FastF1 via the code already in the
  repository.
- Total `data/cache` size after this audit: ~332 MB (up from ~99 MB). `data/fastf1_cache`
  (the unused future-intent directory) was left untouched — verified unchanged in
  size and via `find -newer`. No cache deletion, invalidation, move or cold-cache
  simulation was performed.
- Audit output directory size: 28 KB total across `manifest.csv`, `field_audit.csv`,
  `weather_summary.csv`, `reproducibility_check.csv`, `README.md` — no bulk or
  record-level data committed.

## Comparison table — FastF1-derived candidates vs existing simulator calibration

| Item | Existing simulator value | FastF1-derived candidate | Label |
|---|---|---|---|
| Calibration source rows | `data/silverstone_2024_laps.csv`, 850 rows | Re-extracted via `extract_stint_and_lap_data(2024)`, 850 rows, 0 differing cells | **observed** — exact match confirms current calibration input is reproducible from FastF1 today |
| `base_lap_time`, `compound_offsets`, `deg_per_lap`, `typical_stint` | Computed at runtime by `_load_calibration_2024` from the CSV above | Would be numerically identical if recomputed, since the source rows match exactly | **derived** — not recomputed separately in this audit; reproducibility of the *input* is what was newly confirmed here |
| Track segment count | 36 (18 straights + 18 corners), from `data/silverstone_2024_track_segments.csv` | `circuit_info.corners` returns 18 corner rows in every one of the five audited sessions | **observed** — count matches across all sessions |
| Corner geometry fields consumed | `X`, `Y`, `Number`, `Letter` | Also available and unused: `Angle`, `Distance` | **observed** (available), **non-comparable** to current sim (nothing to compare against — not yet consumed) |
| `pit_loss = 21.5` s (fixed constant) | Simulator constant | `PitInTime`/`PitOutTime` columns are present and non-null on in/out laps for every session, so a FastF1-measured pit-lane loss is in principle derivable | **non-comparable** in this audit — deriving and comparing an actual measured pit loss is new calibration analysis, out of scope for an audit-only phase; flagged as available future evidence for Phase 04, not computed here |
| Wet/dry session identity | Not modelled in the simulator (dry-only compound calibration) | `Rainfall` boolean + compound choice + `TrackTemp` give an empirically groundable classification | **observed** (the raw signal), **assumption** if used to label a whole session as a single condition rather than a time-varying one |
| Track coordinate units (`X`,`Y`,`Z`) | Not used as physical units anywhere in the current simulator (segments use ordering and relative distance only) | Present in telemetry and `pos_data`, unit unconfirmed | **assumption** — do not treat as metres without independent verification |
| SOFT-tyre durability anomaly (documented Phase 2 limitation) | SOFT offset −1.270 s, `deg_per_lap` 0.044; MEDIUM +0.264 s, 0.202 | Same source data (`data/silverstone_2024_laps.csv`) confirmed exactly reproducible in this audit | **non-comparable** — this audit does not re-derive or re-litigate that calibration; it only confirms the underlying data used to produce it is itself stable |

## Limitations

- This audit exercised exactly one event (2024 British Grand Prix) and its five
  conventional-format sessions. No other circuit, season, or Sprint-format event was
  tested; FastF1's general reliability for other events is not established here.
- Position-data coordinate units (`X`, `Y`, `Z`) were not conclusively identified;
  this is recorded as an open assumption, not resolved.
- The pit-loss constant (`pit_loss = 21.5` s) was not independently re-derived from
  FastF1's `PitInTime`/`PitOutTime` columns in this audit; that column pair is
  confirmed present and populated, but computing and comparing an actual measured
  value is new analysis, deliberately left out of an audit-only phase.
- Weather/condition classification (wet vs dry vs mixed) is demonstrated as
  computable from raw fields; no classification scheme was chosen or validated here,
  and none of the five sessions is asserted here as canonically "the wet session" —
  that decision belongs to Phase 04.
- The FP1 "Position data is incomplete" warning was investigated and quantified
  (≤4 padded rows per affected driver, ~0.02%) for this event only; whether the same
  magnitude of padding holds for other events was not tested.
- Telemetry and position sampling-rate figures (≈7.1–8.0 Hz and ≈4.15 Hz
  respectively) were each measured from a single representative sample (the fastest
  lap per session; driver 44 in the Race) rather than across all drivers and laps.
- Network reliability over a longer time horizon (e.g. whether FastF1's remote data
  source remains available months or years from now) cannot be established from a
  single session's testing.

## Recommendation

**Proceed.**

FastF1 loaded all five discovered sessions for the target event successfully, with
no untested or unavailable sessions. Every data group in scope (lap timing, tyre
compounds, stint structure, weather/rainfall, track status, circuit information,
telemetry and position data) was present with actual observed column names, no
material null gaps, and quantifiably minor completeness artefacts. Warm-cache
reproducibility was demonstrated directly, and — more strongly — the exact
calibration input file the simulator already depends on was shown to be
byte-for-cell reproducible from FastF1 via existing repository code. The one open
item (unconfirmed coordinate units for `X`/`Y`/`Z`) does not block Phase 04 (which
depends on weather/compound/lap-timing fields, all confirmed clean) and only
constrains Phase 05 to first establishing that unit before treating those fields as
physical distances.

## Required commands run (for reproducibility)

```
.venv_f1/bin/python --version
.venv_f1/bin/python -c "import fastf1, pandas, numpy; print(...)"
.venv_f1/bin/python -m compileall -q src scripts
.venv_f1/bin/python scripts/audit_fastf1_availability.py --label 20260830T092331
```

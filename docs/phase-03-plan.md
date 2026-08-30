# Phase 03 — FastF1 feasibility audit

Execution-ready handoff prompt for a fresh Claude Code session. Read this file in full
before acting. Do not skip the inspection stage.

## Baseline and branch guardrails

- Work begins from `phase2-recalibration` **only**. Confirm with `git branch --show-current`
  before anything else.
- `main` and `baseline/pre-recalibration` are protected historical branches. Do not modify,
  merge, rebase, force-push or commit onto them. Verify their current refs before relying on
  a commit relationship.
- This phase is an **audit**. It is expected to remain documentation- and
  analysis-artefact-only, so it may stay on `phase2-recalibration`. Create a fresh feature
  branch from `phase2-recalibration` **only if** actual implementation changes begin — and
  only after proposing that step and obtaining approval.
- Enter Plan Mode and inspect the repository before making any edit.
- Keep scope minimal and evidence-driven. Do not invent telemetry fields, parameters,
  empirical claims or FastF1 API capabilities. If a capability is uncertain, test it and
  record the observed result, or record it as unavailable.
- The reward-regime and pace–survival evidence frozen at commit `e5f0d9e` stays frozen. Do
  not retroactively rewrite it to fit FastF1 findings. New findings go in new artefacts.
- Obtain explicit approval before changing environment dynamics, reward logic, hazard logic,
  tyre parameters, training configuration, evaluator semantics, or any existing conclusion.
- Do not push until the user has explicitly approved the phase's final diff and commit.

## Objective

Establish, with reproducible evidence, whether FastF1 can reliably and repeatably supply the
data this project would need for later wet-session calibration (Phase 04) and circuit
geometry work (Phase 05) — and record precisely which fields are available, which are
derived, and which are absent or unreliable. The deliverable is a decision, not a code
change.

## Research question

Can FastF1 access and cache the relevant 2024 British Grand Prix sessions reproducibly, and
which of the data required for wet-condition calibration and geometry-aware simulation is
genuinely available, at what granularity, and with what limitations?

## Inputs and evidence already frozen

Treat the following as established context, verified at the freeze commit `e5f0d9e`. Verify
paths still exist before relying on them; do not change them in this phase.

- Cache paths: code uses `data/cache` (`CACHE_DIR` in both `src/f1_rl_safety/data_loader.py`
  and `src/f1_rl_safety/track.py`, each calling `fastf1.Cache.enable_cache`).
  `data/fastf1_cache` exists on disk but is referenced by **no** code. Both are gitignored.
- An existing on-disk cache for the 2024 British Grand Prix **Race** session is present at
  `data/cache/2024/2024-07-07_British_Grand_Prix/2024-07-07_Race/`. The cache groups
  observed there are: `_extended_timing_data`, `car_data`, `driver_info`, `lap_count`,
  `position_data`, `race_control_messages`, `session_info`, `session_status_data`,
  `timing_app_data`, `track_status_data`, `weather_data`. Use this as evidence of what has
  previously been retrievable — not as proof of current availability.
- Existing FastF1 usage in the repository, and the only API surface currently relied upon:
  `fastf1.Cache.enable_cache(path)`; `fastf1.get_session(year, "Silverstone", "R")`;
  `session.load()`; `session.get_circuit_info()` and `circuit_info.corners` with columns
  `X`, `Y`, `Number`, `Letter` and optionally `Name` (see
  `_build_silverstone_track_segments_from_fastf1` in `src/f1_rl_safety/track.py`).
- Lap-data columns currently consumed by `F1RaceEnv._load_calibration_2024` from
  `data/silverstone_2024_laps.csv`: `LapTime`, `Compound`, `TyreLife`, `Stint`, `Driver`.
- Derived calibration currently in force: `base_lap_time`, per-compound `compound_offsets`,
  `deg_per_lap`, `typical_stint`, and a fixed `pit_loss` of 21.5 s.
- Compound indices used throughout: SOFT 0, MEDIUM 1, HARD 2, INTERMEDIATE 3, WET 4.
- Known limitation to keep in view: calibrated SOFT is both quicker (offset −1.270 s) and
  far more durable (0.044 s per lap of tyre age) than MEDIUM (+0.264 s; 0.202 s). This is a
  documented artefact, not something to fix in this phase.
- Interpreter `.venv_f1` (Python 3.13) with `fastf1==3.8.3` pinned in `requirements.txt`.

## Scope

1. Inspect the installed Python environment and the project's existing dependency
   conventions **before** proposing any installation or dependency change.
2. Determine whether the relevant 2024 British Grand Prix sessions can be located, loaded
   and cached reproducibly. Cover each session identifier the audit finds to exist for that
   event, and record which load successfully and which do not.
3. Audit — do not modify — the availability and shape of:
   - lap timing,
   - tyre compounds,
   - stint structure,
   - weather and rainfall indicators,
   - track status,
   - circuit information,
   - telemetry and position data relevant to later geometry work.
4. For each data group, record: how it is accessed, the observed row/column shape, the actual
   column names present, null/missing behaviour, and any driver- or session-specific gaps.
5. Compare only defensible FastF1-derived candidate measures against the existing simulator
   calibration, keeping four categories strictly separated:
   **observed** values, **derived** values, **assumptions**, and **non-comparable**
   quantities.
6. Characterise cache behaviour non-destructively: warm-cache load behaviour, what is
   written where, whether a second run is reproducible, and approximate sizes and timings.
7. Produce a committed, reviewable audit artefact and a recommendation.

## Explicit non-goals

- Do not alter calibration or simulator code. Any such change requires a separate, approved
  proposal.
- Do not modify `_load_calibration_2024`, `f1_env.py`, `track.py`, wrappers, the evaluator,
  training entrypoints, reward YAML, or existing models and outputs.
- Do not perform wet-session calibration (Phase 04) or geometry redesign (Phase 05).
- Do not retrain any model.
- Do not change the FastF1 cache path from `data/cache` in this phase, even though
  `data/fastf1_cache` is the documented future intent; migrating it is a separate,
  approved change.
- Do not commit bulky raw FastF1 dumps. Cache directories are gitignored and stay local.
- Do not assert real-world physical validity, regulatory equivalence, or predictive
  calibration from any field discovered here.

## Preconditions and repository inspection

Complete all of the following and report findings before the first edit.

1. `git branch --show-current` → must be `phase2-recalibration`; `git status --short` → clean.
2. Confirm `HEAD` is at or descends from the freeze commit `e5f0d9e`.
3. Inspect the interpreter and installed versions without changing them, for example the
   installed `fastf1`, `pandas` and `numpy` versions in `.venv_f1`, and compare against
   `requirements.txt` pins. Record any drift.
4. Read `src/f1_rl_safety/track.py` and `src/f1_rl_safety/data_loader.py` in full to confirm
   the current cache configuration and the exact FastF1 calls in use.
5. Read `F1RaceEnv._load_calibration_2024` in `src/f1_rl_safety/f1_env.py` to confirm which
   lap-data columns and derived quantities the simulator actually depends on.
6. List what is already cached on disk under `data/cache` and `data/fastf1_cache`, with
   sizes, and note that the Race session appears already cached.
7. Confirm whether network access is available in the session. If it is not, the audit must
   proceed against the existing warm cache only and must say so explicitly, marking any
   session it could not test as untested rather than unavailable.
8. Check `.gitignore` so the audit artefact is written to a path that will actually be
   trackable.

## Implementation plan

Proposed sequence. Adjust with justification; report deviations.

1. **Environment provenance.** Capture interpreter path, Python version, and the installed
   versions of `fastf1` and its relevant dependencies. Record whether they match
   `requirements.txt`. Propose no installation unless something is genuinely missing, and
   only with approval.
2. **Session discovery.** Determine which sessions exist for the 2024 British Grand Prix
   using FastF1's own event/schedule facilities as they actually behave in this version.
   Record the event name, date and the session identifiers reported. Do not assume a fixed
   set of session codes.
3. **Load matrix.** For each discovered session, attempt a load and record: success or
   failure, error text if any, wall-clock duration, whether the load was served from cache
   or fetched, and what was written to `data/cache`.
4. **Field audit.** For each successfully loaded session, and for each data group in scope,
   record the access expression used, the observed shape, the actual column names, dtypes,
   and missing-value counts for the columns that matter. Keep raw evidence in the artefact
   as small tabular summaries, not as bulk dumps.
5. **Weather and condition evidence.** Audit the weather data specifically for whatever
   rainfall or condition indicator is genuinely present, plus track status, and describe how
   a wet or mixed session could be identified empirically in Phase 04. Do not assert which
   session is wet; record what the data shows.
6. **Geometry readiness.** Audit circuit information and any position or telemetry data
   relevant to Phase 05: coordinate columns, units if discoverable, corner metadata,
   distance or lap-fraction fields, and sampling rate. State clearly what is and is not
   available for building a geometry schema.
7. **Reproducibility check.** Re-run at least one load against the warm cache and confirm
   the audit's recorded values are stable. Record warm-cache timings.

   Do not delete, clear, move, invalidate, or overwrite the existing FastF1 cache to
   simulate a cold load without explicit approval. The default audit must use
   non-destructive inspection and warm-cache reproducibility. If cold-cache evidence is
   required, first propose an isolated temporary cache directory and wait for approval.
8. **Comparison table.** Place defensible FastF1-derived candidate measures beside the
   existing simulator calibration, with every row labelled observed, derived, assumption or
   non-comparable. Do not compute a comparison you cannot defend; mark it non-comparable and
   say why.
9. **Recommendation.** Conclude with exactly one of: **proceed**, **proceed with
   constraints**, or **do not proceed** — with the constraints or blockers enumerated.

## Validation and acceptance criteria

Select checks from existing documented project commands only; do not invent test commands.
Available and appropriate here:

- `.venv_f1/bin/python -m compileall -q src scripts` — must exit 0 if any script is added.
- `git diff --check` — must report no whitespace or conflict-marker errors.
- `git status --short` — must show only the intended new or modified paths.
- If an audit script is added, run it once at reduced scope as a smoke check before the full
  run, and confirm the output schema, exactly as
  `scripts/analyse_pace_profiles.py` was validated in Phase 2.

Acceptance criteria:

1. Every claim in the audit artefact is traceable to a command, a file path or an observed
   value recorded in the artefact itself.
2. Session availability is reported per session, with untested cases distinguished from
   unavailable cases.
3. Every audited data group has its actual observed column names recorded — not assumed ones.
4. Observed, derived, assumption and non-comparable quantities are separated and labelled.
5. Cache behaviour, including warm-cache reproducibility, is documented.
6. Limitations are stated explicitly, including anything the audit could not test.
7. A single unambiguous recommendation is given.
8. No calibration, environment, reward, hazard, training, model or evaluator code changed.

## Required artefacts

- `docs/phase-03-fastf1-feasibility-audit.md` — the reviewable audit report: objective,
  environment provenance, exact commands run, session availability matrix, per-group field
  audit, weather and condition evidence, geometry readiness, cache behaviour, the labelled
  comparison table, limitations, and the recommendation.
- Optionally, if a repeatable probe is warranted: `scripts/audit_fastf1_availability.py`,
  read-only, writing curated summaries under
  `outputs/phase-03-fastf1-audit/<timestamp>/` with a `manifest.csv` and `README.md`
  following the Phase 2 convention. Bulk or record-level CSVs must be gitignored, matching
  the existing `outputs/**/*_pace_profiles.csv` precedent; propose any new ignore pattern
  before adding it.
- Update `## Completion record` in this file.

## Documentation and memory updates

- Update the memory handoff record at
  `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
  with a new dated section for Phase 03, before ending the session. Preserve all existing
  content and the existing dated sections.
- Record in that section: the recommendation, the audit artefact path, the commit SHA once
  approved, what was found available and unavailable, and any constraint that Phase 04 or
  Phase 05 must respect.
- Do not edit the frozen evidence draft
  `docs/drafts/reward_and_environment_recalibration_evidence.md` in this phase.

## Git and handoff procedure

1. Confirm the branch and a clean tree before starting.
2. Make no commit until the audit artefact is complete and the user has reviewed the diff.
3. Propose a single focused commit. Suggested message shape:
   `docs: FastF1 feasibility audit for 2024 British Grand Prix sessions`.
4. Show `git diff --check`, `git status --short`, the full diff stat and the exact contents
   of new or modified Markdown before requesting approval.
5. **Do not push** until the user explicitly approves the final diff and commit.
6. Do not amend history, rebase, merge, switch branches or create additional commits.

## Stop conditions and decisions requiring approval

Stop and ask before proceeding if any of the following arise.

- Any dependency installation, upgrade or `requirements.txt` change appears necessary.
- Network access is unavailable or partially blocked, so sessions cannot be tested.
- A session fails to load and the cause is not clearly attributable from the error.
- The audit indicates that a required field for Phase 04 or Phase 05 is absent or
  unreliable — report the evidence rather than devising a substitute.
- Any temptation arises to modify calibration, environment, reward, hazard, training,
  evaluator or model code, or the frozen conclusions.
- A change to the FastF1 cache path or a new `.gitignore` pattern seems warranted.
- Cold-cache evidence appears necessary. Do not delete, clear, move, invalidate, or
  overwrite the existing FastF1 cache to simulate a cold load without explicit approval. The
  default audit must use non-destructive inspection and warm-cache reproducibility. If
  cold-cache evidence is required, first propose an isolated temporary cache directory and
  wait for approval.
- The artefact would need to exceed a reviewable size, or bulky raw data would need
  committing.

## Completion record

- **Date completed:** 2026-08-30
- **Completed tasks:** All preconditions (branch/HEAD/tree, interpreter and package
  version provenance, reading `track.py`/`data_loader.py`/`_load_calibration_2024`,
  listing `data/cache` and `data/fastf1_cache`, confirming network access,
  checking `.gitignore`). Full audit: session discovery, load matrix for
  FP1/FP2/FP3/Q/R, per-group field audit (laps, weather, track status, circuit
  info, telemetry, raw position data), weather/condition evidence, geometry
  readiness, warm-cache reproducibility check (including an exact-match
  reproducibility check of `data/silverstone_2024_laps.csv` against a fresh
  FastF1 extraction), labelled comparison table, and recommendation.
- **Files changed (with paths):**
  - `docs/phase-03-fastf1-feasibility-audit.md` (new — the audit report)
  - `scripts/audit_fastf1_availability.py` (new — read-only audit script)
  - `docs/phase-03-plan.md` (this completion record)
  - `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
    (new dated Phase 03 section)
  - Non-tracked, gitignored: `outputs/phase-03-fastf1-audit/20260830T092331/`
    (curated `manifest.csv`, `field_audit.csv`, `weather_summary.csv`,
    `reproducibility_check.csv`, `README.md`, 28 KB total) — **not yet committed
    pending confirmation this path should be tracked; see Unresolved issues.**
    `data/cache` grew from ~99 MB to ~332 MB (gitignored, not committed).
- **Checks run and results:** `.venv_f1/bin/python -m compileall -q src scripts` —
  exit 0. `.venv_f1/bin/python scripts/audit_fastf1_availability.py --label
  smoke_test --sessions R` — reduced-scope smoke check, passed, schema verified,
  smoke output deleted before the full run. `.venv_f1/bin/python
  scripts/audit_fastf1_availability.py --label 20260830T092331` — full run, 5/5
  sessions loaded successfully. `git diff --check` and `git status --short` —
  pending final review before commit.
- **Commit SHA:** _pending — not committed; user has not yet approved the diff._
- **Recommendation issued:** proceed
- **Constraints or blockers recorded:** Coordinate units for telemetry/position
  `X`/`Y`/`Z` are unconfirmed (not metres by inspection; genuine unit unresolved).
  Phase 05 must independently establish this unit/scale before treating those
  fields as physical distances. Phase 04 is not blocked by this — its required
  fields (`Rainfall`, `Compound`, `TrackTemp`, lap timing) were all confirmed
  clean with no material nulls.
- **Unresolved issues:** Whether `outputs/phase-03-fastf1-audit/` needs an
  explicit `.gitignore` allow/deny pattern was not raised as a stop condition
  (no existing pattern excludes it, and its content is small and curated), but
  this should be confirmed explicitly before commit. Pit-loss (`pit_loss =
  21.5` s) was not independently re-derived from `PitInTime`/`PitOutTime`; that
  is deliberately left for Phase 04. Position/telemetry sampling-rate figures
  were measured from a single representative lap/driver per session, not
  exhaustively.
- **Next 3–5 concrete tasks:**
  1. Obtain user approval of this diff; commit with message
     `docs: FastF1 feasibility audit for 2024 British Grand Prix sessions`.
  2. Decide whether to commit the curated `outputs/phase-03-fastf1-audit/20260830T092331/`
     directory alongside the audit report (following the Phase 2
     `pace_diagnostics` precedent) or keep it local-only.
  3. Begin Phase 04 (`docs/phase-04-plan.md`) using `Rainfall`, `Compound` and
     `TrackTemp` as the empirical basis for wet/dry/mixed session
     classification, per this audit's weather evidence.
  4. Before any Phase 05 geometry work, independently establish the physical
     unit of the `X`/`Y`/`Z` coordinate fields (e.g. by fitting arc length
     against the metre-denominated `Distance` telemetry channel).
  5. If a measured pit-lane loss is later wanted to corroborate the fixed
     21.5 s constant, derive it from `PitInTime`/`PitOutTime` as a new, separate,
     approved analysis — not retroactively into this audit.

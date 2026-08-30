# Phase 04 — Wet-session calibration

Execution-ready handoff prompt for a fresh Claude Code session. Read this file in full
before acting. This phase is evidence-first: no calibration code changes until the user
approves both the evidence and a proposed diff.

## Baseline and branch guardrails

- Work begins from `phase2-recalibration` **only**. Confirm with `git branch --show-current`
  before anything else.
- `main` and `baseline/pre-recalibration` are protected historical branches. Do not modify,
  merge, rebase, force-push or commit onto them. Both sit at `74c5f11`, as does the tag
  `baseline-pre-recalibration`.
- Evidence gathering and the proposed-diff document are documentation-only and may remain on
  `phase2-recalibration`. Create a fresh feature branch from `phase2-recalibration` **only
  if** actual implementation changes begin, and only after proposing that step and obtaining
  approval.
- Enter Plan Mode and inspect the repository before making any edit.
- Keep scope minimal and evidence-driven. Do not invent telemetry, parameters, empirical
  claims or API capabilities. If the data does not support a parameter, say so.
- The reward-regime and pace–survival evidence frozen at commit `e5f0d9e` stays frozen. Do
  not retroactively rewrite it to fit wet-session findings. Keep dry-regime behaviour and
  the frozen pace–survival evidence strictly separate from any wet-specific finding.
- Obtain explicit approval before changing environment dynamics, reward logic, hazard logic,
  tyre parameters, training configuration, evaluator semantics, or any existing conclusion.
- Do not push until the user has explicitly approved the phase's final diff and commit.

## Objective

Determine whether a defensible wet or mixed-condition calibration can be derived for the
Silverstone simulator from verifiable 2024 data, and if so, produce an evidence artefact
plus a proposed parameter-change diff for approval. A documented decision **not** to
calibrate is an acceptable and complete outcome.

## Research question

Is there a verifiable wet or mixed-condition 2024 Silverstone session from which
INTERMEDIATE and/or WET compound parameters — pace offset and degradation — can be estimated
with stated uncertainty, using the data confirmed available by the Phase 03 audit?

## Inputs and evidence already frozen

- **Precondition:** this phase begins only after Phase 03 has produced a **committed
  feasibility decision**. Read `docs/phase-03-fastf1-feasibility-audit.md` and the Phase 03
  `## Completion record` first, and honour every constraint recorded there.
- Current calibration is produced at runtime by `F1RaceEnv._load_calibration_2024` in
  `src/f1_rl_safety/f1_env.py` from `data/silverstone_2024_laps.csv`. It fits, per compound:
  `compound_offsets` (median lap time minus a dry baseline), `deg_per_lap` (a clipped linear
  slope of lap time on `TyreLife`), and `typical_stint`. It also carries a fixed `pit_loss`
  of 21.5 s and a `base_lap_time` taken as the median of dry-compound laps.
- The existing fallback table in that method already contains INTERMEDIATE and WET entries.
  Establish by inspection whether the current data path populates them from data or falls
  back, and record which — do not assume.
- Compound indices: SOFT 0, MEDIUM 1, HARD 2, INTERMEDIATE 3, WET 4.
- Wet-condition semantics already present in the reward layer: the rulebook compliance
  condition exempts the two-dry-compound requirement when a wet-weather compound has been
  used. Confirm the exact predicate in `_compute_reward` before relying on it. Do not change
  it in this phase.
- Frozen dry-condition evidence that must not be disturbed: the pace–survival results in
  `docs/drafts/reward_and_environment_recalibration_evidence.md` and the artefacts under
  `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/`.
- Known limitation to keep in view: calibrated SOFT is quicker and far more durable than
  MEDIUM, an artefact of fitting degradation across stints of unequal length. The same
  failure mode threatens any wet estimate, because wet stints are typically short, few and
  run in changing conditions.

## Scope

1. Confirm the Phase 03 decision permits proceeding, and enumerate its constraints.
2. Identify an appropriate, **verifiable** wet or mixed-condition 2024 Silverstone session
   using only the data Phase 03 confirmed available. Do **not** assume that a qualifying or
   race session is suitable; establish it from weather and track-status evidence.
3. Define a measurement protocol **before** touching any calibration, covering at minimum:
   - sample inclusion and exclusion rules,
   - which compounds are in scope,
   - how tyre age is approximated and its limitations,
   - traffic and safety-car or virtual-safety-car limitations,
   - what constitutes weather evidence for classifying a lap as wet or mixed,
   - how in-lap, out-lap, pit-affected and deleted laps are handled,
   - uncertainty quantification and minimum sample thresholds below which no parameter will
     be proposed.
4. Apply the protocol and produce an evidence artefact with observed values, derived
   estimates, uncertainty, and sample counts.
5. Produce a **proposed** parameter-change diff — not an applied one — showing exactly what
   would change and where, with the estimate and uncertainty behind each number.
6. Assess the consequences a wet calibration would have for existing frozen results, and
   state how dry-regime evidence remains separated.

## Explicit non-goals

- Do not edit `_load_calibration_2024` or any environment code until the user has approved
  both the evidence artefact and the proposed diff. This is absolute.
- Do not fabricate INTERMEDIATE or WET parameters. If evidence is inadequate, complete the
  phase with a documented **no-calibration decision**.
- Do not modify dry-compound parameters, the hazard model, `CRASH_HAZARD_SCALE`, pit
  mechanics, wear dynamics, action or observation spaces, wrappers, the evaluator, training
  entrypoints or reward coefficients.
- Do not introduce weather transitions, dynamic track state, or a wet-condition regime into
  the environment. Those are separate proposals.
- Do not retrain any model, and do not re-evaluate existing checkpoints against a changed
  calibration in this phase.
- Do not revise the frozen pace–survival conclusions, the two locked wording decisions, or
  the five retained `[VERIFY]` markers.
- Do not claim real-world meteorological or regulatory validity.

## Preconditions and repository inspection

Complete all of the following and report findings before the first edit.

1. `git branch --show-current` → `phase2-recalibration`; `git status --short` → clean.
2. Read the Phase 03 audit artefact and its `## Completion record`. If the recommendation was
   **do not proceed**, stop and report; do not begin measurement.
3. Read `F1RaceEnv._load_calibration_2024` in full. Record exactly how `compound_offsets`,
   `deg_per_lap` and `typical_stint` are computed, where clipping is applied, and under what
   conditions the fallback table is used.
4. Determine empirically whether INTERMEDIATE and WET entries in the live calibration are
   data-derived or fallback values, and record the actual runtime values.
5. Confirm the wet-exemption predicate in `_compute_reward` and every consumer of compound
   identity, so the blast radius of any proposed parameter change is known.
6. Inspect `data/silverstone_2024_laps.csv` for the columns and compounds actually present,
   including whether any wet-weather compound laps exist in it at all.
7. Confirm cache and network status, and whether the target session is already cached.
8. Check `.gitignore` so the evidence artefact is written to a trackable path.

## Implementation plan

Proposed sequence. Adjust with justification; report deviations.

1. **Gate check.** Restate the Phase 03 decision and its constraints. Proceed only if
   permitted.
2. **Session identification.** Using the audited data, evaluate candidate 2024 Silverstone
   sessions for wet or mixed conditions on the evidence actually available — rainfall or
   condition indicators over time, track status, and compound usage patterns. Report the
   evidence per candidate session and select one, or conclude that none is suitable.
3. **Protocol specification — approval gate.** Write the measurement protocol and present it
   for review **before** computing any calibration estimate. Include the minimum-sample
   thresholds and the exclusion rules, and pre-register them.

   Write the proposed measurement protocol and show it to the user. Do not execute the
   measurement analysis, derive calibration estimates, or create a proposed parameter diff
   until the user has explicitly approved that protocol.
4. **Measurement.** Apply the protocol. For each in-scope compound produce: sample count,
   pace offset relative to a stated baseline with uncertainty, a degradation estimate with
   uncertainty and the fit diagnostics, and observed stint lengths. Report every quantity as
   observed, derived, assumption or non-comparable.
5. **Robustness.** Test sensitivity to the main protocol choices — at minimum the exclusion
   rules and the tyre-age approximation — and report whether the estimates survive. Apply the
   lesson from the SOFT artefact: check explicitly whether short, unequal stints are driving
   the degradation slope.
6. **Proposed diff.** If and only if the evidence clears the pre-registered thresholds,
   author a proposed diff document showing the exact intended change to the calibration path,
   each number with its estimate and uncertainty, and the expected behavioural consequences.
   Do not apply it.
7. **Separation statement.** Document how dry-regime behaviour and the frozen pace–survival
   evidence remain unaffected and how any future wet results will be reported separately.
8. **Decision.** Conclude with either a proposed calibration awaiting approval, or a
   documented no-calibration decision with the reasons.

## Validation and acceptance criteria

Select checks from existing documented project commands only; do not invent test commands.
Available and appropriate here:

- `.venv_f1/bin/python -m compileall -q src scripts` — must exit 0 if any script is added.
- `git diff --check` — no whitespace or conflict-marker errors.
- `git status --short` — only intended paths.
- If a measurement script is added, run a reduced-scope smoke check first and confirm the
  output schema, following the Phase 2 precedent.
- Only if a calibration diff is later approved and applied, the existing regression check is
  `.venv_f1/bin/python scripts/validate_reward_regimes.py --episodes N --seed S`, which must
  still pass its documented acceptance conditions. Running it is **not** part of this phase
  unless a change is approved.

Acceptance criteria:

1. The Phase 03 gate is explicitly checked and cited.
2. Session suitability is established from evidence, not assumed.
3. The measurement protocol is written and pre-registered **before** estimates are computed.
4. Every estimate carries a sample count and an uncertainty statement.
5. Observed, derived, assumption and non-comparable quantities are separated.
6. No calibration or environment code is modified in this phase.
7. Either a proposed diff awaiting approval, or a documented no-calibration decision, exists.
8. Dry-regime and frozen evidence separation is stated explicitly.

## Required artefacts

- `docs/phase-04-wet-session-evidence.md` — session identification evidence, the
  pre-registered measurement protocol, per-compound estimates with uncertainty and sample
  counts, robustness checks, limitations, and the decision.
- `docs/phase-04-proposed-calibration-diff.md` — **only if** the evidence clears the
  thresholds: the exact proposed change, each parameter with its estimate and uncertainty,
  blast radius, and expected behavioural consequences. Explicitly labelled as proposed and
  unapplied.
- Optionally `scripts/measure_wet_calibration.py`, read-only, writing curated summaries under
  `outputs/phase-04-wet-calibration/<timestamp>/` with `manifest.csv` and `README.md`
  following the Phase 2 convention. Record-level CSVs must be gitignored; propose any new
  pattern before adding it.
- Update `## Completion record` in this file.

## Documentation and memory updates

- Update the memory handoff record at
  `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
  with a new dated Phase 04 section before ending the session, preserving all existing
  content.
- Record: the session selected or the no-suitable-session finding, the protocol summary, the
  estimates or the no-calibration decision, whether a proposed diff exists and its approval
  state, the commit SHA once approved, and any constraint Phase 05 must respect.
- Do not edit `docs/drafts/reward_and_environment_recalibration_evidence.md`. If a wet
  calibration is eventually approved and applied, its consequences are documented in a new
  artefact, not by rewriting the frozen section.

## Git and handoff procedure

1. Confirm the branch and a clean tree before starting.
2. Make no commit until the evidence artefact is complete and the user has reviewed the diff.
3. Propose a single focused commit. Suggested message shape:
   `docs: wet-session calibration evidence and proposed parameter diff`, or
   `docs: wet-session calibration evidence and no-calibration decision`.
4. Show `git diff --check`, `git status --short`, the full diff stat and the exact contents
   of new or modified Markdown before requesting approval.
5. **Do not push** until the user explicitly approves the final diff and commit.
6. Do not amend history, rebase, merge, switch branches or create additional commits.

## Stop conditions and decisions requiring approval

Stop and ask before proceeding if any of the following arise.

- The Phase 03 recommendation was **do not proceed**, or its constraints are unclear.
- No 2024 Silverstone session can be verified as wet or mixed on the available evidence.
- Sample counts fall below the pre-registered thresholds for any compound.
- The degradation estimate appears driven by short or unequal stints, repeating the known
  SOFT artefact.
- Applying a calibration change would alter dry-compound behaviour or invalidate frozen
  results.
- Any change to `_load_calibration_2024`, environment, reward, hazard, tyre, training,
  evaluator or model code appears necessary — propose, do not apply.
- Weather or track-status evidence is ambiguous enough that lap classification would rest on
  an assumption rather than an observation.

## Completion record

- **Date completed:** 2026-08-30
- **Completed tasks:** Phase 03 gate check and constraint enumeration; full
  precondition inspection (branch/HEAD/tree, `_load_calibration_2024` read in full,
  wet-exemption predicate and compound-identity blast radius, calibration CSV
  contents, cache/network status, `.gitignore` trackability); runtime calibration
  probe recording actual values for all five compounds; pre-registered measurement
  protocol (P0–P11) written and user-approved **before** any estimate was computed;
  read-only measurement implementing the protocol; P8 clip test; per-lap weather
  classification; robustness checks R1–R7 including the R7 falsification test;
  P9 threshold gate; evidence artefact; separation statement.
- **Files changed (with paths):**
  - `docs/phase-04-wet-session-evidence.md` (new — the evidence artefact)
  - `scripts/measure_wet_calibration.py` (new — read-only measurement script)
  - `docs/phase-04-plan.md` (this completion record)
  - `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
    (new dated Phase 04 section)
  - `outputs/phase-04-wet-calibration/20260830T114500/` (curated, 36 KB, 9 files:
    `manifest.csv`, `attrition.csv`, `weather_classification.csv`, `clip_test.csv`,
    `estimates.csv`, `stint_summary.csv`, `robustness.csv`, `thresholds.csv`,
    `README.md`) — trackable; no record-level CSV written, so no new `.gitignore`
    pattern was required.
  - **`docs/phase-04-proposed-calibration-diff.md` deliberately NOT created** — the
    evidence did not clear the pre-registered thresholds.
- **Checks run and results:** (exact commands and outcomes)
  - `.venv_f1/bin/python -m compileall -q src scripts` — exit 0.
  - `.venv_f1/bin/python scripts/measure_wet_calibration.py --smoke` — reduced-scope
    smoke check (B=200), passed, output schema verified, smoke output deleted before
    the full run.
  - `.venv_f1/bin/python scripts/measure_wet_calibration.py --label 20260830T114500`
    — full run, 5.4 s, B=10,000, seed 20260830, **0 KB `data/cache` growth**,
    9/11 P9 criteria passed.
  - Provenance: 0 unmatched rows and 0 `Compound`/`TyreLife`/`Stint` mismatches
    between the CSV and the cached session; reproduced `base_lap_time` 92.4335 s and
    every live `compound_offsets`/`deg_per_lap` value to six decimals.
  - `git diff --check`, `git status --short` — run before requesting commit approval.
- **Commit SHA:** _pending — not committed; user has not yet approved the diff._
- **Session selected, or no-suitable-session finding:** 2024 British Grand Prix
  **Race** session (the existing calibration source), treated as **time-varying**,
  not as a single condition. A verifiable wet window exists within it: of 188
  INTERMEDIATE laps, 173 classify `wet` and 15 `mixed` from the observed `Rainfall`
  boolean, **zero** classify `dry`, and their median `TrackTemp` is 21.0 °C against
  24.5–25.0 °C for every dry compound. Rainfall and compound choice agree completely
  for the audited compound, so lap classification rests on an observation, not an
  assumption — stop condition 7 did **not** fire. Per user scope decision, FP3/FP2/Q
  laps were not added; no session other than the Race was loaded.
- **Outcome:** **documented no-calibration decision.** No parameter change proposed.
- **Estimates with uncertainty and sample counts:** In-scope INTERMEDIATE sample
  158 laps / 22 `(Driver, Stint)` clusters / 19 drivers; `TyreLife` 3–12. All
  intervals are cluster bootstraps over `(Driver, Stint)`, B=10,000, percentile 95%.
  - **Pace offset — all 4 thresholds passed.** +10.858 s vs `B_current`
    [10.316, 11.755]; +10.966 s vs `B_dry_green` [10.442, 11.790]; +10.613 s vs
    `B_medium` [10.118, 11.412]. Live value +10.7135 lies **inside all three CIs**.
    Location-estimator sensitivity small (median 10.858 / trimmed 11.086 / mean
    10.745). **Corroborated; no change proposed** (0.145 s difference, a fifth of the
    CI halfwidth).
  - **Degradation slope — failed the two decisive thresholds.** Raw unclipped slope
    over all 188 laps is **−0.071881 s/lap**; the live `0.010000` is purely the
    `np.clip(slope, 0.01, 0.40)` floor at `f1_env.py:235`. Three estimators disagree
    in sign: pooled −0.227861 [−0.366, −0.084]; within-stint −0.116270
    [−0.210, **+0.000449**] (CI includes zero); condition-controlled **+0.541856**
    [+0.123, +0.761]. **No-calibration decision**; value retained but relabelled an
    unevidenced assumption.
  - **Non-identifiability, the decisive diagnostic:** within-stint demeaned
    Pearson r(`TyreLife`, `LapNumber`) = **1.0000**. Within a stint, tyre age and
    lap number are the same variable up to a shift, so degradation cannot be
    separated from track drying. All 22 stints fall inside laps 22–38 of a drying
    race; 14 of 22 within-stint slopes are negative and 8 positive.
  - **WET:** zero laps in the calibration source. No estimate attempted.
    Offset 11.0 s / `deg_per_lap` 0.25 / `typical_stint` 10 remain fallback
    constants, labelled unevidenced assumptions.
  - **R7 falsification passed:** the same protocol yields a clean identifiable
    answer for MEDIUM (447 laps, 23 stints, all three estimators within 0.05 s/lap),
    and detects the documented SOFT artefact (within-stint 0.125 vs live 0.044105,
    ~3×). The INTERMEDIATE failure is therefore a property of the data, not the
    method.
- **Unresolved issues:**
  - `typical_stint[3] = 9` is biased low by roughly 2 laps because `IsAccurate`
    removes in-laps and out-laps. It feeds `old_tyre_term` in `_segment_crash_prob`
    (`f1_env.py:521-526`), so the hazard model likely penalises intermediate tyre age
    ~2 laps too early. No change proposed — the protocol pre-registered no threshold
    for this quantity.
  - A **second** dry-compound anomaly, distinct from the documented SOFT one: HARD is
    calibrated 2.5615 s *faster* than the dry baseline and faster than SOFT, from 39
    laps across 3 stints, with a within-stint CI spanning zero. Dry parameters are a
    plan non-goal, so this is reported only.
  - Deleted laps (9 in this race) are **not** excluded by `IsAccurate` and are not
    excluded by the live calibration path.
  - `pit_loss = 21.5` s still never derived from `PitInTime`/`PitOutTime`.
  - The `## Completion record` in `docs/phase-03-plan.md` still records
    `Commit SHA: pending` and describes the Phase 03 curated outputs as uncommitted.
    Both are stale — the commit is `2ddcee1` and all five files are tracked. Not
    edited by this phase; flagged for the user.
- **Next 3–5 concrete tasks:**
  1. Obtain user approval of this diff and commit; do not push before approval.
  2. Decide whether to correct the stale `Commit SHA` and outputs-tracking wording in
     the `docs/phase-03-plan.md` completion record.
  3. If a defensible intermediate degradation estimate is still wanted, propose
     adding FP3 (54/82 rainfall-True, near-exclusively INTERMEDIATE, already cached)
     as a **separate, approved** scope extension — noting it breaks the age/elapsed-time
     collinearity only if stint age ranges actually overlap across sessions.
  4. Consider a separately approved proposal to correct the `typical_stint`
     out-lap-removal bias, since it reaches the hazard model.
  5. Begin Phase 05 (`docs/phase-05-plan.md`), whose precondition is independently
     establishing the physical unit of the `X`/`Y`/`Z` coordinate fields.

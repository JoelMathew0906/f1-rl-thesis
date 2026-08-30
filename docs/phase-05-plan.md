# Phase 05 — Silverstone geometry and digital-twin foundation

Execution-ready handoff prompt for a fresh Claude Code session. Read this file in full
before acting. This phase builds a small, validated, traceable geometry foundation — not a
physics model, and not a claim of digital-twin validity.

## Baseline and branch guardrails

- Work begins from `phase2-recalibration` **only**. Confirm with `git branch --show-current`
  before anything else.
- `main` and `baseline/pre-recalibration` are protected historical branches. Do not modify,
  merge, rebase, force-push or commit onto them. Verify their current refs before relying on
  a commit relationship.
- Schema design, inspection and the proposed-diff document are documentation-only and may
  remain on `phase2-recalibration`. Create a fresh feature branch from
  `phase2-recalibration` **only if** actual implementation changes begin, and only after
  proposing that step and obtaining approval.
- Enter Plan Mode and inspect the repository before making any edit.
- Keep scope minimal and evidence-driven. Do not invent telemetry, geometry, parameters,
  empirical claims or API capabilities. Every geometric fact must trace to verifiable
  circuit metadata or be labelled an assumption.
- The reward-regime and pace–survival evidence frozen at commit `e5f0d9e` stays frozen. Do
  not retroactively rewrite it to fit geometry findings.
- Obtain explicit approval before changing environment dynamics, reward logic, hazard logic,
  tyre parameters, training configuration, evaluator semantics, or any existing conclusion.
- Do not push until the user has explicitly approved the phase's final diff and commit.

## Objective

Establish a traceable, documented geometry schema for Silverstone that maps the existing
36-segment model onto named track regions, with every assumption recorded, while preserving
backwards compatibility for all existing consumers of `segment_id`. Deliver a small
validated foundation, not a physics or telemetry-driven redesign.

## Research question

Which elements of Silverstone circuit geometry are verifiably available and reliable, and
how can they be expressed as a traceable schema over the existing 36-segment model without
breaking any dependent output, log schema or analysis, and without overstating physical
fidelity?

## Inputs and evidence already frozen

- **Precondition:** this phase begins only after Phase 03 has **documented which circuit
  geometry and telemetry fields are actually available and reliable**. Read
  `docs/phase-03-fastf1-feasibility-audit.md` and its `## Completion record` first, and
  honour every constraint recorded there.
- The current track model, verified at the freeze commit: **36 segments — 18 `Straight` and
  18 `Corner`, alternating** — loaded from `data/silverstone_2024_track_segments.csv` by
  `load_or_build_silverstone_segments` in `src/f1_rl_safety/track.py`.
- The `TrackSegment` dataclass fields, and the CSV columns that mirror them: `id`,
  `segment_type`, `length`, `sector`, `corner_number`, `corner_name`, `approx_radius`,
  `start_x`, `start_y`, `end_x`, `end_y`.
- How the CSV was generated, per `_build_silverstone_track_segments_from_fastf1`: FastF1
  `CircuitInfo.corners` (`X`, `Y`, `Number`, `Letter`, optionally `Name`), cumulative
  straight-line distance between successive corner points, sector assigned by proportion of
  cumulative distance, and `approx_radius` computed as the **mean of the distances to the
  previous and next corner points** — a crude proxy, not a fitted radius. Corner segments
  carry `length = 0.0`.
- Known consequence of that proxy: the radius term contributes negligibly to hazard because
  the values are on the order of thousands in FastF1 coordinate units. Verify before relying
  on this.
- Hazard structure that must not change without approval: `_segment_crash_prob` in
  `src/f1_rl_safety/f1_env.py` scales a composite — a base rate multiplied by 8.0 for
  corners and 0.5 for straights, plus a radius term, a wear term, a risk-envelope excess
  term with a per-type tolerance, and a stale-tyre term — by the single uniform factor
  `CRASH_HAZARD_SCALE = 0.04`.
- Frozen evidence that depends on the current segment count: the pace-diagnostics
  aggregation sums exactly 36 segment times per lap. Changing `n_segments` would invalidate
  `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/` and the frozen evidence
  section. Treat this as a hard compatibility constraint.

## Scope

1. Confirm Phase 03 documented which geometry and telemetry fields are available and
   reliable, and enumerate the constraints.
2. Inspect **all** consumers of segment identity before proposing anything. At minimum:
   - `src/f1_rl_safety/track.py` — construction and CSV round-trip;
   - `src/f1_rl_safety/f1_env.py` — segment indexing, `_simulate_segment`,
     `_segment_crash_prob`, `_crash_reason`, and the crash log payload;
   - `src/f1_rl_safety/evaluate_rl.py` — crash-site formatting and the evaluation CSV schema
     columns that carry segment and corner identity;
   - `scripts/diagnose_env_control.py` and the other `scripts/analyse_*.py` consumers;
   - `scripts/analyse_pace_profiles.py` — depends on the 36-segment lap aggregation;
   - any notebook or plotting code that reads segment or corner fields;
   - existing output CSVs whose columns encode segment identity.
3. Build a traceable geometry schema sourced from verifiable circuit metadata, with an
   explicit mapping from existing segment identifiers to named track regions, and documented
   assumptions for straights, corners, distance boundaries and any radius proxy.
4. Preserve backwards compatibility, or produce an approved migration plan for every output,
   log schema and analysis that depends on `segment_id`.
5. Deliver a small validated foundation first, with explicit evidence of what it represents
   and, equally, what it does not represent.

## Explicit non-goals

- Do not claim the project is a physically validated digital twin. It is a geometry-aware
  strategic simulator. State this explicitly in every artefact produced.
- Do not introduce physics, telemetry-derived dynamics, or segment-weighted hazard changes
  unless the data and a testable design justify them **and** the user has approved the
  proposed diff.
- Do not change `n_segments`, the segment count, or the alternating straight/corner structure
  without an approved migration plan — doing so silently invalidates the frozen
  pace-diagnostics evidence.
- Do not modify `_segment_crash_prob`, `CRASH_HAZARD_SCALE`, tyre dynamics, pit mechanics,
  reward coefficients, action or observation spaces, wrappers, training entrypoints or
  evaluator semantics.
- Do not rename or repurpose existing CSV columns that downstream analyses depend on.
- Do not regenerate `data/silverstone_2024_track_segments.csv` in place; any regeneration
  must be additive, versioned and approved.
- Do not retrain or re-evaluate models.
- Do not rewrite the frozen evidence draft, the two locked wording decisions, or the five
  retained `[VERIFY]` markers.

## Preconditions and repository inspection

Complete all of the following and report findings before the first edit.

1. `git branch --show-current` → `phase2-recalibration`; `git status --short` → clean.
2. Read the Phase 03 audit and `## Completion record`. If geometry fields were found
   unavailable or unreliable, stop and report; do not design against absent data.
3. Read `src/f1_rl_safety/track.py` in full. Record the exact construction logic, the sector
   rule, the radius proxy formula, and the CSV round-trip behaviour including null handling.
4. Read the segment-consuming paths in `src/f1_rl_safety/f1_env.py` and record every place a
   segment attribute influences behaviour or logging.
5. Enumerate, by grep, every reference to `segment`, `segment_id`, `corner_number`,
   `corner_name`, `segment_type`, `approx_radius` and `sector` across `src/`, `scripts/`,
   `notebooks/` and committed `outputs/`. Produce a consumer inventory table.
6. Confirm the current segment CSV contents: row count, type counts, which fields are
   populated and which are empty for each type.
7. Confirm which committed evaluation CSVs carry segment identity columns, so the
   compatibility surface is known.
8. Check `.gitignore` so new artefacts are written to trackable paths.

## Implementation plan

Proposed sequence. Adjust with justification; report deviations.

1. **Gate check.** Restate the Phase 03 geometry findings and constraints. Proceed only if
   permitted.
2. **Consumer inventory.** Produce the complete table of segment-identity consumers, each
   with file, line, field used and the compatibility risk if it changed. Present this before
   proposing a schema.
3. **Source verification.** Establish which circuit metadata is verifiably available for
   Silverstone from the Phase 03 audit — corner numbering, corner names, coordinates, any
   distance or lap-fraction measure — and record units and provenance. Label anything not
   directly observed as an assumption.
4. **Schema design.** Define an **additive** geometry schema: named track regions mapped onto
   existing segment identifiers, with explicit documentation for how straights, corners,
   distance boundaries and any radius proxy are derived. The existing `segment_id` values and
   the 36-segment structure remain the primary key; new fields are supplementary.
   **Approval gate.** Do not execute any geometry-mapping generation script, produce any
   additive geometry artefact, or draft any proposed code diff until the user has approved
   the consumer inventory and the proposed schema design.

5. **Assumption register.** For every schema element, record whether it is observed, derived
   or assumed, and what would falsify it. Include an explicit statement of what the geometry
   does not represent: no elevation, no camber, no track width, no grip model, no racing line,
   unless Phase 03 evidenced otherwise.
6. **Compatibility plan.** Demonstrate that every consumer in the inventory continues to work
   unchanged, or produce a migration plan for each affected consumer, output and log schema,
   for approval before implementation.
7. **Proposed diff.** Author a proposed diff document showing exactly what would be added and
   where. Do not apply changes to environment or analysis code without approval; additive,
   read-only artefacts may be created if they touch no existing behaviour.
8. **Small validated foundation.** Validate the mapping — for example that every segment maps
   to exactly one named region, that corner numbering is consistent with the source metadata,
   and that the round-trip through the CSV is lossless — and report the checks and results.
9. **Scope statement.** Conclude with an explicit statement of what the foundation supports
   for future work and what it explicitly does not license claiming.

## Validation and acceptance criteria

Select checks from existing documented project commands only; do not invent test commands.
Available and appropriate here:

- `.venv_f1/bin/python -m compileall -q src scripts` — must exit 0.
- `git diff --check` — no whitespace or conflict-marker errors.
- `git status --short` — only intended paths.
- `.venv_f1/bin/python scripts/analyse_pace_profiles.py --episodes-fixed 3 --episodes-learned 2 --bootstrap 100 --label LABEL`
  — the documented reduced-scope reproducibility check. Its lap aggregation asserts 36
  segments per lap, so it is the direct regression guard for the compatibility constraint.
  Remove the temporary labelled output directory afterwards and say so.
- `.venv_f1/bin/python scripts/validate_reward_regimes.py --episodes N --seed S` — only if a
  change could affect environment behaviour; it must still pass its documented acceptance
  conditions.

Acceptance criteria:

1. The Phase 03 geometry gate is explicitly checked and cited.
2. A complete consumer inventory for segment identity exists and is presented before the
   schema.
3. Every schema element is labelled observed, derived or assumed, with provenance.
4. Backwards compatibility is demonstrated, or an approved migration plan exists for each
   affected consumer.
5. The 36-segment structure and `n_segments` are unchanged, or a migration plan has been
   approved and the frozen-evidence consequence documented.
6. The documented reproducibility check still passes.
7. Artefacts state explicitly that this is a geometry-aware foundation and not a validated
   digital twin, and enumerate what is not represented.
8. No hazard, physics or telemetry-derived dynamics were introduced.

## Required artefacts

- `docs/phase-05-geometry-schema.md` — the schema: source provenance, the segment-to-region
  mapping, derivation rules for straights, corners, distance boundaries and any radius proxy,
  the assumption register, and the explicit statement of what is and is not represented.
- `docs/phase-05-segment-consumer-inventory.md` — the compatibility surface: every consumer of
  segment identity with file, field, risk and compatibility verdict, plus any migration plan.
- `docs/phase-05-proposed-diff.md` — **only if** changes to existing code are warranted: the
  exact proposed change, its blast radius, and expected consequences, labelled as proposed
  and unapplied.
- Optionally an additive geometry artefact under
  `outputs/phase-05-geometry/<timestamp>/` with `manifest.csv` and `README.md` following the
  Phase 2 convention. Do not overwrite `data/silverstone_2024_track_segments.csv`.
- Update `## Completion record` in this file.

## Documentation and memory updates

- Update the memory handoff record at
  `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
  with a new dated Phase 05 section before ending the session, preserving all existing
  content.
- Record: the schema artefact path, the compatibility verdict, whether any code change is
  proposed and its approval state, the commit SHA once approved, and an explicit note that
  the foundation is geometry-aware rather than a validated digital twin.
- Do not edit `docs/drafts/reward_and_environment_recalibration_evidence.md`.

## Git and handoff procedure

1. Confirm the branch and a clean tree before starting.
2. Make no commit until the artefacts are complete and the user has reviewed the diff.
3. Propose a single focused commit. Suggested message shape:
   `docs: Silverstone geometry schema and segment consumer inventory`.
4. Show `git diff --check`, `git status --short`, the full diff stat and the exact contents
   of new or modified Markdown before requesting approval.
5. **Do not push** until the user explicitly approves the final diff and commit.
6. Do not amend history, rebase, merge, switch branches or create additional commits.

## Stop conditions and decisions requiring approval

Stop and ask before proceeding if any of the following arise.

- Phase 03 did not document geometry availability, or found the fields unreliable.
- The schema would require changing the segment count, `n_segments`, or the alternating
  straight/corner structure — this invalidates frozen pace-diagnostics evidence.
- Any consumer of `segment_id` cannot be preserved without a breaking change.
- A change to `_segment_crash_prob`, `CRASH_HAZARD_SCALE`, or any dynamics appears warranted
  by the geometry — propose with a testable design, do not apply.
- Regenerating or overwriting `data/silverstone_2024_track_segments.csv` seems necessary.
- Circuit metadata is ambiguous enough that a region boundary would rest on an assumption
  presented as an observation.
- The work would begin to resemble a physics model, telemetry-derived dynamics, or a
  digital-twin validity claim.

## Completion record

- **Date completed:** 2026-08-30

- **Completed tasks:** All eight preconditions. Precondition 1 verified (branch
  `phase2-recalibration`, tree clean, HEAD `15269ea`, protected refs `main` =
  `baseline/pre-recalibration` = `74c5f11`, annotated tag
  `baseline-pre-recalibration` dereferencing to the same commit). Phase 03 geometry
  gate read and **passed**. `src/f1_rl_safety/track.py` read in full; segment-consuming
  paths of `src/f1_rl_safety/f1_env.py` read and recorded with line references;
  grep enumeration of every segment-identity reference across `src/`, `scripts/`,
  `notebooks/`, `configs/` and committed `outputs/`; segment CSV contents confirmed;
  committed evaluation CSVs carrying segment identity identified; `.gitignore`
  trackability confirmed. Implementation steps 1–9: gate check, consumer inventory,
  source verification (including independently resolving the `X`/`Y`/`Z` coordinate
  unit), additive schema design, assumption register, compatibility plan, additive
  artefact generation, validation, and scope statement. Both approval gates were
  honoured: the consumer inventory and the schema design were each presented and
  approved before the artefact was generated.

- **Files changed (with paths):**
  - `docs/phase-05-geometry-schema.md` (new — the schema, provenance, sector-boundary
    uncertainty, assumption register, non-representation statement, scope statement)
  - `docs/phase-05-segment-consumer-inventory.md` (new — the compatibility surface)
  - `scripts/build_geometry_schema.py` (new — read-only generation script; imports
    nothing from `f1_env`, constructs no environment, reads the existing segments CSV
    for comparison only)
  - `outputs/phase-05-geometry/20260830T112139/` (new — 9 curated files, 52 KB:
    `segment_geometry.csv`, `column_provenance.csv`, `sector_boundaries.csv`,
    `sector_boundary_summary.csv`, `scale_fit_summary.csv`,
    `corner_distance_check.csv`, `validation.csv`, `README.md`, `manifest.csv`).
    Trackable — no `.gitignore` pattern matches, consistent with the Phase 03/04
    precedent that curated phase output directories are committed
  - `docs/phase-05-plan.md` (this completion record)
  - `.claude/projects/-Users-joel-mathew-Documents-GitHub-f1-rl-thesis/memory/project-context.md`
    (new dated Phase 05 section)
  - **Not changed:** `data/silverstone_2024_track_segments.csv` is byte-unchanged
    (sha256 `059983bafcef957fc0954fae89cbff167c1572bfd81e66e36104d4b029c4126e`). No
    environment, reward, hazard, tyre-calibration, training, model or evaluator code
    was modified. `docs/drafts/reward_and_environment_recalibration_evidence.md` was
    not edited. No proposed code diff was drafted, so
    `docs/phase-05-proposed-diff.md` was deliberately **not created** — no change to
    existing code is warranted.

- **Checks run and results:**
  - `.venv_f1/bin/python -m compileall -q src scripts` — **exit 0**
  - `.venv_f1/bin/python scripts/build_geometry_schema.py --label smoke_test --smoke`
    — reduced-scope smoke check (3 reference laps), 11/11 checks PASS; smoke output
    directory deleted before the full run
  - `.venv_f1/bin/python scripts/build_geometry_schema.py --label 20260830T112139`
    — full run, 19 reference laps, **12/12 validation checks PASS**. Headline check:
    the 18 derived straight arc lengths sum to 5837.6948 m against a measured lap
    length of 5837.6948 m, residual **0.000000 m**
  - `.venv_f1/bin/python scripts/analyse_pace_profiles.py --episodes-fixed 3
    --episodes-learned 2 --bootstrap 100 --label phase05_regression_check` — the
    documented reduced-scope reproducibility check. 243 fixed laps, 361 learned laps,
    550 summary rows, manifest `n_segments_per_lap = [36]`. **The 36-segment
    assertion holds.** Temporary labelled directory
    `outputs/phase2-recalibration/pace_diagnostics/phase05_regression_check/` removed
    after the run
  - `git diff --check` — clean; no tracked file modified
  - `git status --short` — only intended new paths
  - **Cache discipline:** `data/cache` **0 KB growth** (339,912 KB before and after);
    `data/fastf1_cache` untouched (101,448 KB); `fastf1.Cache.offline_mode(True)` set
    so a cache miss raises rather than fetching; **only the Race session was loaded**

- **Commit SHA:** _pending — not committed; awaiting user approval of the final diff._

- **Compatibility verdict:** **preserved.** No migration plan required. No consumer
  of segment identity needs any change; no output schema, log schema or committed CSV
  is affected. `n_segments` remains 36 and the alternating 18-straight/18-corner
  structure is unchanged, so the frozen pace-diagnostics evidence at
  `outputs/phase2-recalibration/pace_diagnostics/20260827T124054/` is unaffected —
  re-verified by the reproducibility check above.

- **Schema provenance summary:** 21 columns keyed on the existing `segment_id`:
  **9 observed, 10 derived, 2 assumed** (per
  `outputs/phase-05-geometry/20260830T112139/column_provenance.csv`). Single source
  session, 2024 British Grand Prix Race, from the warm cache offline. The phase's
  substantive new evidence is that **Phase 03's open coordinate-unit question is
  resolved: FastF1 `X`/`Y`/`Z` are decimetres (1/10 m)**, fitted over 19 laps at
  9.994137–10.053723 raw units per metre (mean 10.016642, 0.1664% from exactly 10.0);
  and that **`corners.Distance` is in metres on the telemetry origin** (12/18 corners
  agree exactly, max disagreement 5.04 m, below one 7.4 m sample interval). Distance
  boundaries are therefore **observed**, not assumed. Sector boundaries are published
  as **intervals** over 19 laps, because absolute boundaries shift materially
  (end-of-S1 17.27 m, end-of-S2 33.43 m) while lap fractions are tighter but
  **non-uniformly so** (S1 only 1.2x tighter, S2 2.8x, lap boundary 71.5x). The only
  `assumed` columns are `region_name_external` (FastF1 supplies no `Name` column;
  nullable, never a key) and `corner_angle_raw` (values observed, semantics
  undocumented, consumed by nothing). `region_key` is the authoritative identifier and
  is fully observed.

- **Explicit non-representation statement recorded:** **yes** — in
  `docs/phase-05-geometry-schema.md` (sections "Scope statement", "What this geometry
  does NOT represent", "What this foundation supports, and what it does not license")
  and in `outputs/phase-05-geometry/20260830T112139/README.md`. Both state that this
  is a geometry-aware strategic simulator, not a validated digital twin, and enumerate
  nine unrepresented properties: elevation, camber/banking, track width, kerb/run-off
  geometry, surface state or grip model, racing-line model, corner entry/exit speeds,
  fitted corner radii, and any vehicle-dynamics validation.

- **Unresolved issues:**
  1. **Two upstream defects in `track.py` are documented but not repaired.** Segment
     1's `length` is `0.0` where its true extent is 615.93 m of arc — `track.py:87`
     computes `total_len - s_coords[-1]`, identically zero for any circuit. And
     `track.py:90-91` under-assigns sector 1 by applying the proportion rule to the
     corner rather than the preceding straight. Both are **non-behavioural**: `length`
     and `sector` are read by no code. Repairing them means regenerating
     `data/silverstone_2024_track_segments.csv`, a plan stop condition, and requires a
     separate approved proposal.
  2. **`approx_radius` is numerically inert.** It contributes 0.015%–0.086% of the
     corner base rate, so all 18 corners are effectively identical to the hazard
     model, and the proxy is *larger* in fast open sections — inverting the intended
     "tighter corner is riskier" semantics. No change proposed: rewiring it is a
     dynamics change requiring separate approval and would invalidate frozen evidence.
  3. **`corners.Angle` semantics remain unresolved.** Two values look like angle-wrap
     artefacts (T18 = −178.57, T10 = −159.33, both real corners). Carried, consumed by
     nothing.
  4. **Sector boundaries derive from Race laps only.** The S1 boundary carries real
     uncertainty (~0.25% of a lap) even in fractional form. Sector assignment is
     nevertheless verified robust at a 9.7x margin.
  5. **Arc and chord measure different reference paths** — `corners.Distance` follows
     the racing line, `X`/`Y` chords the centreline map. 4 of 18 chords exceed their
     own arc (segments 3, 21, 23, 33). The two must never be mixed in one calculation.
  6. **The Phase 03 `## Completion record` carries two stale statements** — "Commit
     SHA: pending" and the claim that
     `outputs/phase-03-fastf1-audit/20260830T092331/` is gitignored/local-only. Both
     are false: that work is committed at `2ddcee1` and all five files are tracked.
     Flagged in Phase 03's and Phase 04's records; still uncorrected in
     `docs/phase-03-plan.md`.
  7. **`pit_loss = 21.5` s still never derived** from `PitInTime`/`PitOutTime`.
     Carried forward from Phase 03 and Phase 04.

- **Next 3–5 concrete tasks:**
  1. Obtain user approval of this diff; commit with message
     `docs: Silverstone geometry schema and segment consumer inventory`.
  2. Decide whether the two non-behavioural `track.py` defects (segment-1 `length`,
     scalar `sector`) warrant a separate approved proposal to regenerate
     `data/silverstone_2024_track_segments.csv` additively and versioned, or whether
     documenting them in the Phase 05 artefact is sufficient.
  3. Correct the two stale statements in the `## Completion record` of
     `docs/phase-03-plan.md` as a small, separate, approved documentation fix.
  4. If region-level reporting is wanted, aggregate the existing committed crash
     columns (`crash_segment_id`, `crash_corner_number`) against
     `segment_geometry.csv` on `segment_id` — a read-only analysis needing no code
     change and no new data.
  5. If a measured pit-lane loss is wanted to corroborate the fixed 21.5 s constant,
     derive it from `PitInTime`/`PitOutTime` as a new, separately approved analysis.

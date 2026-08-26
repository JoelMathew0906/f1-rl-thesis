# Reward-Regime Redesign and Environment Recalibration: Evidence-Grounded Draft

**Status:** working draft for thesis incorporation. All numerical statements carry a
parenthetical source path referring to committed repository artefacts. Items that could
not be substantiated from committed files are marked `[VERIFY]`.

**Provenance:** commits `cc27282` (reward regimes), `4b4448d` (environment recalibration
and revised acceptance tests), `d9b36f7` (PPO baseline study artefacts), `4af3f85`
(configurable discount factor and gamma ablation), `be6ce5c` (artefact hygiene), on branch
`phase2-recalibration`, with the pre-recalibration state preserved at tag
`baseline-pre-recalibration`.

---

# Scope and rationale

This chapter documents a preparatory methodological phase undertaken before any
comparison of reinforcement-learning architectures. The motivating concern is one of
**internal validity**. A comparative study asks which learning algorithm best acquires a
target behaviour under a given objective; such a comparison is only interpretable if the
objective in fact rewards the target behaviour, and if the environment in fact permits it.
If either condition fails, an architecture comparison measures the algorithms' responses
to a misspecified problem rather than their relative competence at the intended task.

Evidence from the frozen baseline state indicated that neither condition held. Under the
pre-recalibration environment, the segment hazard model implied a per-lap crash
probability of approximately 0.27 even for a cautious driver, corresponding to a
probability of completing the fifty-two-lap race of the order of 1e-7 (source:
`src/f1_rl_safety/f1_env.py`, lines 19–22, recorded as an in-code calibration note)
`[VERIFY: the underlying probe was computed inline and was not persisted as a committed
artefact]`. Race completion was therefore effectively unreachable, and every reward term
conditioned on finishing the race could not be encountered during training. In parallel,
the reward specification paid the performance regime directly for elevated risk, and the
compliance provisions of the rulebook regime were gated on a terminal condition that could
not be reached. Under these circumstances the three regimes produced behaviour that was
qualitatively indistinguishable.

The work reported here therefore pursues **behavioural alignment** rather than physical
fidelity. The objective was to establish that the three reward regimes encode
distinguishable strategic preferences, and that the simulated environment admits the
strategies those preferences describe. No claim is advanced that the simulator constitutes
a physical model of Formula 1, that its hazard rates correspond to real-world accident
probabilities, or that its pit and compound provisions reproduce the FIA Sporting
Regulations. The compliance construct used throughout is an explicitly stated proxy
(at least one pit stop and at least two distinct dry compounds, subject to a
wet-tyre exemption), not a representation of the regulations themselves.

---

# Environment recalibration

Two defects were identified by inspection and corrected. Both are calibration or
placement corrections rather than redesigns; the structural form of the hazard model, the
observation and action spaces, the tyre-degradation model and the Gymnasium interface were
left unchanged.

## Crash-hazard recalibration

A single uniform scaling constant, `CRASH_HAZARD_SCALE = 0.04`, was introduced and applied
once to the composite segment hazard (source: `src/f1_rl_safety/f1_env.py`, line 25, and
lines 528–530, where the scaled quantity is formed as
`CRASH_HAZARD_SCALE * (base + radius_term + wear_term + exceed_term + old_tyre_term)`).
Because the factor multiplies the assembled composite, every qualitative relation within
the hazard model is preserved exactly: corner segments remain more hazardous than
straights, hazard remains monotonically increasing in tyre wear and in risk demanded beyond
the segment-specific envelope, the stale-tyre term continues to apply beyond the typical
stint length, and the catastrophic-outcome mechanism continues to be conditioned on a
sampled crash. The pre-existing clipping of the resulting probability was retained.

The motivation was that the unscaled hazard rendered the terminal reward structure
unreachable, as described above. The recalibration was intended to make cautious
completion of a full race plausible while retaining a substantial, strategy-dependent
attrition risk.

## Pit-loss timing correction

In the pre-recalibration implementation, pit-lane time loss was charged only on segments of
type *straight*, whereas the tyre change itself was executed on lap completion, which
occurs on a corner segment. The two events therefore never coincided: an effective pit stop
changed tyres without incurring the intended time penalty, while a pit signal issued
mid-lap incurred a time cost without changing tyre state. The correction charges the full
calibrated pit-lane loss on precisely the step at which the pit takes effect (source:
`src/f1_rl_safety/f1_env.py`, lines 457–461, conditioned on `pit_decision and
completed_lap`), using the existing calibration value of 21.5 seconds (source:
`src/f1_rl_safety/f1_env.py`, lines 196 and 253). Action semantics were not altered: a
mid-lap pit signal was already a no-op with respect to tyre state and now simply carries no
spurious cost.

## Action and pit-boundary treatment

The environment executes a pit stop only when a pit is requested on the final segment of a
lap. This boundary was confirmed by inspection and was deliberately left unchanged; it is
a property of the frozen action semantics rather than a defect. It has a material
consequence for learning, since the pit decision is effective on only one of the
thirty-six segment steps per lap, and it accordingly informed the design of the
pit-opportunity diagnostics described below.

## Validation method and validated figures

Validation used a deterministic scripted-policy harness
(`scripts/validate_reward_regimes.py`) which drives five hand-specified strategies through
each regime with matched episode seeds, together with inline hazard and pit probes. The
committed scripted-policy evidence supports the following:

- A strategy that pits on every lap, and therefore maintains fresh tyres throughout,
  attained a crash rate of 0.468, implying completion of approximately 53 per cent of
  episodes, with a catastrophic-event rate of 0.012 (source:
  `outputs/phase2-recalibration/reward_validation/scripted_policy_seed123_n250_20260826T160205_summary.csv`).
  This provides committed evidence that full-race completion is attainable under the
  recalibrated hazard for a tyre-managed policy.
- Single-stop strategies, which run long stints, attained crash rates of 0.904 and 0.908,
  and zero-stop strategies 0.852–0.956, with catastrophic rates ordering monotonically with
  scripted risk from 0.012 to 0.228 (same source). Attrition therefore remains
  strategy-dependent rather than uniform.
- The correct pricing of pit stops is evidenced indirectly but strongly by the collapse of
  the every-lap-pitting strategy to a mean return of −722.11 in the unconstrained regime
  (same source), consistent with approximately forty stops each charged at the calibrated
  pit-lane loss.

A direct probe reporting a cautious two-stop completion probability of approximately 0.357
and a single-step pit cost of approximately 24.1 seconds was executed during development
but was not written to a committed artefact
`[VERIFY: re-run and persist a hazard-and-pit probe artefact if these specific figures are
to be cited in the thesis]`.

---

# Reward-regime specification

All reward coefficients are declared in the `reward_regimes` block of
`configs/configs_silverstone.yaml`, which is loaded and strictly validated at environment
construction; the loader requires every key for every regime and fails with an explicit
error naming the offending regime and key, so that partial or silently defaulted
coefficients cannot occur (source: `src/f1_rl_safety/f1_env.py`, `load_reward_weights`).
The three regimes share a single term schema and differ only in coefficients, a term being
disabled by a coefficient of zero.

| Regime | Intended optimisation objective | Principal active incentives (source: `configs/configs_silverstone.yaml`) | Expected behavioural signature |
|---|---|---|---|
| UNCONSTRAINED | Maximise race performance, tolerating greater risk than the safe regime, while retaining a genuine incentive to complete rather than to crash | Highest pace weighting (`w_pace: 1.0`); largest position-change weight (`w_position_delta: 0.5`); lowest crash penalty (`crash_penalty: 10.0`); **no** per-step risk penalty (`w_risk_step: 0.0`); no pit or compound milestone; no compliance settlement | Fast, risk-tolerant driving; little or no incentive to pit; higher attrition |
| RULEBOOK | Balance performance against a proxy for regulatory compliance and moderate safety pressure | Moderate pace (`w_pace: 0.4`); risk penalised only beyond a tolerance (`w_risk_step: 0.02`, `risk_free_threshold: 0.4`); one-off `pit_milestone_bonus: 8.0` and `compound_milestone_bonus: 6.0`; terminal `finish_compliant_bonus: 10.0` against `finish_noncompliant_penalty: 30.0`; small `no_pit_step_penalty: 0.005`; `over_pit_penalty: 2.0` beyond `over_pit_cap: 3` | Compliant one-stop strategies with at least two dry compounds; moderate risk |
| SAFE | Prefer conservative, low-risk, compliant strategies, accepting a slower race | Lowest pace weighting (`w_pace: 0.3`); highest survival shaping (`w_alive: 0.03`, `w_lap: 1.0`, `w_finish: 30.0`); highest crash and catastrophic penalties (`crash_penalty: 40.0`, `catastrophic_penalty: 60.0`); risk penalised from zero (`w_risk_step: 0.03`, `risk_free_threshold: 0.0`); stale-tyre pressure via `w_wear_step: 0.1` beyond `wear_threshold: 0.6`; modest `pit_milestone_bonus: 2.0`; no compound milestone | Low or negative demanded risk; tyre management; longest survival; slower races |

Two design properties merit emphasis. First, the performance regime is not paid for risk as
such: its advantage arises only through the pace term, so that elevated risk is beneficial
only insofar as it produces measurable lap-time gain and is otherwise penalised through
increased hazard exposure. Second, the compliance settlement is applied only to episodes
that complete the race; crashed episodes settle no compliance term. This avoids both
penalising an agent for a provision that never became binding and creating an incentive to
crash early in order to avoid an accumulating liability. Pit and compound milestones are
paid once per episode, so that repeated stops cannot be used to farm reward.

---

# Behavioural validation

Two forms of evidence must be distinguished, and are reported separately throughout.

**Deterministic scripted-policy validation of reward ordering.** Five deterministic
strategies expressible in the existing action space — an aggressive zero-stop, a fast
zero-stop, a compliant one-stop, a conservative one-stop and an every-lap-pitting policy —
were evaluated over 250 episodes per regime with matched episode seeds. Seven
pre-specified ordering conditions were assessed and all seven were satisfied (source:
`outputs/phase2-recalibration/reward_validation/scripted_policy_seed123_n250_20260826T160205_checks.csv`).
The supporting margins were substantial: in RULEBOOK the compliant one-stop attained 23.51
against −22.57 for the aggressive zero-stop and −33.41 for the fast zero-stop; in SAFE the
conservative one-stop attained 4.26 against −51.67 for the aggressive zero-stop; and in
UNCONSTRAINED the compliant one-stop attained 9.54 against −9.39 for the conservative
one-stop, −19.90 for the aggressive zero-stop and −722.11 for the every-lap-pitting policy
(source: `..._160205_checks.csv` and `..._160205_summary.csv`). In SAFE the full ranking
ordered monotonically with scripted risk exposure (same source).

It should be stated precisely what this establishes. Passing these acceptance conditions
provides evidence that, **for the specific policies tested and under the evaluated
protocol**, each regime's reward ordering agrees with its intended qualitative preference.
It does not establish that the intended behaviour is globally optimal under each reward
function, nor that no unintended high-return policy exists outside the tested set. The
tests are a falsification instrument for reward misspecification, not a proof of
optimality.

**Learned behaviour after finite training.** Separately, and with weaker evidential status,
the behaviour actually acquired by an agent under a finite interaction budget was examined.
The two are logically independent: a reward function may order strategies correctly while
remaining difficult to optimise within a given budget, which is precisely the pattern
observed below.

An intermediate iteration of the acceptance suite is retained in the repository as a record
of the diagnostic process. An earlier condition comparing two zero-stop strategies failed
once the recalibrated tyre degradation was in force, because a strategy running a single
medium compound for roughly twenty-nine laps is itself strategically pathological and is
correctly penalised by the pace term (source:
`outputs/phase2-recalibration/reward_validation/scripted_policy_seed123_n250_20260826T155553_checks.csv`).
The condition was accordingly re-specified to compare a pit-managed, pace-viable policy
against an envelope-exceeding early-crash policy, which is the comparison that isolates the
intended pathology.

---

# PPO baseline and gamma ablation

## Protocol

Proximal Policy Optimisation was trained across the three regimes with three training seeds
(0, 1, 2) and a budget of 100,000 environment steps per run, and each resulting model was
evaluated over 50 deterministic episodes, giving 150 evaluation episodes per regime per
discount condition (source:
`outputs/phase2-recalibration/ppo_baseline_100k_20260826T161335/manifest.csv` and
`outputs/phase2-recalibration/ppo_gamma_ablation_0999_20260826T162818/manifest.csv`, which
record the exact training command, evaluator invocation, model path and timestamps for each
of the nine runs per condition). Evaluation used the existing evaluator with a deterministic
policy. In addition, a diagnostic replay of the identical deterministic episodes recorded
the pit action channel at every lap-final segment, that is, at every step where a pit stop
could take effect; this measurement altered no policy, reward, threshold or timing.

Only the discount factor differed between conditions. The baseline condition used the
default value of 0.99 and the ablation condition used 0.999; all other hyperparameters, the
environment and the reward configuration were identical.

## Results

| Metric (mean across 3 seeds; SD in parentheses) | UNCONSTRAINED γ=0.99 → γ=0.999 | RULEBOOK γ=0.99 → γ=0.999 | SAFE γ=0.99 → γ=0.999 |
|---|---|---|---|
| Mean return | −27.79 (31.00) → −33.59 (15.39) | −34.27 (7.04) → **+2.54 (32.29)** | −42.15 (5.69) → −25.66 (27.60) |
| Pit stops per episode | 0.000 → 0.000 | 0.013 (0.023) → **0.580 (0.502)** | 0.000 → **0.240 (0.416)** |
| Compliance rate | 0.000 → 0.000 | 0.013 (0.023) → **0.567 (0.491)** | 0.000 → 0.000 |
| Mean risk | 0.763 (0.360) → 0.680 (0.197) | 0.091 (0.576) → 0.197 (0.186) | −0.225 (0.438) → **−0.411 (0.049)** |
| Catastrophic rate | 0.220 → 0.200 (0.080) | 0.147 (0.031) → 0.120 (0.069) | 0.173 (0.042) → 0.140 (0.120) |
| Finish rate | 0.027 (0.046) → 0.033 (0.031) | 0.100 (0.035) → 0.080 (0.020) | 0.107 (0.061) → **0.167 (0.083)** |
| Completed laps | 20.05 (6.44) → 20.98 (2.41) | 25.31 (2.93) → 26.21 (2.11) | 27.01 (2.17) → **28.43 (1.01)** |
| Mean time, finished races (s) | 5119.57 → 5130.60 (8.33) | 5449.63 (27.97) → **5058.73 (333.91)** | 5155.44 (38.26) → 5124.67 (58.81) |

(Sources: `outputs/phase2-recalibration/ppo_baseline_100k_20260826T161335/summary_aggregated.csv`
and `outputs/phase2-recalibration/ppo_gamma_ablation_0999_20260826T162818/summary_aggregated.csv`.
Each cell aggregates 150 deterministic evaluation episodes per condition.)

**Pit discovery under the baseline discount factor.** At γ=0.99, pitting was essentially
absent: 0.013 stops per episode in RULEBOOK and none in the other two regimes (source:
baseline `summary_aggregated.csv`). The diagnostics indicate that this reflected learned
avoidance rather than absent exploration: across nine models the mean pit-channel action at
lap-final opportunities was at or near the lower bound of its range, with a maximum of
exactly zero in seven of the nine models, and only 2 pit intents recorded across 11,354
lap-final opportunities (source: baseline `summary_pit_diagnostics.csv`).

**Effect of γ=0.999 on RULEBOOK.** Raising the discount factor was associated with a
substantial increase in pit and compliance behaviour in the RULEBOOK regime, from 0.013 to
0.580 stops per episode and from 0.013 to 0.567 compliance, accompanied by a change in mean
return from −34.27 to +2.54 and a reduction in mean time among finished races from 5449.63
to 5058.73 seconds (sources as tabulated above). The diagnostics show the corresponding
mechanism: in the two discovering seeds the pit-channel action attained mean values of
0.1104 and 0.0969 with maxima of 0.5941 and 0.6212, and pit intent was expressed at 3.62
and 3.22 per cent of lap-final opportunities, which corresponds to approximately one stop
per race rather than indiscriminate pitting (source: ablation
`summary_pit_diagnostics.csv`). Compound diversity appeared concurrently, with a
`SOFT;MEDIUM` combination recorded in 43 and 42 of 50 episodes respectively (source:
ablation `summary_per_seed.csv`). This pattern is consistent with the interpretation that
the immediate pit-lane cost and the delayed benefit of fresh tyres require a longer
effective credit-assignment horizon than γ=0.99 affords, although a single ablation on one
environment does not establish that mechanism conclusively.

**SAFE regime: risk and survival.** Under γ=0.999 the SAFE regime exhibited the lowest
learned mean demanded risk of the three regimes, −0.411, with a notably small
across-seed standard deviation of 0.049, together with the highest finish rate (0.167) and
the greatest mean number of completed laps (28.43) (source: ablation
`summary_aggregated.csv`). Pit behaviour emerged in one of three seeds, at 0.240 stops per
episode, without compound diversity, which is consistent with SAFE possessing a pit
milestone but no compound milestone in its configuration (source:
`configs/configs_silverstone.yaml`). Under the baseline discount factor, crashes in this
regime were attributed predominantly to `tyre_overheating`, at mean tyre ages of 23.4–24.6
laps and mean wear of 0.834–0.876 (source: baseline `summary_per_seed.csv`), indicating
that attrition arises from degradation rather than from demanded aggression
`[VERIFY: the ablation per-seed summary does not carry crash-reason or terminal tyre-state
columns; these are available only in the uncommitted per-episode evaluation files]`.

**UNCONSTRAINED regime: incentive-consistent aggression.** The performance regime retained a
zero-stop profile under both discount conditions, with the highest mean demanded risk
(0.763 and 0.680), the highest catastrophic rate (0.220 and 0.200), the lowest finish rate
(0.027 and 0.033) and the fewest completed laps (20.05 and 20.98) (sources as tabulated).
Among finished races its mean time was the shortest of the three regimes at γ=0.99
(5119.57 s), albeit on a very small number of finishing episodes. This profile is
consistent with the configured incentives, which provide the highest pace weighting, the
lowest crash penalty, no per-step risk penalty and no pit or compound milestone; the
behaviour is therefore interpreted as incentive-consistent rather than as a failure of
learning.

**Ordering across regimes.** Under γ=0.999 the mean demanded risk ordered as intended,
0.680 for UNCONSTRAINED, 0.197 for RULEBOOK and −0.411 for SAFE, with catastrophic rates
ordering in the same direction (0.200, 0.120, 0.140) (source: ablation
`summary_aggregated.csv`).

**Seed variability.** Pit discovery was bimodal across seeds: two of three RULEBOOK seeds
and one of three SAFE seeds acquired pitting, while the remaining seeds recorded none
(source: ablation `summary_per_seed.csv`; corroborated by the zero pit-intent rows of
ablation `summary_pit_diagnostics.csv`). The large across-seed standard deviations on the
affected metrics — 0.502 on RULEBOOK stops and 32.29 on RULEBOOK return — are a direct
consequence, and mean values alone are therefore an inadequate summary of this behaviour.

---

# Decision and transition

The reward specification and environment dynamics were frozen at the conclusion of this
phase. The rationale is methodological: a comparison across architectures requires a fixed
problem definition, since any concurrent adjustment of the reward function or the dynamics
would confound algorithmic differences with changes to the task itself. Freezing occurred
only after both the reward ordering and the environment's admission of the intended
strategies had been evidenced, so that the frozen definition is a validated one rather than
merely a fixed one. The freeze is conditional: it is to be revisited only if subsequent
evidence identifies a genuine defect, in which case the defect, the correction and a
re-validation should be documented as this phase has been.

PPO with γ=0.999 is adopted as the reference configuration for the corrected environment.
It is important to record that this is a study-level protocol decision and not a change of
software default: the command-line interface retains `default=0.99`, described in the help
text as matching the library default and all pre-existing runs, and the value is applied to
the PPO constructor only (source: `src/f1_rl_safety/train_rl.py`, lines 269–277 and line
91). Reproducing the reference configuration therefore requires an explicit override. It
follows that any future algorithm intended for fair comparison against this reference must
have its discount factor set explicitly and identically, since a challenger left at 0.99
would be disadvantaged by precisely the long-horizon credit-assignment limitation
documented above.

Two reporting requirements follow from the evidence. First, future experiments must employ
at least three training seeds, because the principal behaviour of interest was acquired by
some seeds and not others under an identical configuration. Second, reporting must include
discovery rates — the proportion of seeds, and of episodes, exhibiting the target behaviour
— alongside central tendency. A mean of 0.580 stops per episode is compatible both with
uniform moderate pitting and with two seeds pitting consistently while a third does not,
and only the latter is what occurred here (source: ablation `summary_per_seed.csv`).
Reporting means alone would misrepresent the finding.

---

# Limitations

| Limitation | Basis in evidence | Consequence for interpretation |
|---|---|---|
| Single custom simulated circuit and environment | One environment implementation, `src/f1_rl_safety/f1_env.py`, with segments derived for a single event | No generalisation across circuits or events is supported |
| No real-world predictive calibration | The hazard scale is a chosen constant (`CRASH_HAZARD_SCALE = 0.04`, `f1_env.py:25`) with no external validation target in the repository | Crash rates, catastrophic rates and finish rates are internal quantities only; no correspondence to real accident probabilities is claimed |
| Compliance is a proxy construct | Implemented as at least one pit stop and at least two dry compounds, with a wet-tyre exemption (`f1_env.py`, `_compute_reward`; coefficients in `configs/configs_silverstone.yaml`) | Results speak to a stated proxy, not to the FIA Sporting Regulations |
| Seed variability and bimodal pit discovery | 2/3 RULEBOOK and 1/3 SAFE seeds acquired pitting; SD 0.502 on stops, 32.29 on return (ablation `summary_per_seed.csv`, `summary_aggregated.csv`) | Central tendency is insufficient; discovery rates must be reported; three seeds is a minimum, not a comfortable margin |
| Limited interaction budget | 100,000 environment steps per run (`manifest.csv`, both studies) | Findings describe behaviour attainable under this budget; longer training may alter conclusions |
| Finite deterministic evaluation sample | 50 episodes per model, 150 per regime per condition (`manifest.csv`) | Rates such as finish rate (0.027–0.167) rest on few positive events and carry wide uncertainty |
| Single-condition ablation | One discount value compared against one baseline, one budget, three seeds (both `summary_aggregated.csv`) | The credit-assignment interpretation is suggested, not established; no dose–response over γ was measured |
| Scripted validation covers tested policies only | Seven conditions over five scripted strategies, one seed stream (`..._160205_checks.csv`) | Verifies intended ordering for those policies; does not guarantee global optimality or absence of unintended exploits |
| Action-interface asymmetry in planned future work | Continuous action space in `f1_env.py`; discrete mapping in `src/f1_rl_safety/wrappers.py` | Value-based and policy-gradient candidates will not act through an identical interface; this must be reported as a fairness limitation of the subsequent comparison |
| Effective pit decisions occur at one step per lap | Pit executes only on lap completion (`f1_env.py:457–461`) | Pit exploration is intrinsically sparse; algorithms with differing exploration mechanisms may be affected unequally |
| Unpersisted development probes | Cautious-completion probability (~0.357) and single-step pit cost (~24.1 s) not written to a committed artefact | Marked `[VERIFY]`; not cited as evidence pending a persisted re-run |

---

# Thesis-ready text

## 1. Methods: reward-regime specification and environment recalibration

Prior to comparing learning architectures, a preparatory phase established the internal
validity of the objective functions and the plausibility of the simulated environment. The
motivation was that a comparative study presupposes a problem definition in which the
intended behaviour is both rewarded and attainable; evidence from the preceding
implementation indicated that neither condition was satisfied, since the segment hazard
model implied a per-lap crash probability of approximately 0.27 for a cautious driver and a
correspondingly negligible probability of completing a fifty-two-lap race (source:
`src/f1_rl_safety/f1_env.py`, lines 19–22).

Two corrections were applied to the environment. First, the segment hazard was rescaled by
a single uniform constant applied to the assembled composite hazard, `CRASH_HAZARD_SCALE =
0.04` (source: `src/f1_rl_safety/f1_env.py`, line 25 and lines 528–530). Because the factor
multiplies the composite rather than any individual term, all qualitative relations within
the hazard model were preserved: corners remained more hazardous than straights, hazard
remained increasing in tyre wear and in demanded risk beyond a segment-specific envelope,
and the catastrophic-outcome mechanism remained conditioned on a sampled crash. Second, the
pit-lane time loss was relocated so that it is charged on the step at which the tyre change
takes effect, namely lap completion, using the existing calibration value of 21.5 seconds
(source: `src/f1_rl_safety/f1_env.py`, lines 457–461 and lines 196 and 253). In the previous
implementation the cost and the tyre change never coincided, so that effective stops were
untimed while mid-lap pit requests incurred cost without effect. Observation and action
spaces, tyre degradation, the environment interface and the learning algorithm's
hyperparameters were not modified.

The reward function was re-specified so that all three regimes share a single term schema
and differ only in coefficients, which are declared exclusively in the `reward_regimes`
block of `configs/configs_silverstone.yaml` and are strictly validated at construction; the
loader requires every coefficient for every regime and raises an error naming any missing or
non-numeric key, precluding silent defaults. The schema comprises a pace term defined
relative to a policy-independent baseline, per-step survival and per-lap completion terms, a
race-completion bonus, a position-change term, crash and catastrophic penalties, a one-sided
risk penalty applied only above a regime-specific tolerance, a thresholded tyre-wear
penalty, single-payment milestones for a first pit stop and for the use of two dry
compounds, a penalty for stops beyond a cap, and a terminal compliance settlement.

Three properties of this design warrant statement. The performance regime is not rewarded
for risk as such; its risk tolerance arises solely because risk may reduce lap time, while
being penalised implicitly through hazard exposure. The compliance settlement is applied
only to episodes that complete the race, which avoids both penalising an agent for a
provision that never became binding and creating an incentive to terminate an episode early
in order to escape an accumulating liability. Milestones are paid once per episode, so
repeated stops cannot be used to accumulate reward. Compliance is operationalised as a
stated proxy — at least one pit stop and at least two distinct dry compounds, subject to a
wet-tyre exemption — and is not presented as a representation of the sporting regulations.

Validation employed a deterministic scripted-policy harness in which five hand-specified
strategies expressible within the existing action space were driven through each regime
under matched episode seeds, with seven pre-specified ordering conditions assessed over 250
episodes per regime. The reward specification was accepted only when all conditions were
satisfied. Learned behaviour was then examined separately, with Proximal Policy Optimisation
trained across three regimes and three seeds at 100,000 environment steps per run and
evaluated over 50 deterministic episodes per model, supplemented by a diagnostic replay
recording the pit action channel at every step at which a pit could take effect (sources:
`scripts/validate_reward_regimes.py`; manifests under
`outputs/phase2-recalibration/`).

## 2. Results: validation of reward ordering and learned behaviour

Deterministic scripted-policy validation satisfied all seven pre-specified ordering
conditions (source:
`outputs/phase2-recalibration/reward_validation/scripted_policy_seed123_n250_20260826T160205_checks.csv`).
In the rulebook regime the compliant one-stop strategy attained a mean return of 23.51,
against −22.57 for an aggressive zero-stop and −33.41 for a fast zero-stop; in the safe
regime the conservative one-stop attained 4.26 against −51.67 for the aggressive zero-stop,
with the full ranking ordering monotonically with scripted risk exposure; and in the
unconstrained regime the compliant one-stop attained 9.54 against −9.39 for a deliberately
slower conservative one-stop and −722.11 for an every-lap-pitting strategy (source:
`..._160205_summary.csv`). The last figure also evidences the corrected pricing of pit
stops. Attrition remained strategy-dependent under the recalibrated hazard: crash rates
ranged from 0.468 for the fresh-tyre every-lap-pitting policy to 0.956 for the aggressive
zero-stop, with catastrophic rates ordering from 0.012 to 0.228 (same source). These results
provide evidence that, for the policies tested and under the evaluated protocol, each
regime's reward ordering agrees with its intended preference; they do not establish that the
intended behaviour is globally optimal.

Learned behaviour diverged from this ordering under the default discount factor. At γ=0.99,
pitting was effectively absent, at 0.013 stops per episode in the rulebook regime and none
elsewhere (source:
`outputs/phase2-recalibration/ppo_baseline_100k_20260826T161335/summary_aggregated.csv`).
Diagnostics indicated learned avoidance rather than absent exploration, with only 2 pit
intents across 11,354 lap-final opportunities and a maximum pit-channel action of exactly
zero in seven of nine models (source: baseline `summary_pit_diagnostics.csv`). Crashes were
attributed predominantly to tyre overheating at mean tyre ages of 16.3–25.3 laps and mean
wear of 0.738–0.985 (source: baseline `summary_per_seed.csv`), indicating degradation-driven
rather than aggression-driven attrition.

Raising the discount factor to 0.999, with all else held constant, was associated with a
marked change in the rulebook regime: stops per episode rose from 0.013 to 0.580,
compliance from 0.013 to 0.567, mean return from −34.27 to +2.54, and mean time among
finished races fell from 5449.63 to 5058.73 seconds (sources: baseline and ablation
`summary_aggregated.csv`). Diagnostics showed pit intent expressed at 3.62 and 3.22 per cent
of lap-final opportunities in the two discovering seeds, corresponding to approximately one
stop per race, accompanied by soft–medium compound diversity in 43 and 42 of 50 episodes
respectively (sources: ablation `summary_pit_diagnostics.csv` and `summary_per_seed.csv`).
The safe regime exhibited the lowest learned demanded risk, −0.411 with an across-seed
standard deviation of 0.049, together with the highest finish rate (0.167) and the greatest
mean completed laps (28.43), and acquired pitting in one of three seeds without compound
diversity, consistent with its configuration providing a pit milestone but no compound
milestone. The unconstrained regime retained a zero-stop, high-risk profile, with the
highest mean demanded risk (0.680), the highest catastrophic rate (0.200) and the fewest
completed laps (20.98), which is consistent with its configured incentives and is therefore
interpreted as incentive-consistent rather than as a learning failure. Mean demanded risk
ordered across regimes as intended (0.680, 0.197, −0.411). Pit discovery was, however,
bimodal across seeds, occurring in two of three rulebook seeds and one of three safe seeds,
with correspondingly large across-seed standard deviations of 0.502 on stops and 32.29 on
return (source: ablation `summary_per_seed.csv`).

## 3. Discussion and transition to architecture comparison

Taken together, the evidence supports a narrow but useful conclusion. The reward
specification orders strategies in the intended manner for the policies tested, and the
recalibrated environment admits the strategies that those preferences describe; yet
acquiring the central strategic behaviour proved sensitive to a single optimisation
parameter. That a change in discount factor, with the objective and dynamics held constant,
was associated with the emergence of pitting and proxy compliance in the rulebook regime
suggests that the difficulty lay in long-horizon credit assignment rather than in reward
misspecification. This interpretation is consistent with the structure of the task, in which
the pit-lane cost is immediate whereas the benefit of fresh tyres accrues over many
subsequent laps, but it rests on a single ablation at one budget with three seeds and should
be regarded as suggestive rather than demonstrated.

Two methodological consequences follow. First, the reward specification and environment
dynamics were frozen before the comparative phase, so that algorithmic differences are not
confounded with concurrent changes to the task; the freeze is conditional on no further
defect being evidenced. Second, because the behaviour of principal interest was acquired by
some seeds and not others under an identical configuration, the discount factor must be set
explicitly and identically for every algorithm subsequently compared, and results must
report discovery rates alongside means. A mean value is compatible with uniform moderate
behaviour and with strongly bimodal behaviour, and only the latter was observed here.

The comparative phase can therefore proceed on a validated problem definition, subject to
an acknowledged asymmetry that no reward or environment change can remove: policy-gradient
candidates act through the continuous action space while value-based candidates act through
a discrete mapping of it. That asymmetry is a property of the algorithms' action interfaces
rather than of the task, and it should be reported as a limitation of the comparison rather
than treated as a confound to be eliminated.

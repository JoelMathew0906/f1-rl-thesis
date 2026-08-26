import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yaml
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any

from .track import load_or_build_silverstone_segments, TrackSegment


class RaceRegime(Enum):
    UNCONSTRAINED = auto()
    RULEBOOK = auto()
    SAFE = auto()


# Single source of truth for reward coefficients: the reward_regimes block
# of configs/configs_silverstone.yaml. Term semantics live in
# F1RaceEnv._compute_reward; only coefficients live in the YAML.
_REWARD_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "configs_silverstone.yaml"
)

REQUIRED_REWARD_KEYS = (
    "w_pace",
    "w_alive",
    "w_lap",
    "w_finish",
    "w_position_delta",
    "crash_penalty",
    "catastrophic_penalty",
    "w_risk_step",
    "risk_free_threshold",
    "w_wear_step",
    "wear_threshold",
    "pit_milestone_bonus",
    "compound_milestone_bonus",
    "finish_compliant_bonus",
    "finish_noncompliant_penalty",
    "over_pit_cap",
    "over_pit_penalty",
    "no_pit_step_penalty",
    "no_pit_grace_laps",
)

_reward_weights_cache: Optional[Dict[str, Dict[str, float]]] = None


def load_reward_weights(
    config_path: Path = _REWARD_CONFIG_PATH,
) -> Dict[str, Dict[str, float]]:
    """Load and strictly validate per-regime reward coefficients.

    Fails fast (naming the offending regime and key) on a missing config
    file, missing regime block, missing/unknown/non-numeric keys. There
    are deliberately no fallback defaults: the YAML is the only source
    of truth for reward coefficients.
    """
    global _reward_weights_cache
    if _reward_weights_cache is not None and config_path == _REWARD_CONFIG_PATH:
        return _reward_weights_cache

    if not config_path.exists():
        raise FileNotFoundError(f"Reward config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    regimes_cfg = cfg.get("reward_regimes") if isinstance(cfg, dict) else None
    if not isinstance(regimes_cfg, dict):
        raise ValueError(
            f"'reward_regimes' block missing or malformed in {config_path}"
        )

    weights: Dict[str, Dict[str, float]] = {}
    for regime in RaceRegime:
        name = regime.name.lower()
        block = regimes_cfg.get(name)
        if not isinstance(block, dict):
            raise ValueError(
                f"reward_regimes is missing regime '{name}' in {config_path}"
            )

        missing = [k for k in REQUIRED_REWARD_KEYS if k not in block]
        if missing:
            raise ValueError(
                f"reward_regimes['{name}'] is missing required keys "
                f"{missing} in {config_path}"
            )
        unknown = [k for k in block if k not in REQUIRED_REWARD_KEYS]
        if unknown:
            raise ValueError(
                f"reward_regimes['{name}'] has unknown keys {unknown} "
                f"in {config_path}"
            )

        parsed: Dict[str, float] = {}
        for key in REQUIRED_REWARD_KEYS:
            value = block[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"reward_regimes['{name}']['{key}'] must be numeric, "
                    f"got {value!r} in {config_path}"
                )
            parsed[key] = float(value)
        weights[name] = parsed

    if config_path == _REWARD_CONFIG_PATH:
        _reward_weights_cache = weights
    return weights


class F1RaceEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        regime: RaceRegime = RaceRegime.UNCONSTRAINED,
        n_laps: int = 52,
        seed: int | None = None,
        laps_data_path: str = "data/silverstone_2024_laps.csv",
    ):
        super().__init__()
        self.regime = regime
        self.n_laps = n_laps
        self.rng = np.random.default_rng(seed)
        self.laps_data_path = Path(laps_data_path)

        self.compound_to_idx = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 3,
            "WET": 4,
        }
        self.idx_to_compound = {v: k for k, v in self.compound_to_idx.items()}

        # load track segmentation and calibration from 2024 race data
        self.track_segments = load_or_build_silverstone_segments(year=2024)
        self.n_segments = len(self.track_segments)
        self.calibration = self._load_calibration_2024()

        # reward coefficients for this regime (validated YAML, fail-fast)
        self.reward_weights = load_reward_weights()[self.regime.name.lower()]

        obs_dim = 1 + 1 + 1 + 2 + 1 + 1 + 1 + 3 + 5 + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([0, 0, -1.0], dtype=np.float32),
            high=np.array([1, 4, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._reset_internal_state()

    # --- calibration from 2024 race data ---

    def _load_calibration_2024(self) -> Dict[str, Any]:
        fallback = {
            "base_lap_time": 92.0,
            "compound_offsets": {
                0: -1.2,
                1: 0.0,
                2: 0.8,
                3: 7.0,
                4: 11.0,
            },
            "deg_per_lap": {
                0: 0.18,
                1: 0.11,
                2: 0.07,
                3: 0.20,
                4: 0.25,
            },
            "typical_stint": {
                0: 12,
                1: 18,
                2: 24,
                3: 10,
                4: 10,
            },
            "pit_loss": 21.5,
        }

        if not self.laps_data_path.exists():
            return fallback

        try:
            df = pd.read_csv(self.laps_data_path)
            df = df.dropna(subset=["LapTime", "Compound", "TyreLife"]).copy()
            df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
            df = df[np.isfinite(df["LapTimeSeconds"])]
            df["Compound"] = df["Compound"].astype(str).str.upper()

            dry_df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
            if dry_df.empty:
                return fallback

            base_lap = float(dry_df["LapTimeSeconds"].median())

            compound_offsets: Dict[int, float] = {}
            deg_per_lap: Dict[int, float] = {}
            typical_stint: Dict[int, int] = {}

            for name, idx in self.compound_to_idx.items():
                cdf = df[df["Compound"] == name].copy()
                if cdf.empty:
                    compound_offsets[idx] = fallback["compound_offsets"][idx]
                    deg_per_lap[idx] = fallback["deg_per_lap"][idx]
                    typical_stint[idx] = fallback["typical_stint"][idx]
                    continue

                median_time = float(cdf["LapTimeSeconds"].median())
                compound_offsets[idx] = median_time - base_lap

                if cdf["TyreLife"].nunique() > 1:
                    x = cdf["TyreLife"].astype(float).values
                    y = cdf["LapTimeSeconds"].astype(float).values
                    slope = np.polyfit(x, y, 1)[0]
                    # allow a wider but still reasonable range for 2024 data
                    deg_per_lap[idx] = float(np.clip(slope, 0.01, 0.40))
                else:
                    deg_per_lap[idx] = fallback["deg_per_lap"][idx]

                if "Stint" in cdf.columns:
                    stint_lengths = cdf.groupby(["Driver", "Stint"]).size()
                    if len(stint_lengths) > 0:
                        typical_stint[idx] = int(np.clip(stint_lengths.median(), 8, 32))
                    else:
                        typical_stint[idx] = fallback["typical_stint"][idx]
                else:
                    typical_stint[idx] = fallback["typical_stint"][idx]

            return {
                "base_lap_time": base_lap,
                "compound_offsets": compound_offsets,
                "deg_per_lap": deg_per_lap,
                "typical_stint": typical_stint,
                "pit_loss": 21.5,
            }
        except Exception:
            return fallback

    # --- internal state ---

    def _reset_internal_state(self):
        self.current_lap = 0
        self.current_segment_idx = 0
        self.race_time = 0.0
        self.position = 10
        self.gap_ahead = 1.0
        self.gap_behind = 1.0
        self.tyre_compound = 1
        self.tyre_age = 0
        self.tyre_wear = 0.0
        self.fuel_level = 1.0
        self.track_status = 0
        self.made_pitstop = False
        self.pit_count = 0
        self.crashed = False
        self.catastrophic_event = False
        self.last_risk_level = 0.0

        # track which compounds have been used for rulebook compliance
        self.used_compounds: set[int] = {self.tyre_compound}

        # one-time milestone / anti-farming state for reward terms
        self._pit_milestone_paid = False
        self._compound_milestone_paid = False
        self._over_pits_charged = 0

        # logging of crash locations
        self.last_segment: Optional[TrackSegment] = None
        self.crash_log: list[dict[str, Any]] = []

        # reward component accumulators for per-episode decomposition
        self.reward_time_total = 0.0
        self.reward_risk_total = 0.0
        self.reward_crash_total = 0.0
        self.reward_pit_total = 0.0
        self.reward_compound_total = 0.0
        self.reward_compliance_total = 0.0
        self.reward_lap_completion_total = 0.0
        self.episode_return = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self._reset_internal_state()
        return self._get_obs(), {}

    # --- observations ---

    def _get_obs(self):
        lap_frac = self.current_lap / max(self.n_laps - 1, 1)
        race_time_norm = np.tanh(self.race_time / 5000.0)
        pos_norm = (self.position - 10) / 10.0
        gaps = np.tanh(np.array([self.gap_ahead, self.gap_behind]) / 10.0)
        tyre_age_norm = np.tanh(self.tyre_age / 20.0)
        tyre_wear = np.clip(self.tyre_wear, 0.0, 1.0)
        fuel_norm = np.clip(self.fuel_level, 0.0, 1.0)

        track_status_onehot = np.zeros(3, dtype=np.float32)
        track_status_onehot[self.track_status] = 1.0

        tyre_onehot = np.zeros(5, dtype=np.float32)
        tyre_onehot[self.tyre_compound] = 1.0

        risk_indicator = np.float32(self._estimate_risk())
        pit_count_norm = np.tanh(self.pit_count / 4.0)

        return np.concatenate(
            [
                np.array([lap_frac, race_time_norm, pos_norm], dtype=np.float32),
                gaps.astype(np.float32),
                np.array([tyre_age_norm, tyre_wear, fuel_norm], dtype=np.float32),
                track_status_onehot,
                tyre_onehot,
                np.array([risk_indicator, pit_count_norm], dtype=np.float32),
            ]
        )

    # --- stepping ---

    def step(self, action):
        pit_decision = int(action[0] > 0.5)
        tyre_choice = int(np.clip(round(action[1]), 0, 4))
        risk_level = float(np.clip(action[2], -1.0, 1.0))
        self.last_risk_level = risk_level

        segment = self.track_segments[self.current_segment_idx]
        self.last_segment = segment

        # progress along track
        self.current_segment_idx += 1
        completed_lap = False
        if self.current_segment_idx >= self.n_segments:
            self.current_segment_idx = 0
            self.current_lap += 1
            completed_lap = True

        lap_time, pos_delta, crash, sc_triggered = self._simulate_segment(
            segment, pit_decision, tyre_choice, risk_level
        )

        self.race_time += lap_time
        self.position = int(np.clip(self.position + pos_delta, 1, 20))

        if completed_lap:
            self.tyre_age += 1
            self.tyre_wear = np.clip(
                self.tyre_wear + self._tyre_wear_increment(risk_level), 0.0, 1.5
            )
            self.fuel_level = max(0.0, self.fuel_level - 1.0 / self.n_laps)

        self.track_status = 2 if sc_triggered else 0

        if pit_decision and completed_lap:
            self.made_pitstop = True
            self.pit_count += 1
            self.tyre_compound = tyre_choice
            self.tyre_age = 0
            self.tyre_wear = 0.0
            self.used_compounds.add(self.tyre_compound)

        self.crashed = crash
        if crash:
            self.catastrophic_event = self._is_catastrophic(crash, risk_level)
            self.crash_log.append(
                {
                    "lap": self.current_lap,
                    "segment_id": segment.id,
                    "corner_number": segment.corner_number,
                    "corner_name": segment.corner_name,
                    "segment_type": segment.segment_type,
                    "tyre_compound": self.idx_to_compound.get(
                        self.tyre_compound, str(self.tyre_compound)
                    ),
                    "tyre_age": self.tyre_age,
                    "tyre_wear": float(self.tyre_wear),
                    "risk_level": risk_level,
                    "crash_reason": self._crash_reason(segment, risk_level),
                }
            )

        terminated = self.crashed or (self.current_lap >= self.n_laps)
        truncated = False

        if terminated and self.regime == RaceRegime.RULEBOOK:
            if self.current_lap >= self.n_laps and self.pit_count < 1:
                self.position = 20
                self.race_time += 300.0

        reward = self._compute_reward(
            lap_time, pos_delta, crash, risk_level, terminated
        )

        # accumulate episode return after each step for logging
        self.episode_return += reward

        obs = self._get_obs()
        info = {
            "lap_time": lap_time,
            "position": self.position,
            "crash": crash,
            "sc_triggered": sc_triggered,
            "risk_level": risk_level,
            "pit_count": self.pit_count,
            "segment_id": segment.id,
            "corner_name": segment.corner_name,
            "corner_number": segment.corner_number,
            "segment_type": segment.segment_type,
            "segment_risk": self._segment_crash_prob(segment, risk_level),
        }
        return obs, reward, terminated, truncated, info

    # --- segment and crash modelling ---

    def _base_lap_time(self):
        return float(self.calibration["base_lap_time"])

    def _simulate_segment(
        self,
        segment: TrackSegment,
        pit_decision: int,
        tyre_choice: int,
        risk_level: float,
    ):
        base_lap = self._base_lap_time()
        base = base_lap / max(self.n_segments, 1)

        compound_pen = self.calibration["compound_offsets"].get(
            self.tyre_compound, 0.0
        ) / max(self.n_segments, 1)
        deg = self.calibration["deg_per_lap"].get(self.tyre_compound, 0.1)
        wear_pen = (deg * self.tyre_age) / max(self.n_segments, 1)
        fuel_pen = (2.5 * self.fuel_level) / max(self.n_segments, 1)

        risk_gain = -1.8 * max(0.0, risk_level) / max(self.n_segments, 1)
        overcaution_pen = 0.8 * max(0.0, -risk_level) / max(self.n_segments, 1)

        pit_loss = 0.0
        # approximate pit loss on pit entry straight at lap completion
        seg_type = str(segment.segment_type).lower()
        if pit_decision and seg_type == "straight":
            pit_loss = self.calibration["pit_loss"] / max(self.n_segments, 1)

        noise = self.rng.normal(0, 0.2)

        seg_time = (
            base
            + compound_pen
            + wear_pen
            + fuel_pen
            + risk_gain
            + overcaution_pen
            + pit_loss
            + noise
        )

        pos_delta = 0
        if seg_time < base - 0.15 and self.rng.random() < 0.05:
            pos_delta -= 1
        elif seg_time > base + 0.30 and self.rng.random() < 0.05:
            pos_delta += 1

        crash_prob = self._segment_crash_prob(segment, risk_level)
        crash = self.rng.random() < crash_prob

        sc_prob = 0.001 + 0.05 * float(crash)
        sc_triggered = self.rng.random() < sc_prob

        return float(seg_time), int(pos_delta), bool(crash), bool(sc_triggered)

    def _segment_crash_prob(self, segment: TrackSegment, risk_level: float) -> float:
        """Segment-level crash probability.

        Combines a small lap-wide baseline with corner/straight type,
        crude radius, tyre wear, and aggression relative to a simple
        risk envelope. This is still a simplified hazard model and
        not a full physics-based envelope.
        """
        base = 0.002

        seg_type = str(segment.segment_type).lower()
        is_corner = seg_type == "corner"

        if is_corner:
            base *= 8.0
        else:
            base *= 0.5

        # radius: tighter corners are riskier
        radius_term = 0.0
        if segment.approx_radius is not None and segment.approx_radius > 0:
            inv_r = 1.0 / max(segment.approx_radius, 1.0)
            radius_term = 0.015 * inv_r

        wear_term = 0.015 * self.tyre_wear

        # simple risk envelope: corners tolerate less explicit risk
        safe_risk = 0.3 if is_corner else 0.6
        envelope_excess = max(0.0, risk_level - safe_risk)
        exceed_term = 0.04 * envelope_excess

        old_tyre_term = (
            0.02
            if self.tyre_age
            > self.calibration["typical_stint"].get(self.tyre_compound, 18)
            else 0.0
        )

        base_prob = base + radius_term + wear_term + exceed_term + old_tyre_term
        return float(np.clip(base_prob, 0.0, 0.35))

    def _tyre_wear_increment(self, risk_level):
        base = {
            0: 0.050,
            1: 0.035,
            2: 0.025,
            3: 0.045,
            4: 0.055,
        }.get(self.tyre_compound, 0.03)
        aggression = 0.015 * max(0.0, risk_level)
        return base + aggression

    def _estimate_risk(self):
        return float(
            np.clip(
                0.6 * self.tyre_wear + 0.4 * max(0.0, self.last_risk_level), 0.0, 1.0
            )
        )

    def _is_catastrophic(self, crash: bool, risk_level: float):
        if not crash:
            return False
        return self.rng.random() < max(
            0.02, 0.20 * max(0.0, risk_level) + 0.10 * self.tyre_wear
        )

    def _crash_reason(self, segment: TrackSegment, risk_level: float) -> str:
        """Heuristic crash attribution.

        This is not a full physics model, but maps observed
        state to coarse failure modes for analysis.
        """
        if self.tyre_wear > 0.8:
            return "tyre_overheating"
        if (
            self.tyre_age
            > self.calibration["typical_stint"].get(self.tyre_compound, 18)
            and self.tyre_wear > 0.6
        ):
            return "insufficient_grip"
        seg_type = str(segment.segment_type).lower()
        if seg_type == "corner" and max(0.0, risk_level) > 0.7 and self.tyre_wear > 0.4:
            return "combined_load_exceedance"
        if max(0.0, risk_level) > 0.8:
            return "speed_exceedance"
        return "stochastic_incident"

    def _compute_reward(self, seg_time, pos_delta, crash, risk_level, terminated):
        """Compute reward from the per-regime coefficients in
        configs/configs_silverstone.yaml (see load_reward_weights).

        One generic term schema is shared by all regimes; regimes differ
        only in coefficients. Terms:
          pace           w_pace * (base_seg_time - seg_time) per step —
                         rewards speed/progress only; risk is never paid
                         directly and helps solely through its effect on
                         lap time versus crash probability
          position       w_position_delta * (-pos_delta) on position change
          alive          +w_alive per surviving step
          lap            +w_lap per completed lap
          finish         +w_finish for completing all laps without crash
          crash          -crash_penalty (- catastrophic_penalty extra)
          risk           -w_risk_step * max(0, risk - risk_free_threshold)
          wear           -w_wear_step * max(0, tyre_wear - wear_threshold)
          pit milestone  +pit_milestone_bonus once, on first effective pit
          compound       +compound_milestone_bonus once, when >=2 dry
                         compounds used in a dry race
          over-pit       -over_pit_penalty per pit beyond over_pit_cap,
                         charged at the offending pit (anti-farming)
          no-pit drip    -no_pit_step_penalty per step once past
                         no_pit_grace_laps without a pit; kept smaller
                         than w_alive so survival stays net positive
          compliance     settled at race *finish* only: crashed episodes
                         settle no compliance, because the rule applies to
                         classified finishers and settling at crash would
                         reward crashing early to escape the penalty
        """
        w = self.reward_weights

        # Pace baseline includes the deterministic, policy-independent fuel
        # penalty so the pace term is centred and does not tax mere survival.
        base_seg = (
            self._base_lap_time() + 2.5 * self.fuel_level
        ) / max(self.n_segments, 1)

        # pace + position-change (accumulated as the "time" component)
        reward_time = w["w_pace"] * (base_seg - seg_time)
        reward_time += w["w_position_delta"] * (-pos_delta)

        # survival shaping, lap and finish bonuses
        reward_lap_completion = w["w_alive"]
        lap_completed = (self.current_segment_idx == 0 and not crash)
        if lap_completed:
            reward_lap_completion += w["w_lap"]
        finished = (
            terminated and self.current_lap >= self.n_laps and not self.crashed
        )
        if finished:
            reward_lap_completion += w["w_finish"]

        # one-sided risk and thresholded wear penalties
        reward_risk = -w["w_risk_step"] * max(
            0.0, risk_level - w["risk_free_threshold"]
        )
        reward_compound = -w["w_wear_step"] * max(
            0.0, float(self.tyre_wear) - w["wear_threshold"]
        )

        reward_crash = 0.0
        if crash:
            reward_crash -= w["crash_penalty"]
            if self.catastrophic_event:
                reward_crash -= w["catastrophic_penalty"]

        # one-time pit milestone: repeated pits earn nothing extra
        reward_pit = 0.0
        if not self._pit_milestone_paid and self.pit_count >= 1:
            reward_pit += w["pit_milestone_bonus"]
            self._pit_milestone_paid = True

        # over-pitting charged per excess pit at the pit itself
        over_pits = (
            max(0, self.pit_count - int(w["over_pit_cap"])) - self._over_pits_charged
        )
        if over_pits > 0:
            reward_pit -= w["over_pit_penalty"] * over_pits
            self._over_pits_charged += over_pits

        if self.pit_count == 0 and self.current_lap >= int(w["no_pit_grace_laps"]):
            reward_pit -= w["no_pit_step_penalty"]

        # compound diversity milestone (dry-compound rule; wet usage exempts)
        dry_indices = {
            self.compound_to_idx["SOFT"],
            self.compound_to_idx["MEDIUM"],
            self.compound_to_idx["HARD"],
        }
        used_dry = {c for c in self.used_compounds if c in dry_indices}
        used_wet = any(c not in dry_indices for c in self.used_compounds)
        if (
            not self._compound_milestone_paid
            and not used_wet
            and len(used_dry) >= 2
        ):
            reward_compound += w["compound_milestone_bonus"]
            self._compound_milestone_paid = True

        # terminal compliance settlement — finished races only
        reward_compliance = 0.0
        if finished:
            compliant = self.pit_count >= 1 and (used_wet or len(used_dry) >= 2)
            if compliant:
                reward_compliance += w["finish_compliant_bonus"]
            else:
                reward_compliance -= w["finish_noncompliant_penalty"]

        # final reward and accumulation of components
        reward = (
            reward_time
            + reward_lap_completion
            + reward_risk
            + reward_crash
            + reward_pit
            + reward_compound
            + reward_compliance
        )

        self.reward_time_total += reward_time
        self.reward_lap_completion_total += reward_lap_completion
        self.reward_risk_total += reward_risk
        self.reward_crash_total += reward_crash
        self.reward_pit_total += reward_pit
        self.reward_compound_total += reward_compound
        self.reward_compliance_total += reward_compliance

        return reward

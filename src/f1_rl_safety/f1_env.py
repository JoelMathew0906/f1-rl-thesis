import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Dict, Any

from .track import load_or_build_silverstone_segments, TrackSegment


class RaceRegime(Enum):
    UNCONSTRAINED = auto()
    RULEBOOK = auto()
    SAFE = auto()


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

        # logging of crash locations
        self.last_segment: Optional[TrackSegment] = None
        self.crash_log: list[dict[str, Any]] = []

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

        self.crashed = crash
        if crash:
            self.catastrophic_event = self._is_catastrophic(crash, risk_level)
            self.crash_log.append(
                {
                    "lap": self.current_lap,
                    "segment_id": segment.id,
                    "corner_number": segment.corner_number,
                    "corner_name": segment.corner_name,
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
        if pit_decision and segment.segment_type == "straight":
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
        # base scale from lap-wide calibration
        base = 0.002

        if segment.segment_type == "corner":
            base *= 8.0
        else:
            base *= 0.5

        # radius: tighter corners are riskier
        radius_term = 0.0
        if segment.approx_radius is not None and segment.approx_radius > 0:
            inv_r = 1.0 / max(segment.approx_radius, 1.0)
            radius_term = 0.015 * inv_r

        wear_term = 0.015 * self.tyre_wear
        aggressive_term = 0.035 * max(0.0, risk_level)
        old_tyre_term = (
            0.02
            if self.tyre_age
            > self.calibration["typical_stint"].get(self.tyre_compound, 18)
            else 0.0
        )

        base_prob = base + radius_term + wear_term + aggressive_term + old_tyre_term
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

    def _compute_reward(self, seg_time, pos_delta, crash, risk_level, terminated):
        # approximate lap-equivalent terms for reward shaping; still segment-based
        time_term = -seg_time / 100.0
        pos_term = (10 - self.position) / 10.0
        progress_term = -0.02 / max(self.n_segments, 1)

        reward = time_term + pos_term + progress_term

        if self.regime == RaceRegime.UNCONSTRAINED:
            reward += 0.40 * max(0.0, risk_level) / max(self.n_segments, 1)
            if crash:
                reward -= 8.0
            if self.catastrophic_event:
                reward -= 25.0
            return reward

        if self.regime == RaceRegime.RULEBOOK:
            if crash:
                reward -= 20.0
            if self.catastrophic_event:
                reward -= 60.0

            reward -= 1.0 * max(0.0, risk_level - 0.4) / max(self.n_segments, 1)

            if self.current_lap > int(0.6 * self.n_laps) and self.pit_count < 1:
                reward -= 3.0 / max(self.n_segments, 1)

            if terminated:
                if self.current_lap >= self.n_laps and self.pit_count < 1:
                    reward -= 300.0
                if self.pit_count > 3:
                    reward -= 20.0 * (self.pit_count - 3)

            return reward

        if self.regime == RaceRegime.SAFE:
            reward -= 3.0 * max(0.0, risk_level) / max(self.n_segments, 1)
            reward -= 0.5 * max(0.0, abs(risk_level) - 0.3) / max(self.n_segments, 1)

            if crash:
                reward -= 60.0
            if self.catastrophic_event:
                reward -= 200.0

            if self.pit_count > 3:
                reward -= 15.0 * (self.pit_count - 3)

            reward -= 3.0 * max(0.0, self.tyre_wear - 0.65) / max(
                self.n_segments, 1
            )

            if terminated and self.current_lap >= self.n_laps and self.pit_count < 1:
                reward -= 40.0

            return reward

        return reward

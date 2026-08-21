import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from enum import Enum, auto
from pathlib import Path


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
        data_path: str = "data/silverstone_2025_laps.csv",
    ):
        super().__init__()
        self.regime = regime
        self.n_laps = n_laps
        self.rng = np.random.default_rng(seed)
        self.data_path = Path(data_path)

        self.compound_to_idx = {
            "SOFT": 0,
            "MEDIUM": 1,
            "HARD": 2,
            "INTERMEDIATE": 3,
            "WET": 4,
        }
        self.idx_to_compound = {v: k for k, v in self.compound_to_idx.items()}

        self.calibration = self._load_calibration()

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

    def _load_calibration(self):
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

        if not self.data_path.exists():
            return fallback

        try:
            df = pd.read_csv(self.data_path)
            df = df.copy()
            df = df.dropna(subset=["LapTime", "Compound", "TyreLife"])
            df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
            df = df[np.isfinite(df["LapTimeSeconds"])]
            df["Compound"] = df["Compound"].astype(str).str.upper()

            dry_df = df[df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()
            if dry_df.empty:
                return fallback

            base_lap = float(dry_df["LapTimeSeconds"].median())

            compound_offsets = {}
            deg_per_lap = {}
            typical_stint = {}

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
                    deg_per_lap[idx] = float(np.clip(slope, 0.03, 0.35))
                else:
                    deg_per_lap[idx] = fallback["deg_per_lap"][idx]

                if "Stint" in cdf.columns:
                    stint_lengths = cdf.groupby(["Driver", "Stint"]).size()
                    if len(stint_lengths) > 0:
                        typical_stint[idx] = int(np.clip(stint_lengths.median(), 8, 28))
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

    def _reset_internal_state(self):
        self.current_lap = 0
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self._reset_internal_state()
        return self._get_obs(), {}

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

    def step(self, action):
        pit_decision = int(action[0] > 0.5)
        tyre_choice = int(np.clip(round(action[1]), 0, 4))
        risk_level = float(np.clip(action[2], -1.0, 1.0))
        self.last_risk_level = risk_level

        self.current_lap += 1

        lap_time, pos_delta, crash, sc_triggered = self._simulate_lap(
            pit_decision, tyre_choice, risk_level
        )

        self.race_time += lap_time
        self.position = int(np.clip(self.position + pos_delta, 1, 20))
        self.tyre_age += 1
        self.tyre_wear = np.clip(
            self.tyre_wear + self._tyre_wear_increment(risk_level), 0.0, 1.5
        )
        self.fuel_level = max(0.0, self.fuel_level - 1.0 / self.n_laps)
        self.track_status = 2 if sc_triggered else 0

        if pit_decision:
            self.made_pitstop = True
            self.pit_count += 1
            self.tyre_compound = tyre_choice
            self.tyre_age = 0
            self.tyre_wear = 0.0

        self.crashed = crash
        if crash:
            self.catastrophic_event = self._is_catastrophic(crash, risk_level)

        terminated = self.crashed or (self.current_lap >= self.n_laps)
        truncated = False

        if terminated and self.regime == RaceRegime.RULEBOOK:
            if self.current_lap >= self.n_laps and self.pit_count < 1:
                self.position = 20
                self.race_time += 300.0

        reward = self._compute_reward(lap_time, pos_delta, crash, risk_level, terminated)

        obs = self._get_obs()
        info = {
            "lap_time": lap_time,
            "position": self.position,
            "crash": crash,
            "sc_triggered": sc_triggered,
            "risk_level": risk_level,
            "pit_count": self.pit_count,
        }
        return obs, reward, terminated, truncated, info

    def _base_lap_time(self):
        return float(self.calibration["base_lap_time"])

    def _simulate_lap(self, pit_decision, tyre_choice, risk_level):
        base = self._base_lap_time()

        compound_pen = self.calibration["compound_offsets"].get(self.tyre_compound, 0.0)
        deg = self.calibration["deg_per_lap"].get(self.tyre_compound, 0.1)

        wear_pen = deg * self.tyre_age
        fuel_pen = 2.5 * self.fuel_level

        risk_gain = -1.8 * max(0.0, risk_level)
        overcaution_pen = 0.8 * max(0.0, -risk_level)

        pit_loss = 0.0
        if pit_decision:
            pit_loss = self.calibration["pit_loss"] + self.rng.normal(0, 0.8)

        noise = self.rng.normal(0, 0.6)

        lap_time = (
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
        if lap_time < base - 0.7 and self.rng.random() < 0.35:
            pos_delta -= 1
        elif lap_time > base + 2.0 and self.rng.random() < 0.30:
            pos_delta += 1

        base_crash = 0.002
        wear_risk = 0.015 * self.tyre_wear
        aggressive_risk = 0.035 * max(0.0, risk_level)
        old_tyre_risk = (
            0.02
            if self.tyre_age > self.calibration["typical_stint"].get(self.tyre_compound, 18)
            else 0.0
        )
        crash_prob = np.clip(base_crash + wear_risk + aggressive_risk + old_tyre_risk, 0.0, 0.35)
        crash = self.rng.random() < crash_prob

        sc_prob = 0.01 + 0.10 * float(crash)
        sc_triggered = self.rng.random() < sc_prob

        return float(lap_time), int(pos_delta), bool(crash), bool(sc_triggered)

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
        return float(np.clip(0.6 * self.tyre_wear + 0.4 * max(0.0, self.last_risk_level), 0.0, 1.0))

    def _is_catastrophic(self, crash: bool, risk_level: float):
        if not crash:
            return False
        return self.rng.random() < max(0.02, 0.20 * max(0.0, risk_level) + 0.10 * self.tyre_wear)

    def _compute_reward(self, lap_time, pos_delta, crash, risk_level, terminated):
        time_term = -lap_time / 100.0
        pos_term = (10 - self.position) / 10.0
        progress_term = -0.02

        reward = time_term + pos_term + progress_term

        if self.regime == RaceRegime.UNCONSTRAINED:
            reward += 0.40 * max(0.0, risk_level)
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

            reward -= 1.0 * max(0.0, risk_level - 0.4)

            if self.current_lap > int(0.6 * self.n_laps) and self.pit_count < 1:
                reward -= 3.0

            if terminated:
                if self.current_lap >= self.n_laps and self.pit_count < 1:
                    reward -= 300.0
                if self.pit_count > 3:
                    reward -= 20.0 * (self.pit_count - 3)

            return reward

        if self.regime == RaceRegime.SAFE:
            reward -= 3.0 * max(0.0, risk_level)
            reward -= 0.5 * max(0.0, abs(risk_level) - 0.3)

            if crash:
                reward -= 60.0
            if self.catastrophic_event:
                reward -= 200.0

            if self.pit_count > 3:
                reward -= 15.0 * (self.pit_count - 3)

            reward -= 3.0 * max(0.0, self.tyre_wear - 0.65)

            if terminated and self.current_lap >= self.n_laps and self.pit_count < 1:
                reward -= 40.0

            return reward

        return reward
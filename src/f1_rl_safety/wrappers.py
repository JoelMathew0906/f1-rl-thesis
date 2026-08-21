import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple

from .f1_env import F1RaceEnv, RaceRegime


class DiscreteF1ActionWrapper(gym.Env):
    """Environment wrapper exposing a discrete action space.

    Internally wraps F1RaceEnv and maps each discrete action index to a
    (pit_decision, tyre_choice, risk_level) triple in the underlying
    continuous action space.

    This is designed to work with Stable-Baselines3's DQN, which expects
    a Discrete action space and calls env.step() with scalar actions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        regime: RaceRegime = RaceRegime.UNCONSTRAINED,
        n_laps: int = 52,
        seed: int | None = None,
    ):
        super().__init__()
        self._env = F1RaceEnv(regime=regime, n_laps=n_laps, seed=seed)

        # Define a compact discrete action set.
        # Pit: 0 = no pit, 1 = pit
        # Tyre: 0 = SOFT, 1 = MEDIUM, 2 = HARD (dry-focused baseline)
        # Risk bins: [-1.0, -0.5, 0.0, 0.5, 1.0]
        self._risk_bins: List[float] = [-1.0, -0.5, 0.0, 0.5, 1.0]
        self._tyre_indices: List[int] = [0, 1, 2]
        self._pit_options: List[int] = [0, 1]

        self._action_map: List[Tuple[int, int, float]] = []
        for pit in self._pit_options:
            for tyre in self._tyre_indices:
                for risk in self._risk_bins:
                    self._action_map.append((pit, tyre, risk))

        # Expose discrete action space; observation space is inherited from
        # the underlying F1RaceEnv.
        self.action_space = spaces.Discrete(len(self._action_map))
        self.observation_space = self._env.observation_space

    def reset(self, seed: int | None = None, options=None):
        """Reset the underlying environment and return its observation.

        Seeds are passed through to F1RaceEnv to preserve reproducibility.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: int):
        """Map a discrete action index to the Box action and step F1RaceEnv."""
        pit_decision, tyre_choice, risk_level = self._action_map[int(action)]
        box_action = np.array(
            [float(pit_decision), float(tyre_choice), float(risk_level)],
            dtype=np.float32,
        )
        obs, reward, terminated, truncated, info = self._env.step(box_action)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

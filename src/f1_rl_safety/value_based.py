from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .wrappers import DiscreteF1ActionWrapper
from .f1_env import RaceRegime


class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete actions."""

    def __init__(self, input_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_sarsa(
    regime: RaceRegime,
    total_timesteps: int,
    seed: int,
    log_dir: Path,
    model_dir: Path,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 25_000.0,
):
    """On-policy SARSA training over DiscreteF1ActionWrapper.

    The training loop runs episodes until the cumulative number of
    environment steps reaches total_timesteps. A small MLP Q-network
    is trained with TD(0) SARSA updates.
    """

    device = torch.device("cpu")

    env = DiscreteF1ActionWrapper(regime=regime, seed=seed)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)

    # Epsilon-greedy schedule
    def epsilon(t: int) -> float:
        frac = min(t / epsilon_decay, 1.0)
        return epsilon_start + frac * (epsilon_end - epsilon_start)

    timesteps = 0

    # Prepare directories
    tensorboard_log = log_dir / "sarsa" / regime.name.lower() / f"seed_{seed}"
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    model_dir = model_dir / "sarsa" / regime.name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    while timesteps < total_timesteps:
        obs, _ = env.reset(seed=seed + timesteps)
        done = False

        # Initial action
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_values = q_net(obs_tensor.unsqueeze(0))[0]
        eps = epsilon(timesteps)
        if np.random.rand() < eps:
            action = np.random.randint(n_actions)
        else:
            action = int(torch.argmax(q_values).item())

        episode_steps = 0

        while not done and timesteps < total_timesteps:
            # Step environment
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            timesteps += 1
            episode_steps += 1

            # Next action (on-policy)
            next_obs_tensor = torch.as_tensor(
                next_obs, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                next_q_values = q_net(next_obs_tensor.unsqueeze(0))[0]
            eps_next = epsilon(timesteps)
            if np.random.rand() < eps_next:
                next_action = np.random.randint(n_actions)
            else:
                next_action = int(torch.argmax(next_q_values).item())

            # Compute TD target
            reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=device)
            q_value = q_net(obs_tensor.unsqueeze(0))[0, action]
            if done:
                target = reward_tensor
            else:
                next_q_value = q_net(next_obs_tensor.unsqueeze(0))[0, next_action]
                target = reward_tensor + gamma * next_q_value

            loss = (q_value - target.detach()) ** 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Move to next state/action
            obs_tensor = next_obs_tensor
            action = next_action

        # Episodes are short (52 laps), so we don't log per-episode stats here.

    # Save trained Q-network
    model_path = model_dir / f"sarsa_regime={regime.name.lower()}_seed={seed}_steps={total_timesteps}.pt"
    torch.save(q_net.state_dict(), model_path)

    env.close()

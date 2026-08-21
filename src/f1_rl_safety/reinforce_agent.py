from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .f1_env import F1RaceEnv, RaceRegime


class PolicyNetwork(nn.Module):
    """Policy network producing distributions over pit, tyre and risk.

    - Pit decision: Bernoulli over {no pit, pit}
    - Tyre choice: categorical over 5 compounds
    - Risk level: Gaussian over continuous scalar
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pit_head = nn.Linear(hidden_dim, 1)
        self.tyre_head = nn.Linear(hidden_dim, 5)
        self.risk_mean = nn.Linear(hidden_dim, 1)
        self.risk_log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        h = self.base(x)
        pit_logits = self.pit_head(h)          # shape (batch, 1)
        tyre_logits = self.tyre_head(h)        # shape (batch, 5)
        risk_mean = self.risk_mean(h)          # shape (batch, 1)
        # Clamp log-std to a reasonable range to avoid extreme variances
        risk_log_std = torch.clamp(
            self.risk_log_std.expand_as(risk_mean), min=-2.0, max=2.0
        )
        return pit_logits, tyre_logits, risk_mean, risk_log_std


def sample_action(pit_logits, tyre_logits, risk_mean, risk_log_std):
    """Sample pit/tyre/risk actions and return env-compatible Box action.

    All tensors are assumed to have a leading batch dimension of size 1.
    NaNs are defensively handled to keep the sampler stable.
    """
    # Defensive: replace any NaNs in logits with zeros
    pit_logits = torch.nan_to_num(pit_logits, nan=0.0)
    tyre_logits = torch.nan_to_num(tyre_logits, nan=0.0)
    risk_mean = torch.nan_to_num(risk_mean, nan=0.0)
    risk_log_std = torch.nan_to_num(risk_log_std, nan=0.0)

    # Bernoulli pit decision
    pit_prob = torch.sigmoid(pit_logits)          # (1, 1), in (0,1)
    pit_prob = torch.clamp(pit_prob, 1e-6, 1.0 - 1e-6)
    pit_dist = torch.distributions.Bernoulli(probs=pit_prob)
    pit_action = pit_dist.sample()                # (1, 1)

    # Categorical tyre choice
    tyre_dist = torch.distributions.Categorical(logits=tyre_logits)  # (1, 5)
    tyre_action = tyre_dist.sample()             # (1,)

    # Gaussian risk level
    risk_std = torch.exp(risk_log_std)           # (1, 1)
    risk_dist = torch.distributions.Normal(loc=risk_mean, scale=risk_std)
    risk_action = risk_dist.sample()             # (1, 1)

    # Joint log-prob
    log_prob = (
        pit_dist.log_prob(pit_action).squeeze(-1) +
        tyre_dist.log_prob(tyre_action) +
        risk_dist.log_prob(risk_action).squeeze(-1)
    )  # shape (1,)

    # Map to environment action space: [pit_decision, tyre_choice, risk_level]
    pit = pit_action.squeeze(-1).float()        # (1,)
    tyre = tyre_action.float()                  # (1,)
    risk = torch.tanh(risk_action.squeeze(-1))  # (1,)

    action = torch.stack([pit, tyre, risk], dim=-1)  # (1, 3)
    return action, log_prob.squeeze(-1)               # action (1, 3), log_prob (1,)


def train_reinforce(
    regime: RaceRegime,
    total_episodes: int,
    seed: int,
    log_dir: Path,
    model_dir: Path,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
):
    """Simple REINFORCE training on F1RaceEnv.

    Uses an episodic Monte-Carlo policy gradient with a shared policy
    network over pit, tyre and risk decisions.
    """

    device = torch.device("cpu")

    env = F1RaceEnv(regime=regime, seed=seed)
    obs_dim = env.observation_space.shape[0]

    policy = PolicyNetwork(obs_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    # Prepare directories
    tensorboard_log = log_dir / "reinforce" / regime.name.lower() / f"seed_{seed}"
    tensorboard_log.mkdir(parents=True, exist_ok=True)

    model_dir = model_dir / "reinforce" / regime.name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for episode in range(total_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_log_probs = []
        episode_rewards = []

        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            pit_logits, tyre_logits, risk_mean, risk_log_std = policy(obs_tensor.unsqueeze(0))
            action_tensor, log_prob = sample_action(
                pit_logits, tyre_logits, risk_mean, risk_log_std
            )

            # Convert to numpy for env.step: shape (3,)
            action = action_tensor.detach().cpu().numpy()[0]
            next_obs, reward, terminated, truncated, _info = env.step(action)

            done = terminated or truncated

            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            obs = next_obs

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=device)
        # Normalise only when we have more than one timestep to avoid degenerate std
        if returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-8
            )
        else:
            returns_tensor = returns_tensor - returns_tensor.mean()

        log_probs_tensor = torch.stack(episode_log_probs)
        loss = -(log_probs_tensor * returns_tensor).sum()

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to reduce risk of exploding weights/NaNs
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    # Save trained policy
    model_path = model_dir / f"reinforce_regime={regime.name.lower()}_seed={seed}_episodes={total_episodes}.pt"
    torch.save(policy.state_dict(), model_path)

    env.close()

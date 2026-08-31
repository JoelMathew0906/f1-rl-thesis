"""Lightweight, non-training check for deterministic REINFORCE evaluation.

Run directly (no pytest dependency required):

    .venv_f1/bin/python tests/test_reinforce_eval_determinism.py

This does not train, evaluate against F1RaceEnv, or touch checkpoints/logs.
It only exercises action construction from fixed, synthetic policy outputs.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1_rl_safety.f1_env import F1RaceEnv
from f1_rl_safety.reinforce_agent import select_action_deterministic


def _fixed_network_outputs():
    """Fixed synthetic PolicyNetwork-shaped outputs (batch size 1)."""
    pit_logits = torch.tensor([[0.73]])
    tyre_logits = torch.tensor([[0.1, -2.0, 3.5, -0.4, 1.2]])
    risk_mean = torch.tensor([[-0.35]])
    risk_log_std = torch.tensor([[0.0]])
    return pit_logits, tyre_logits, risk_mean, risk_log_std


def test_deterministic_repeatability():
    pit_logits, tyre_logits, risk_mean, risk_log_std = _fixed_network_outputs()

    action_1 = select_action_deterministic(pit_logits, tyre_logits, risk_mean, risk_log_std)
    action_2 = select_action_deterministic(pit_logits, tyre_logits, risk_mean, risk_log_std)

    assert torch.equal(action_1, action_2), (
        f"Deterministic action changed across calls: {action_1} vs {action_2}"
    )
    print("PASS: repeated deterministic calls produce identical actions")


def test_action_matches_expected_values():
    pit_logits, tyre_logits, risk_mean, risk_log_std = _fixed_network_outputs()
    action = select_action_deterministic(pit_logits, tyre_logits, risk_mean, risk_log_std)
    action_np = action.detach().cpu().numpy()[0]

    # pit_logits > 0 -> pit decision = 1
    assert action_np[0] == 1.0, f"Expected pit=1.0 (logit>0), got {action_np[0]}"
    # argmax(tyre_logits) = index 2 (value 3.5)
    assert action_np[1] == 2.0, f"Expected tyre=2.0 (argmax logit), got {action_np[1]}"
    # risk = tanh(risk_mean), not a sampled value
    expected_risk = np.tanh(-0.35)
    assert np.isclose(action_np[2], expected_risk, atol=1e-6), (
        f"Expected risk=tanh(risk_mean)={expected_risk}, got {action_np[2]}"
    )
    print("PASS: deterministic action matches highest-logit pit/tyre and risk mean")


def test_action_valid_for_env_action_space():
    env = F1RaceEnv()
    try:
        pit_logits, tyre_logits, risk_mean, risk_log_std = _fixed_network_outputs()
        action = select_action_deterministic(pit_logits, tyre_logits, risk_mean, risk_log_std)
        action_np = action.detach().cpu().numpy()[0]

        assert action_np.dtype == np.float32, f"Expected float32, got {action_np.dtype}"
        assert action_np.shape == env.action_space.shape, (
            f"Expected shape {env.action_space.shape}, got {action_np.shape}"
        )
        assert env.action_space.contains(action_np), (
            f"Action {action_np} is out of env.action_space bounds "
            f"{env.action_space.low}..{env.action_space.high}"
        )
        print("PASS: deterministic action has valid dtype/shape/bounds for F1RaceEnv")
    finally:
        env.close()


def test_no_sampling_on_deterministic_path():
    """Confirm the deterministic path never calls torch.distributions samplers."""
    calls = {"sample": 0}

    orig_bernoulli_sample = torch.distributions.Bernoulli.sample
    orig_categorical_sample = torch.distributions.Categorical.sample
    orig_normal_sample = torch.distributions.Normal.sample

    def _tracked(orig):
        def _wrapper(self, *args, **kwargs):
            calls["sample"] += 1
            return orig(self, *args, **kwargs)
        return _wrapper

    torch.distributions.Bernoulli.sample = _tracked(orig_bernoulli_sample)
    torch.distributions.Categorical.sample = _tracked(orig_categorical_sample)
    torch.distributions.Normal.sample = _tracked(orig_normal_sample)
    try:
        pit_logits, tyre_logits, risk_mean, risk_log_std = _fixed_network_outputs()
        select_action_deterministic(pit_logits, tyre_logits, risk_mean, risk_log_std)
        assert calls["sample"] == 0, (
            f"Deterministic path invoked a sampler {calls['sample']} time(s)"
        )
        print("PASS: no distribution sampling occurs on the deterministic evaluation path")
    finally:
        torch.distributions.Bernoulli.sample = orig_bernoulli_sample
        torch.distributions.Categorical.sample = orig_categorical_sample
        torch.distributions.Normal.sample = orig_normal_sample


def test_training_sample_action_still_stochastic():
    """Confirm sample_action (training) is untouched and still samples."""
    from f1_rl_safety.reinforce_agent import sample_action

    torch.manual_seed(0)
    pit_logits, tyre_logits, risk_mean, risk_log_std = _fixed_network_outputs()

    actions = set()
    for _ in range(20):
        action, log_prob = sample_action(pit_logits, tyre_logits, risk_mean, risk_log_std)
        assert log_prob.numel() == 1, f"Unexpected log_prob numel {log_prob.numel()}"
        actions.add(tuple(np.round(action.detach().cpu().numpy()[0], 4)))

    assert len(actions) > 1, (
        "sample_action produced identical actions across 20 draws; "
        "training must remain stochastic"
    )
    print(f"PASS: sample_action (training) remains stochastic ({len(actions)} distinct draws/20)")


if __name__ == "__main__":
    test_deterministic_repeatability()
    test_action_matches_expected_values()
    test_action_valid_for_env_action_space()
    test_no_sampling_on_deterministic_path()
    test_training_sample_action_still_stochastic()
    print("\nALL CHECKS PASSED")

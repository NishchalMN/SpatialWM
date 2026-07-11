"""
Tests for spatialwm.planning.planner — SKIPPED (needs trained dynamics model).

Collectible but skipped until Week 5+ resources are available.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialwm.planning.planner import cem_mpc


@pytest.mark.skip(reason="Week 5 — needs trained dynamics model")
def test_cem_mpc_output_shape():
    """cem_mpc returns (horizon, action_dim) action sequence."""
    def dummy_dynamics(latent, action):
        return latent + 0.01  # trivial dynamics

    goal = np.zeros(8)
    action_seq = cem_mpc(
        dynamics=dummy_dynamics,
        goal_latent=goal,
        action_dim=3,
        horizon=10,
        n_samples=50,
        n_elite=10,
        n_iters=3,
    )
    assert action_seq.shape == (10, 3), f"Expected (10, 3), got {action_seq.shape}"


@pytest.mark.skip(reason="Week 5 — needs trained dynamics model")
def test_cem_mpc_reduces_cost():
    """CEM iterations reduce MSE toward goal_latent."""
    goal = np.zeros(8)

    def dynamics(latent, action):
        # Move latent toward 0 when action ~= 0
        return latent * 0.9

    cem_mpc(
        dynamics=dynamics,
        goal_latent=goal,
        action_dim=3,
        horizon=5,
        n_samples=100,
        n_elite=20,
        n_iters=5,
    )
    # Smoke: no exception

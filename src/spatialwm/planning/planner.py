"""CEM-based MPC planner for latent-space control."""

import numpy as np


def cem_mpc(
    dynamics,
    goal_latent,
    action_dim: int,
    horizon: int,
    n_samples: int = 1000,
    n_elite: int = 100,
    n_iters: int = 5,
) -> np.ndarray:
    """CEM/MPC in latent space.

    Sample action seqs -> roll latent dynamics -> score MSE vs goal_latent ->
    refit elite Gaussian -> execute first action.
    
    Args:
        dynamics: Callable that takes (latent, action) and returns next latent state
        goal_latent: Target latent state to reach
        action_dim: Dimensionality of action space
        horizon: Planning horizon (number of timesteps)
        n_samples: Number of action sequences to sample per iteration
        n_elite: Number of elite samples to use for refitting
        n_iters: Number of CEM iterations
    
    Returns:
        Best action sequence of shape (horizon, action_dim)
    """
    raise NotImplementedError


def _demo():
    """Demo CEM-MPC with dummy dynamics."""
    # Dummy dynamics that raises to show it gets called
    def dummy_dynamics(latent, action):
        raise RuntimeError("Dummy dynamics called")
    
    goal = np.zeros(8)
    action_seq = cem_mpc(
        dummy_dynamics,
        goal,
        action_dim=3,
        horizon=10,
        n_samples=100,
        n_elite=10,
        n_iters=3,
    )
    print(f"Best action sequence shape: {action_seq.shape}")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        _demo()

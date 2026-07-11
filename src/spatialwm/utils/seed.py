"""Global seeding for reproducibility."""

from __future__ import annotations


def seed_all(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (incl. mps/cuda) with ``seed``.

    Torch is imported inside the function so this module imports without torch.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

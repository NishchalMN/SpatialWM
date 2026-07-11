"""Device selection helper."""

from __future__ import annotations


def get_device(pref: str = "auto") -> str:
    """Return a torch device string: ``"mps" | "cuda" | "cpu"``.

    ``pref="auto"`` prefers mps -> cuda -> cpu based on availability.
    Any explicit ``pref`` is returned verbatim (caller's responsibility).
    Torch is imported inside the function so importing this module never
    requires torch to be installed.
    """
    import torch

    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

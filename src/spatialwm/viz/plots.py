"""Visualization utilities for analysis and debugging."""

import os

import numpy as np


def plot_singular_spectrum(singular_values, ax=None, label: str | None = None):
    """Plot sorted singular values to show representation collapse.
    
    Useful for analyzing rank deficiency in learned representations.
    
    Args:
        singular_values: Array of singular values
        ax: Matplotlib axes (creates new figure if None)
        label: Optional label for the plot
    
    Returns:
        Matplotlib axes object
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    sorted_sv = np.sort(singular_values)[::-1]
    ax.semilogy(sorted_sv, marker='o', markersize=3, label=label)
    ax.set_xlabel('Index')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title('Singular Value Spectrum')
    ax.grid(True, alpha=0.3)
    
    if label is not None:
        ax.legend()
    
    return ax


def save_fig(fig, path: str) -> None:
    """Save figure to file with proper directory creation.
    
    Args:
        fig: Matplotlib figure
        path: Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')

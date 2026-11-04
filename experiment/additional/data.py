"""Stream and prior-count data generation for the additional experiments."""

from __future__ import annotations

import numpy as np


def make_theta_nonuniform(n_bins: int, sharpness: float = 1.6) -> np.ndarray:
    """Create a smooth non-uniform bin distribution over [0, 1]."""
    x = np.linspace(0.0, 1.0, n_bins, endpoint=False) + 0.5 / n_bins
    # Symmetric, mildly U-shaped profile to avoid overfitting to one edge.
    raw = 1.0 + sharpness * np.abs(x - 0.5)
    theta = raw / raw.sum()
    return theta.astype(np.float64)


def build_pre_counts(theta: np.ndarray, m_pre: int, mode: str) -> np.ndarray:
    """Construct pre-shift bin counts A for aligned/misaligned cases."""
    n_bins = len(theta)
    if mode == "aligned":
        p = theta
    elif mode == "misaligned":
        # Intentionally anti-aligned but not degenerate (keeps MC variance sane).
        p = 1.0 / np.maximum(theta, 1e-12)
        p = p / p.sum()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rng_local = np.random.default_rng(2024 if mode == "aligned" else 2025)
    A = rng_local.multinomial(n=m_pre, pvals=p)
    return A.astype(np.int64)


def stream_multistep_piecewise(
    rng: np.random.Generator,
    n: int,
    onset_step: int,
    mid_step: int,
    max_step: int,
    k_levels: tuple[float, float, float, float] = (1.0, 0.85, 0.65, 0.45),
) -> np.ndarray:
    """Piecewise-constant PIT stream with several step changes in drift intensity.

    A smaller symmetric-Beta shape parameter implies stronger non-uniformity.
    """
    idx = np.arange(1, n + 1)
    k = np.full(n, k_levels[0], dtype=np.float64)
    k[idx >= onset_step] = k_levels[1]
    k[idx >= mid_step] = k_levels[2]
    k[idx >= max_step] = k_levels[3]

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rng.beta(k[i], k[i])
    return np.clip(out, 1e-12, 1.0 - 1e-12)

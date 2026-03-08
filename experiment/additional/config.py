"""Experiment configuration for PITMonitor additional verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for the verification experiments."""

    seed: int = 42
    alpha: float = 0.05
    n_bins: int = 100

    # Proposition verification parameters
    m_pre: int = 2_500
    n_grid_max: int = 2_500
    n_grid_points: int = 20
    n_mc_formula: int = 10_000

    # Multi-step drift localization parameters
    n_trials_localization: int = 10_000
    n_steps_localization: int = 5_000
    onset_step: int = 2_000
    mid_step: int = 2_900
    max_step: int = 3_700

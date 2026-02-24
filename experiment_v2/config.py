"""Experiment v2 configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """All experiment parameters in one place.

    The FriedmanDrift stream is divided into three contiguous segments:
        [0, n_train)                           → model training data
        [n_train, n_train + n_stable)          → monitoring, pre-drift
        [n_train + n_stable, n_train + n_total) → monitoring, post-drift

    The first drift position is set to ``n_train + n_stable`` so that
    the model is always trained entirely on pre-drift data.
    """

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Data geometry ────────────────────────────────────────────────
    n_train: int = 5_000  # samples for model training
    n_stable: int = 2_500  # pre-drift monitoring samples
    n_post: int = 2_500  # post-drift monitoring samples

    # ── Drift scenarios ──────────────────────────────────────────────
    # Each scenario is a (drift_type, transition_window) pair.
    # FriedmanDrift position is computed from n_train + n_stable.
    drift_scenarios: tuple[tuple[str, int], ...] = (
        ("gra", 0),  # Global Recurring Abrupt
        ("gsg", 1000),  # Global Slow Gradual (1000 sample transition)
        ("lea", 0),  # Local Expanding Abrupt
    )

    # ── PITMonitor settings ──────────────────────────────────────────
    alpha: float = 0.05
    n_monitor_bins: int = 10

    # ── Monte-Carlo trials ───────────────────────────────────────────
    n_trials: int = 10_000
    max_workers: int = 8

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "out"

    # ── Derived helpers (not frozen fields, just methods) ────────────
    @property
    def n_total(self) -> int:
        return self.n_train + self.n_stable + self.n_post

    @property
    def drift_index(self) -> int:
        """Absolute sample index where the first drift occurs."""
        return self.n_train + self.n_stable

    @property
    def out_path(self) -> Path:
        return Path(self.output_dir)

    def positions_for(self, drift_type: str) -> tuple[int, ...]:
        """Return FriedmanDrift ``position`` tuple for the given drift type.

        Unused drift points are pushed far beyond the data window so they
        never fire during the experiment.
        """
        dp = self.drift_index
        far = dp + self.n_post + 100_000  # safely beyond the stream
        if drift_type == "gra":  # needs exactly 2 positions
            return (dp, far)
        elif drift_type == "gsg":  # needs exactly 2 positions
            return (dp, far)
        elif drift_type == "lea":  # needs exactly 3 positions
            return (dp, far, far + 1)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type!r}")

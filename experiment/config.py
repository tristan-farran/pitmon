"""Experiment configuration.

All experiment hyperparameters live here so that run.py, experiment.py,
and train_model.py all share one source of truth.

Stream layout
-------------
The FriedmanDrift stream is divided into three contiguous segments:

    [0, n_train)                         → model training data
    [n_train, n_train + n_stable)        → monitoring, pre-drift (null)
    [n_train + n_stable, n_total)        → monitoring, post-drift

The drift point is placed at index ``drift_index = n_train + n_stable``
so the model is always trained entirely on pre-drift data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """All experiment parameters in one immutable configuration object.

    Parameters
    ----------
    seed : int
        Master RNG seed for reproducibility.
    epochs : int
        Training epochs for the ProbabilisticMLP.
    lr : float
        Adam learning rate for training.
    n_train : int
        Number of pre-drift samples used to train the model.
    n_stable : int
        Length of the pre-drift monitoring window (used for FPR estimation).
    n_post : int
        Length of the post-drift monitoring window (used for TPR estimation).
    drift_scenarios : tuple of (str, int) pairs
        Each entry is (drift_type, transition_window).  Supported drift types
        are those accepted by ``river.datasets.synth.FriedmanDrift``.
    alpha : float
        Nominal false alarm rate passed to PITMonitor (and used as the FPR
        reference line in plots).
    n_bins_list : tuple of int
        PITMonitor histogram sizes to sweep over.  The first entry is used for
        the primary experiment; all entries are used in the n_bins sensitivity
        plot.
    n_trials : int
        Number of Monte-Carlo trials per drift scenario.
    max_workers : int
        Thread pool size for parallel trial execution.
    output_dir : str
        Directory for all output artefacts (JSON results, plots, model bundle).
    """

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Training ─────────────────────────────────────────────────────
    epochs: int = 10_000
    lr: float = 3e-4

    # ── Data geometry ────────────────────────────────────────────────
    n_train: int = 10_000  # samples for model training
    n_stable: int = 2_500  # pre-drift monitoring samples
    n_post: int = 2_500  # post-drift monitoring samples

    # ── Drift scenarios ──────────────────────────────────────────────
    # Each element is a (drift_type, transition_window) pair.
    drift_scenarios: Tuple = (
        ("gra", 0),  # Global Recurring Abrupt
        ("gsg", 500),  # Global Slow Gradual
        ("lea", 0),  # Local Expanding Abrupt
    )

    # ── Detector settings ──────────────────────────────────────────
    alpha: float = 0.05  # PITMonitor
    delta: float = 0.05  # ADWIN

    # Bin sizes for the sensitivity sweep. The first value is used for the
    # main experiment; all values appear in the n_bins sensitivity plot.
    n_bins_list: Tuple = (100,)

    # ── Monte-Carlo trials ───────────────────────────────────────────
    n_trials: int = 10_000
    max_workers: int = 8

    # ── Output ───────────────────────────────────────────────────────
    output_dir: str = "out"

    # ── Derived helpers ──────────────────────────────────────────────

    @property
    def n_monitor_bins(self) -> int:
        """Default (primary) number of PITMonitor histogram bins."""
        return self.n_bins_list[0]

    @property
    def n_total(self) -> int:
        """Total stream length (train + stable + post)."""
        return self.n_train + self.n_stable + self.n_post

    @property
    def drift_index(self) -> int:
        """Absolute sample index where the first drift occurs."""
        return self.n_train + self.n_stable

    @property
    def out_path(self) -> Path:
        """Resolved output directory as a Path object."""
        return Path(self.output_dir)

    @property
    def bundle_path(self) -> Path:
        """Path to the saved model bundle produced by train_model.py."""
        return self.out_path / "model.pkl"

    def positions_for(self, drift_type: str) -> Tuple:
        """Return the FriedmanDrift ``position`` tuple for *drift_type*.

        Unused drift points are pushed far beyond the data window so they
        never fire during the experiment.

        Parameters
        ----------
        drift_type : str
            One of ``'gra'``, ``'gsg'``, or ``'lea'``.

        Returns
        -------
        tuple of int
        """
        dp = self.drift_index
        far = dp + self.n_post + 100_000  # safely beyond the stream
        if drift_type == "gra":  # expects exactly 2 positions
            return (dp, far)
        elif drift_type == "gsg":  # expects exactly 2 positions
            return (dp, far)
        elif drift_type == "lea":  # expects exactly 3 positions
            return (dp, far, far + 1)
        else:
            raise ValueError(f"Unknown drift_type: {drift_type!r}")

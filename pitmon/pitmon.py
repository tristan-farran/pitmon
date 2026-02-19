import json
import bisect
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Union


@dataclass
class Alarm:
    """Result of each update."""

    triggered: bool
    time: int
    evidence: float
    threshold: float

    def __bool__(self) -> bool:
        return self.triggered


class PITMonitor:
    """
    Monitor for detecting calibration changes in probabilistic predictions.

    Detects when the distribution of PITs changes over time. Under stable
    calibration (even if imperfect), PITs are exchangeable and no alarm fires.
    When calibration changes, exchangeability breaks and evidence accumulates.

    Parameters
    ----------
    alpha : float, default=0.05
        Anytime-valid false alarm rate: P(ever alarm | H0) ≤ α

    n_bins : int, default=10
        Histogram bins for density estimation. More bins = faster adaptation
        but more variance. 10 is the MDL-optimal choice for most settings.
    """

    def __init__(self, alpha: float = 0.05, n_bins: int = 10):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 2 <= n_bins <= 100:
            raise ValueError("n_bins must be in [2, 100]")

        self.alpha = alpha
        self.n_bins = n_bins
        self.threshold = 1.0 / alpha

        # State
        self.t = 0
        self._sorted_pits: List[float] = []
        self._bin_counts = np.ones(n_bins)  # Laplace prior (pseudocount = 1)

        # Evidence tracking
        self._M = 0.0  # Mixture e-process
        self._history: List[Tuple[float, float, float]] = []  # (pit, pval, M)

        # Alarm
        self.alarm_triggered = False
        self.alarm_time: Optional[int] = None

    def update(self, pit: float) -> Alarm:
        """
        Process one PIT value.

        Parameters
        ----------
        pit : float
            Probability integral transform in [0, 1].

        Returns
        -------
        Alarm
            Use as boolean; True if alarm triggered.
        """
        if self.alarm_triggered:
            return Alarm(True, self.alarm_time, self._M, self.threshold)

        if not 0 <= pit <= 1:
            raise ValueError(f"PIT {pit} not in [0, 1]")

        self.t += 1
        bisect.insort(self._sorted_pits, pit)

        # p_t = (rank of u_t among u_1,...,u_t) / (t+1)
        # Under exchangeability, p_t ~ Uniform(0,1).
        # For exact uniformity randomize within ties
        left = bisect.bisect_left(self._sorted_pits, pit)
        right = bisect.bisect_right(self._sorted_pits, pit)
        U = np.random.uniform(0, right - left)
        p = (left + U) / self.t  # Maps to (0, 1)

        # Clamp to (epsilon, 1-epsilon) to avoid numerical issues
        p = np.clip(p, 1e-10, 1 - 1e-10)

        # First observation: initialize but no test
        if self.t == 1:
            self._history.append((pit, p, 0.0))
            return Alarm(False, self.t, 0.0, self.threshold)

        # e_t = estimated density at p_t (from histogram of past p-values)
        bin_idx = min(int(p * self.n_bins), self.n_bins - 1)
        density = self._bin_counts[bin_idx] / self._bin_counts.sum()
        e = density * self.n_bins  # Scale to integrate to 1
        self._bin_counts[bin_idx] += 1

        # Mixture e-process
        s = self.t - 1
        w = 1.0 / (s * (s + 1))
        self._M = e * (self._M + w)

        self._history.append((pit, p, self._M))

        if self._M >= self.threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t

        return Alarm(self.alarm_triggered, self.t, self._M, self.threshold)

    def update_with_cdf(self, cdf: Callable[[float], float], y: float) -> Alarm:
        """Convenience: compute PIT as cdf(y) and process it."""
        return self.update(cdf(y))

    @property
    def evidence(self) -> float:
        """Current evidence against exchangeability."""
        return self._M

    @property
    def pits(self) -> np.ndarray:
        """All observed PITs in order."""
        return np.array([h[0] for h in self._history])

    @property
    def pvalues(self) -> np.ndarray:
        """All conformal p-values."""
        return np.array([h[1] for h in self._history])

    def __repr__(self) -> str:
        """Detailed representation."""
        status = "ALARM" if self.alarm_triggered else "monitoring"
        return (
            f"PITMonitor(alpha={self.alpha}, n_bins={self.n_bins}, "
            f"t={self.t}, status={status}, evidence={self._M:.2f})"
        )

    def __str__(self) -> str:
        """Human-readable status."""
        if self.t == 0:
            return f"PITMonitor: Not started (α={self.alpha})"
        status = (
            f"ALARM at t={self.alarm_time}" if self.alarm_triggered else "monitoring"
        )
        return (
            f"PITMonitor: {status} | "
            f"t={self.t}, evidence={self._M:.2f}, threshold={self.threshold:.0f}"
        )

    def changepoint(self) -> Optional[int]:
        """
        Estimate when calibration started changing.

        Uses maximum likelihood: finds t that maximizes evidence for
        "change occurred at time t".

        Returns
        -------
        int or None
            Estimated changepoint, or None if no alarm yet.
        """
        if not self.alarm_triggered or self.t < 3:
            return None

        # Compute evidence for each possible changepoint
        pvals = self.pvalues[1:]
        n = len(pvals)

        # Score each split: KL divergence of empirical distribution from uniform
        scores = []
        for k in range(max(1, n // 10), min(n, self.alarm_time or n)):
            after = pvals[k:]
            if len(after) < 5:
                continue

            counts, _ = np.histogram(after, bins=self.n_bins, range=(0, 1))
            counts = counts + 1  # Laplace smoothing
            freq = counts / counts.sum()

            # KL from uniform (negative entropy + log(n_bins))
            kl = np.sum(freq * np.log(freq * self.n_bins + 1e-10))
            scores.append((k + 1, kl))

        if not scores:
            return 1

        return max(scores, key=lambda x: x[1])[0]

    def summary(self) -> dict:
        """
        Get summary statistics of the monitoring session.

        Returns
        -------
        dict
            Dictionary with monitoring statistics including:
            - t: current time step
            - alarm_triggered: whether alarm has been triggered
            - alarm_time: when alarm was triggered (if applicable)
            - evidence: current evidence value
            - threshold: alarm threshold
            - changepoint: estimated changepoint (if alarm triggered)
            - calibration_score: KS statistic measuring deviation from uniformity
        """
        summary = {
            "t": self.t,
            "alarm_triggered": self.alarm_triggered,
            "alarm_time": self.alarm_time,
            "evidence": self._M,
            "threshold": self.threshold,
            "changepoint": self.changepoint() if self.alarm_triggered else None,
        }

        if self.t > 0:
            pits = self.pits
            summary["calibration_score"] = self.calibration_score()
        else:
            summary["calibration_score"] = None

        return summary

    def calibration_score(self) -> float:
        """
        Compute calibration score (Kolmogorov-Smirnov statistic).

        Measures maximum deviation of empirical CDF from uniform.
        Returns value in [0, 1], where 0 = perfect calibration.

        Returns
        -------
        float
            KS statistic: max|ECDF - Uniform|
        """
        if self.t == 0:
            return 0.0

        pits_sorted = np.sort(self.pits)
        uniform_cdf = np.linspace(1 / self.t, 1, self.t)
        ks_stat = np.max(np.abs(pits_sorted - uniform_cdf))
        return float(ks_stat)

    def get_status(self) -> str:
        """
        Get current monitoring status.

        Returns
        -------
        str
            One of: 'not_started', 'monitoring', 'alarm'
        """
        if self.t == 0:
            return "not_started"
        return "alarm" if self.alarm_triggered else "monitoring"

    def reset(self):
        """Reset to initial state."""
        self.t = 0
        self._sorted_pits = []
        self._bin_counts = np.ones(self.n_bins)
        self._M = 0.0
        self._history = []
        self.alarm_triggered = False
        self.alarm_time = None

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save monitor state to file.

        Parameters
        ----------
        filepath : str or Path
            Path to save file. Extension determines format:
            - .pkl: pickle format (preserves all state)
            - .json: JSON format (human-readable, limited precision)
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            state = {
                "alpha": self.alpha,
                "n_bins": self.n_bins,
                "t": self.t,
                "sorted_pits": self._sorted_pits,
                "bin_counts": self._bin_counts.tolist(),
                "M": self._M,
                "history": [
                    {"pit": float(pit), "pval": float(pval), "M": float(M)}
                    for pit, pval, M in self._history
                ],
                "alarm_triggered": self.alarm_triggered,
                "alarm_time": self.alarm_time,
            }
            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)
        else:
            # Default to pickle for full fidelity
            state = {
                "alpha": self.alpha,
                "n_bins": self.n_bins,
                "threshold": self.threshold,
                "t": self.t,
                "_sorted_pits": self._sorted_pits,
                "_bin_counts": self._bin_counts,
                "_M": self._M,
                "_history": self._history,
                "alarm_triggered": self.alarm_triggered,
                "alarm_time": self.alarm_time,
            }
            with open(filepath, "wb") as f:
                pickle.dump(state, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PITMonitor":
        """
        Load monitor state from file.

        Parameters
        ----------
        filepath : str or Path
            Path to saved state file.

        Returns
        -------
        PITMonitor
            Restored monitor instance.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                state = json.load(f)

            monitor = cls(alpha=state["alpha"], n_bins=state["n_bins"])
            monitor.t = state["t"]
            monitor._sorted_pits = state["sorted_pits"]
            monitor._bin_counts = np.array(state["bin_counts"])
            monitor._M = state["M"]
            monitor._history = [(h["pit"], h["pval"], h["M"]) for h in state["history"]]
            monitor.alarm_triggered = state["alarm_triggered"]
            monitor.alarm_time = state["alarm_time"]
        else:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            monitor = cls(alpha=state["alpha"], n_bins=state["n_bins"])
            monitor.threshold = state["threshold"]
            monitor.t = state["t"]
            monitor._sorted_pits = state["_sorted_pits"]
            monitor._bin_counts = state["_bin_counts"]
            monitor._M = state["_M"]
            monitor._history = state["_history"]
            monitor.alarm_triggered = state["alarm_triggered"]
            monitor.alarm_time = state["alarm_time"]

        return monitor

    def export_data(self) -> dict:
        """
        Export all monitoring data.

        Returns
        -------
        dict
            Dictionary containing:
            - metadata: monitor configuration and status
            - timeseries: arrays of PITs, p-values, evidence over time
            - statistics: summary statistics
        """
        metadata = {
            "alpha": self.alpha,
            "n_bins": self.n_bins,
            "threshold": self.threshold,
            "t": self.t,
            "alarm_triggered": self.alarm_triggered,
            "alarm_time": self.alarm_time,
        }

        timeseries = {}
        if self.t > 0:
            timeseries = {
                "time": list(range(1, self.t + 1)),
                "pits": self.pits.tolist(),
                "pvalues": self.pvalues.tolist(),
                "evidence": [h[2] for h in self._history],
            }

        return {
            "metadata": metadata,
            "timeseries": timeseries,
            "statistics": self.summary(),
        }

    def plot(self, figsize: Tuple[float, float] = (12, 4)) -> Optional[object]:
        """Create diagnostic plot.

        Parameters
        ----------
        figsize : tuple of float, default=(12, 4)
            Figure size (width, height) in inches.

        Returns
        -------
        Figure or None
            Matplotlib figure object, or None if insufficient data.
        """
        if self.t < 2:
            print("Need ≥2 observations")
            return

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        pvals = self.pvalues
        evidence = np.array([h[2] for h in self._history])
        times = np.arange(1, self.t + 1)
        cp = self.changepoint()

        # 1: Evidence
        ax = axes[0]
        ax.semilogy(times, np.maximum(evidence, 1e-10), "steelblue", lw=2)
        ax.axhline(
            self.threshold,
            color="crimson",
            ls="--",
            lw=2,
            label=f"Threshold ({self.threshold:.0f})",
        )
        if self.alarm_triggered:
            ax.axvline(
                self.alarm_time,
                color="orange",
                ls=":",
                lw=2,
                label=f"Alarm (t={self.alarm_time})",
            )
        if cp:
            ax.axvline(
                cp,
                color="green",
                ls="--",
                lw=2,
                alpha=0.7,
                label=f"Est. change (t≈{cp})",
            )
        ax.set(xlabel="Time", ylabel="Evidence", title="E-Process")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2: P-value histogram
        ax = axes[1]
        ax.hist(
            pvals[1:],
            bins=self.n_bins,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
        )
        ax.axhline(1, color="crimson", ls="--", lw=2)
        ax.set(
            xlabel="Conformal p-value",
            ylabel="Density",
            title="P-Values (uniform under H₀)",
        )

        plt.tight_layout()
        return fig

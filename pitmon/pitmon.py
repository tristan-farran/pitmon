import json
import pickle
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Union
from sortedcontainers import SortedList


@dataclass
class Alarm:
    """Result of each update."""

    triggered: bool
    time: int
    evidence: float
    threshold: float

    def __bool__(self) -> bool:
        return self.triggered


@dataclass
class PlotResult:
    """Result of calling plot()."""

    created: bool
    figure: Optional[plt.Figure]
    message: Optional[str] = None

    def __bool__(self) -> bool:
        return self.created


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
        but more variance. 10 is an MDL-reasonable choice for most settings.

    weight_schedule : Callable[[int], float], optional
        Mixture weight schedule over changepoint indices (index = 1, 2, ...).
        Must be deterministic, nonnegative, and sum to 1.
        Default is w(index) = 1 / (index * (index + 1)).
        These are mixture weights ensuring the combined e-process is a
        supermartingale under the null, enabling anytime-valid guarantees via
        Ville's inequality.

    Notes
    -----
    **Thread Safety:** This class is not thread-safe. External synchronization
    is required if multiple threads call ``update()`` concurrently.
    For parallel inference, create independent monitor instances.
    """

    _default_schedule_validated = False

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins: int = 10,
        weight_schedule: Optional[Callable[[int], float]] = None,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 5 <= n_bins <= 500:
            raise ValueError("n_bins must be in [5, 500]")

        self.alpha = alpha
        self.n_bins = n_bins
        self.threshold = 1.0 / alpha
        self._rng = np.random.default_rng(rng)
        self._uses_default_weight_schedule = weight_schedule is None
        self._weight_schedule = weight_schedule or self._default_weight_schedule
        if self._uses_default_weight_schedule:
            self._validate_default_weight_schedule_once()
        else:
            self._validate_weight_schedule()

        self.t = 0
        self._sorted_pits: SortedList[float] = SortedList()
        self._bin_counts = np.ones(n_bins)  # Laplace prior (pseudocount = 1)

        self._M = 0.0  # Mixture e-process (per paper)
        self._history: List[Tuple[float, float, float]] = []  # (pit, pval, M)

        self.alarm_triggered = False
        self.alarm_time: Optional[int] = None

    @staticmethod
    def _default_weight_schedule(index: int) -> float:
        """Default mixture weights over changepoint indices (index >= 1)."""
        if index < 1:
            raise ValueError("weight index must be >= 1")
        return 1.0 / (index * (index + 1))

    @classmethod
    def _validate_default_weight_schedule_once(cls) -> None:
        """One-time validation for the fixed default schedule w_t = 1/(t(t+1))."""
        if cls._default_schedule_validated:
            return

        horizon = 100_000
        total = 1.0 - 1.0 / (horizon + 1)
        mass_tol = 1e-3

        if abs(total - 1.0) > mass_tol:
            raise ValueError(
                "default weight schedule failed mass check "
                f"(mass over first {horizon} terms was {total:.6f})"
            )

        cls._default_schedule_validated = True

    def _validate_weight_schedule(self) -> None:
        """Validate that the configured mixture weights define a PMF over indices."""
        horizon = 100_000
        tol = 1e-8
        mass_tol = 1e-3

        # Deterministic check on a small prefix.
        for idx in range(1, 17):
            v1 = self._weight_schedule(idx)
            v2 = self._weight_schedule(idx)
            if not np.isclose(v1, v2, atol=0.0, rtol=0.0):
                raise ValueError("weight_schedule must be deterministic")

        total = 0.0
        for idx in range(1, horizon + 1):
            w = float(self._weight_schedule(idx))
            if not np.isfinite(w):
                raise ValueError("weight_schedule must return finite values")
            if w < 0:
                raise ValueError("weight_schedule must return nonnegative values")
            total += w
            if total > 1.0 + tol:
                raise ValueError("weight_schedule partial sums must not exceed 1")

        if abs(total - 1.0) > mass_tol:
            raise ValueError(
                "weight_schedule must sum to 1 over t=1,2,... "
                f"(mass over first {horizon} terms was {total:.6f})"
            )

    def _weight_at_time(self, t: int) -> float:
        """Mixture weight used at monitor time t (t >= 1)."""
        w = float(self._weight_schedule(t))
        if w < 0 or not np.isfinite(w):
            raise ValueError("weight_schedule produced an invalid weight")
        return w

    def _conformal_pvalue(self, pit: float) -> float:
        """Insert PIT and return tie-randomized conformal p-value."""
        self._sorted_pits.add(pit)
        left = self._sorted_pits.bisect_left(pit)
        right = self._sorted_pits.bisect_right(pit)
        U = self._rng.uniform(0, right - left)
        p = (left + U) / self.t
        return float(np.clip(p, 1e-10, 1 - 1e-10))

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
        if not 0 <= pit <= 1:
            raise ValueError(f"PIT {pit} not in [0, 1]")

        self.t += 1

        p = self._conformal_pvalue(pit)

        # Still track PIT/p-values after alarm (for complete diagnostics),
        # but freeze evidence process.
        if self.alarm_triggered:
            self._history.append((pit, p, self._M))
            return Alarm(True, self.alarm_time, self._M, self.threshold)

        # e_t = estimated density at p_t
        bin_idx = min(int(p * self.n_bins), self.n_bins - 1)
        density = self._bin_counts[bin_idx] / self._bin_counts.sum()
        e = density * self.n_bins  # scale to integrate to 1
        self._bin_counts[bin_idx] += 1  # update last to avoid peeking

        # Mixture e-process: M_t = e_t * (M_{t-1} + w_t).
        # These are mixture weights ensuring the combined e-process is a
        # supermartingale under the null, enabling anytime-valid guarantees via
        # Ville's inequality.
        w = self._weight_at_time(self.t)
        self._M = e * (self._M + w)
        self._history.append((pit, p, self._M))

        if self._M >= self.threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t

        return Alarm(self.alarm_triggered, self.t, self._M, self.threshold)

    def update_with_cdf(self, cdf: Callable[[float], float], y: float) -> Alarm:
        """
        Convenience method: compute PIT and process it.

        Parameters
        ----------
        cdf : Callable[[float], float]
            Cumulative distribution function F(y).
        y : float
            Observed value.

        Returns
        -------
        Alarm
            Same as update().

        Example
        -------
        >>> monitor = PITMonitor()
        >>> from scipy.stats import norm
        >>> # Predict with N(0,1), observe y=0.5
        >>> alarm = monitor.update_with_cdf(norm.cdf, 0.5)
        """
        return self.update(cdf(y))

    def update_many(self, pits: np.ndarray, stop_on_alarm: bool = True) -> Alarm:
        """
        Process a sequence of PIT values.

        Parameters
        ----------
        pits : array-like
            PIT values in [0, 1].
        stop_on_alarm : bool, default=True
            If True, stop processing once an alarm is triggered.

        Returns
        -------
        Alarm
            Alarm object for the final processed step.
        """
        last_alarm = Alarm(self.alarm_triggered, self.t, self._M, self.threshold)
        for pit in np.asarray(pits, dtype=float):
            last_alarm = self.update(float(pit))
            if stop_on_alarm and last_alarm.triggered:
                break
        return last_alarm

    def trial_summary(self, n_stable: int) -> dict:
        """
        Standardized trial diagnostics for a stable-then-shift stream.

        Parameters
        ----------
        n_stable : int
            Number of pre-change observations in the stream.

        Returns
        -------
        dict
            Dictionary with alarm, delay, and evidence diagnostics.
        """
        if n_stable < 0:
            raise ValueError("n_stable must be non-negative")

        s = self.summary()
        alarm_time = s["alarm_time"]
        false_alarm = alarm_time is not None and alarm_time <= n_stable
        detection_delay = (
            None if (alarm_time is None or false_alarm) else int(alarm_time - n_stable)
        )

        return {
            "alarm_fired": bool(s["alarm_triggered"]),
            "alarm_time": alarm_time,
            "detection_delay": detection_delay,
            "final_evidence": float(s["evidence"]),
            "false_alarm": bool(false_alarm),
        }

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

    def history(self) -> List[Alarm]:
        """Return history of all alarms (one per update)."""
        return [
            Alarm(
                self.alarm_triggered and t >= (self.alarm_time or float("inf")),
                t,
                M,
                self.threshold,
            )
            for t, (_, _, M) in enumerate(self._history, 1)
        ]

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
        Estimate changepoint by maximizing a Bayes factor score.

        For each candidate split k, compare post-split p-values under:
        - H0: fixed uniform categorical probabilities (1 / n_bins)
        - H1: unknown categorical probabilities with symmetric Dirichlet prior
              (Jeffreys prior: alpha = 1/2 per bin)

          Jeffreys prior is used as an objective, reparameterization-invariant
          default for multinomial probabilities.

        The selected changepoint maximizes the log Bayes factor
        log p(data_after | H1) - log p(data_after | H0).

        Returns
        -------
        int or None
            Estimated changepoint, or None if no alarm yet.
        """
        if not self.alarm_triggered or self.t < 3:
            return None

        pvals = self.pvalues
        n = len(pvals)
        max_k = min(n, self.alarm_time or n)

        # Score each admissible split with a log Bayes factor.
        # Candidate k means the post-change segment starts at index k+1.
        scores = []
        B = self.n_bins
        alpha = 0.5
        alpha0 = B * alpha

        # Precompute constants
        log_gamma_alpha = math.lgamma(alpha)
        log_gamma_alpha0 = math.lgamma(alpha0)
        log_B = math.log(B)

        for k in range(1, max_k):
            after = pvals[k:]
            N = len(after)
            if N == 0:
                continue

            counts, _ = np.histogram(after, bins=B, range=(0, 1))

            # log p(data | H1): Dirichlet-multinomial marginal likelihood
            #   log Γ(Bα) - log Γ(N + Bα) + Σ_j [log Γ(c_j + α) - log Γ(α)]
            # where c_j are post-split bin counts and α=1/2 (Jeffreys prior).
            log_p_h1 = log_gamma_alpha0 - math.lgamma(N + alpha0)
            log_p_h1 += sum(
                math.lgamma(int(c) + alpha) - log_gamma_alpha for c in counts
            )

            # log p(data | H0): fixed uniform categorical model (p_j = 1/B)
            #   Σ_j c_j log(1/B) = -N log(B)
            log_p_h0 = -N * log_B

            # Log Bayes factor in favor of "post-split is non-uniform".
            score = log_p_h1 - log_p_h0
            scores.append((k + 1, score))

        if not scores:
            return None

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
        # ECDF at sorted points: i/n for i=1,...,n
        ecdf = np.arange(1, self.t + 1) / self.t
        # KS statistic: max deviation between ECDF and uniform CDF
        ks_stat = np.max(np.abs(ecdf - pits_sorted))
        return float(ks_stat)

    def reset(self):
        """Reset to initial state."""
        self.t = 0
        self._sorted_pits = SortedList()
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

        Warnings
        --------
        **Pickle Security Warning:** pickle can execute arbitrary code from
        untrusted files. Only load pickle states from sources you trust.
        For untrusted inputs, use JSON format instead.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            state = {
                "alpha": self.alpha,
                "n_bins": self.n_bins,
                "t": self.t,
                "sorted_pits": list(self._sorted_pits),
                "bin_counts": self._bin_counts.tolist(),
                "M": self._M,
                "rng_state": self._rng.bit_generator.state,
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
            state = {
                "alpha": self.alpha,
                "n_bins": self.n_bins,
                "threshold": self.threshold,
                "t": self.t,
                "_sorted_pits": list(self._sorted_pits),
                "_bin_counts": self._bin_counts,
                "_M": self._M,
                "_rng_state": self._rng.bit_generator.state,
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
            Path to saved state file. Supports .pkl (pickle) and .json.

        Returns
        -------
        PITMonitor
            Restored monitor instance.

        Warnings
        --------
        **Pickle Security Warning:** Do not load pickle files from untrusted
        sources; pickle execution is not sandboxed. Prefer JSON for inputs
        from external/untrusted origins.

        Notes
        -----
        Backward compatible with legacy saved states; missing fields are
        reconstructed to safe defaults.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".json":
            with open(filepath, "r") as f:
                state = json.load(f)

            monitor = cls(alpha=state["alpha"], n_bins=state["n_bins"])
            monitor.t = state["t"]
            sorted_pits = state.get("sorted_pits", state.get("_sorted_pits", []))
            monitor._sorted_pits = SortedList(sorted_pits)
            bin_counts = state.get(
                "bin_counts", state.get("_bin_counts", np.ones(monitor.n_bins))
            )
            monitor._bin_counts = np.array(bin_counts)
            monitor._M = state.get("M", state.get("_M", 0.0))

            rng_state = state.get("rng_state", state.get("_rng_state"))
            if rng_state is not None:
                monitor._rng.bit_generator.state = rng_state

            history = state.get("history", state.get("_history", []))
            if history and isinstance(history[0], dict):
                monitor._history = [(h["pit"], h["pval"], h["M"]) for h in history]
            else:
                monitor._history = history

            monitor.alarm_triggered = state.get("alarm_triggered", False)
            monitor.alarm_time = state.get("alarm_time")
        else:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

            monitor = cls(alpha=state["alpha"], n_bins=state["n_bins"])
            monitor.threshold = state.get("threshold", 1.0 / monitor.alpha)
            monitor.t = state["t"]
            sorted_pits = state.get("_sorted_pits", state.get("sorted_pits", []))
            monitor._sorted_pits = SortedList(sorted_pits)
            monitor._bin_counts = state.get(
                "_bin_counts",
                np.array(state.get("bin_counts", np.ones(monitor.n_bins))),
            )
            monitor._M = state.get("_M", state.get("M", 0.0))

            rng_state = state.get("_rng_state", state.get("rng_state"))
            if rng_state is not None:
                monitor._rng.bit_generator.state = rng_state

            monitor._history = state.get("_history", state.get("history", []))
            monitor.alarm_triggered = state.get("alarm_triggered", False)
            monitor.alarm_time = state.get("alarm_time")

        return monitor

    def plot(self, figsize: Tuple[float, float] = (12, 4)) -> PlotResult:
        """
        Create diagnostic plot.

        Parameters
        ----------
        figsize : tuple of float, default=(12, 4)
            Figure size (width, height) in inches.

        Returns
        -------
        PlotResult
            Structured plot result with creation status, figure, and message.
        """
        if self.t < 2:
            return PlotResult(
                created=False,
                figure=None,
                message="Need ≥2 observations",
            )

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
                alpha=0.5,
                label=f"Est. change (t≈{cp})",
            )
        ax.set(xlabel="Time", ylabel="Evidence", title="E-Process")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.5)

        # 2: P-value histogram
        ax = axes[1]
        ax.hist(
            pvals[1:],
            bins=self.n_bins,
            density=True,
            alpha=0.5,
            color="steelblue",
            edgecolor="white",
        )
        ax.axhline(1, color="crimson", ls="--", lw=2)
        ax.set(
            xlabel="P-value",
            ylabel="Density",
            title="P-Values",
        )

        plt.tight_layout()
        return PlotResult(created=True, figure=fig, message=None)

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List


@dataclass
class AlarmInfo:
    """Information about a triggered alarm."""

    triggered: bool
    baseline_complete: bool = False
    alarm_time: Optional[int] = None
    changepoint_estimate: Optional[int] = None
    ks_distance: Optional[float] = None
    threshold: Optional[float] = None
    diagnosis: Optional[str] = None

    def __bool__(self):
        return self.triggered


class PITMonitor:
    """
    Monitor for changes in probabilistic model calibration via PIT analysis.

    Establishes a baseline calibration during initial observations, then monitors
    for significant changes from that baseline using two-sample testing of PIT
    distributions.

    This approach allows imperfect but stable models to pass monitoring, only
    alarming when calibration degrades or shifts. Unlike absolute uniformity
    testing, it won't eventually reject every slightly miscalibrated model.

    Parameters
    ----------
    false_alarm_rate : float, default=0.05
        Maximum probability of false alarm during monitoring period.
    baseline_size : int, default=50
        Number of initial observations to establish baseline calibration.
        Larger values give better baseline characterization but delay monitoring.
    changepoint_budget : float, default=0.5
        Fraction of false_alarm_rate reserved for changepoint localization.

    Attributes
    ----------
    t : int
        Total observations processed (baseline + monitoring)
    baseline_locked : bool
        Whether baseline collection is complete
    baseline_pits : list
        PITs from baseline period
    monitoring_pits : list
        PITs from monitoring period
    alarm_triggered : bool
        Whether calibration change was detected
    alarm_time : int or None
        Time when alarm was triggered

    Examples
    --------
    >>> monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)
    >>>
    >>> # Baseline phase
    >>> for prediction, outcome in data_stream[:50]:
    ...     info = monitor.update(prediction, outcome)
    ...     if info.baseline_complete:
    ...         print(f"Baseline established")
    >>>
    >>> # Monitoring phase
    >>> for prediction, outcome in data_stream[50:]:
    ...     alarm = monitor.update(prediction, outcome)
    ...     if alarm:
    ...         print(f"Calibration changed at t={monitor.t}")
    ...         cp = monitor.localize_changepoint()
    ...         print(f"Change began around t={cp}")
    ...         break

    Notes
    -----
    The monitor uses a two-phase approach:
    1. Baseline (first baseline_size observations): Establish reference calibration
    2. Monitoring (subsequent observations): Detect changes from baseline

    This is fundamentally different from testing absolute calibration quality.
    A poorly calibrated but stable model will not alarm, only changes trigger alarms.
    """

    def __init__(self, false_alarm_rate: float = 0.05,
                 baseline_size: int = 50,
                 changepoint_budget: float = 0.5):
        if not 0 < false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if baseline_size < 1:
            raise ValueError("baseline_size must be >= 1")
        if baseline_size < 30:
            import warnings
            warnings.warn(
                f"baseline_size={baseline_size} is small and may have low power. "
                "Consider baseline_size >= 30 for reliable change detection."
            )
        if not 0 < changepoint_budget < 1:
            raise ValueError("changepoint_budget must be in (0, 1)")

        self.alpha = false_alarm_rate
        self.baseline_size = baseline_size
        self.changepoint_budget = changepoint_budget

        # State
        self.t = 0
        self.baseline_pits = []
        self.baseline_locked = False
        self.monitoring_pits = []
        self._baseline_ks_dist = None
        self.alarm_triggered = False
        self.alarm_time = None
        self._alarm_info = None

    def update(
        self, predicted_cdf: Callable[[float], float], outcome: float
    ) -> AlarmInfo:
        """
        Process one new observation and check for alarm.

        Parameters
        ----------
        predicted_cdf : callable
            The model's CDF function F(y). Should map outcome values to [0,1].
        outcome : float
            The observed outcome value.

        Returns
        -------
        AlarmInfo
            Information about alarm status. Evaluates to True if alarm triggered.

        Examples
        --------
        >>> from scipy.stats import norm
        >>> monitor = PITMonitor()
        >>> predicted_dist = norm(loc=0, scale=1)
        >>> alarm = monitor.update(predicted_dist.cdf, outcome=2.5)
        """
        if self.alarm_triggered:
            return self._alarm_info

        # Compute PIT value
        u = predicted_cdf(outcome)

        # Validate PIT is in [0,1]
        if not 0 <= u <= 1:
            raise ValueError(
                f"PIT value {u} outside [0,1]. Check that predicted_cdf is a valid CDF."
            )

        self.t += 1

        # Phase 1: Baseline Collection
        if not self.baseline_locked:
            self.baseline_pits.append(u)

            if len(self.baseline_pits) >= self.baseline_size:
                # Lock baseline and compute quality metrics
                self.baseline_locked = True
                sorted_baseline = np.sort(self.baseline_pits)
                k = np.arange(1, len(self.baseline_pits) + 1)
                self._baseline_ks_dist = float(
                    np.max(np.abs(k / len(self.baseline_pits) - sorted_baseline))
                )

                # Warn if baseline appears poorly calibrated
                if self._baseline_ks_dist > 0.2:
                    import warnings
                    warnings.warn(
                        f"Baseline PITs show poor calibration (KS={self._baseline_ks_dist:.3f}). "
                        f"Monitor will detect CHANGES from this baseline, but baseline itself "
                        f"may not represent good calibration."
                    )

                return AlarmInfo(
                    triggered=False,
                    baseline_complete=True,
                    ks_distance=self._baseline_ks_dist,
                    threshold=None,
                )

            # Still collecting baseline
            return AlarmInfo(
                triggered=False,
                baseline_complete=False,
                ks_distance=None,
                threshold=None,
            )

        # Phase 2: Monitoring for changes from baseline
        self.monitoring_pits.append(u)

        # Compute two-sample test statistic and threshold
        ks_dist = self._compute_ks_distance()
        threshold = self._compute_threshold()

        # Check for alarm
        if ks_dist > threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t
            diagnosis = self._diagnose_deviation()

            self._alarm_info = AlarmInfo(
                triggered=True,
                baseline_complete=True,
                alarm_time=self.t,
                ks_distance=ks_dist,
                threshold=threshold,
                diagnosis=diagnosis,
            )
            return self._alarm_info

        return AlarmInfo(
            triggered=False,
            baseline_complete=True,
            ks_distance=ks_dist,
            threshold=threshold
        )

    def _compute_ks_distance(self) -> float:
        """Compute KS distance (baseline vs monitoring in change detection mode)."""
        if not self.baseline_locked:
            # During baseline: compute one-sample KS for diagnostics
            if len(self.baseline_pits) == 0:
                return 0.0
            sorted_pits = np.sort(self.baseline_pits)
            k = np.arange(1, len(self.baseline_pits) + 1)
            return float(np.max(np.abs(k / len(self.baseline_pits) - sorted_pits)))

        # During monitoring: two-sample KS
        if len(self.monitoring_pits) == 0:
            return 0.0
        return self._compute_ks_two_sample(
            np.array(self.baseline_pits),
            np.array(self.monitoring_pits)
        )

    @staticmethod
    def _compute_ks_two_sample(pits1: np.ndarray, pits2: np.ndarray) -> float:
        """
        Compute two-sample Kolmogorov-Smirnov distance: max |F1(u) - F2(u)|

        Parameters
        ----------
        pits1 : np.ndarray
            First sample of PITs
        pits2 : np.ndarray
            Second sample of PITs

        Returns
        -------
        float
            Maximum difference between empirical CDFs
        """
        n1, n2 = len(pits1), len(pits2)
        if n1 == 0 or n2 == 0:
            return 0.0

        sorted1 = np.sort(pits1)
        sorted2 = np.sort(pits2)

        # Combined unique values where we evaluate CDFs
        all_vals = np.unique(np.concatenate([sorted1, sorted2]))

        # Compute max difference between empirical CDFs
        max_dist = 0.0
        for u in all_vals:
            F1_u = np.sum(sorted1 <= u) / n1
            F2_u = np.sum(sorted2 <= u) / n2
            max_dist = max(max_dist, abs(F1_u - F2_u))

        return float(max_dist)

    def _compute_threshold(self) -> float:
        """Compute two-sample sequential threshold."""
        if not self.baseline_locked:
            return np.inf  # No threshold during baseline

        t_monitor = len(self.monitoring_pits)
        if t_monitor == 0:
            return np.inf

        n_baseline = len(self.baseline_pits)
        # Effective sample size for two-sample test (harmonic mean)
        n_eff = (n_baseline * t_monitor) / (n_baseline + t_monitor)

        # Alpha spending on monitoring time steps
        alpha_t = self.alpha / (np.pi**2 * t_monitor**2)
        return np.sqrt(np.log(2 / alpha_t) / (2 * n_eff))

    def _diagnose_deviation(self) -> str:
        """Diagnose how monitoring distribution differs from baseline."""
        baseline_sorted = np.sort(self.baseline_pits)
        monitoring_sorted = np.sort(self.monitoring_pits)

        # Find where maximum deviation occurs
        all_vals = np.unique(np.concatenate([baseline_sorted, monitoring_sorted]))
        max_dev = 0
        u_star = 0.5

        for u in all_vals:
            F_base = np.sum(baseline_sorted <= u) / len(baseline_sorted)
            F_mon = np.sum(monitoring_sorted <= u) / len(monitoring_sorted)
            dev = F_mon - F_base
            if abs(dev) > abs(max_dev):
                max_dev = dev
                u_star = u

        location = "lower tail" if u_star < 0.1 else ("upper tail" if u_star > 0.9 else "central")
        direction = "more extremes" if max_dev > 0 else "fewer extremes"
        return f"{location} - {direction} vs baseline"

    def get_baseline_diagnostics(self) -> dict:
        """
        Get baseline collection status and quality.

        Returns
        -------
        dict
            Dictionary with baseline status and quality metrics
        """
        if not self.baseline_locked:
            return {
                'complete': False,
                'collected': len(self.baseline_pits),
                'target': self.baseline_size
            }

        return {
            'complete': True,
            'size': len(self.baseline_pits),
            'ks_from_uniform': self._baseline_ks_dist,
            'mean_pit': float(np.mean(self.baseline_pits)),
            'quality': 'good' if self._baseline_ks_dist < 0.15 else 'poor'
        }

    def localize_changepoint(self, method: str = "backward_scan") -> Optional[int]:
        """
        Estimate when calibration started changing from baseline.

        Scans through monitoring phase to find earliest detectable change.

        Parameters
        ----------
        method : str, default='backward_scan'
            Method for localization ('backward_scan' or 'binary_search')

        Returns
        -------
        int or None
            Estimated changepoint time (absolute time step), or None if no alarm triggered
        """
        if not self.alarm_triggered or not self.baseline_locked:
            return None

        if method == "backward_scan":
            return self._localize_backward_scan()
        elif method == "binary_search":
            return self._localize_binary_search()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _localize_backward_scan(self) -> int:
        """Backward scan for changepoint in monitoring phase."""
        if self.alarm_time is None:
            return self.baseline_size + 1

        alpha_cp = self.alpha * self.changepoint_budget
        baseline_array = np.array(self.baseline_pits)
        n_baseline = len(baseline_array)

        # Scan forward through monitoring phase
        n_monitor = len(self.monitoring_pits)
        segment_lengths = self._geometric_sequence(1, n_monitor)

        for k in segment_lengths:
            # Test if first k monitoring observations differ from baseline
            monitoring_segment = np.array(self.monitoring_pits[:k])

            # Two-sample KS test
            ks_dist = self._compute_ks_two_sample(baseline_array, monitoring_segment)

            # Threshold for this segment size
            n_eff = (n_baseline * k) / (n_baseline + k)
            alpha_k = alpha_cp / (np.pi**2 * int(np.log2(k) + 1) ** 2) if k > 1 else alpha_cp / 2
            threshold_k = np.sqrt(np.log(2 / alpha_k) / (2 * n_eff))

            if ks_dist > threshold_k:
                # Change detected in first k monitoring steps
                return self.baseline_size + k

        # Default to start of monitoring if no earlier detection
        return self.baseline_size + 1

    def _localize_binary_search(self) -> int:
        """Binary search for changepoint (simple approximation)."""
        if self.alarm_time is None:
            return self.baseline_size + 1

        # Estimate midpoint of monitoring phase
        n_monitor = len(self.monitoring_pits)
        return self.baseline_size + max(1, n_monitor // 2)

    @staticmethod
    def _compute_ks_on_segment(pits: np.ndarray) -> float:
        """Compute KS distance on a segment of PITs."""
        n = len(pits)
        if n == 0:
            return 0.0
        sorted_pits = np.sort(pits)
        k = np.arange(1, n + 1)
        deviations = np.abs(k / n - sorted_pits)
        return float(np.max(deviations))

    @staticmethod
    def _geometric_sequence(start: int, end: int, base: float = 1.5) -> List[int]:
        """Generate geometrically spaced integers."""
        if start >= end:
            return [end]
        sequence = []
        current = start
        while current < end:
            sequence.append(int(current))
            current *= base
        sequence.append(end)
        return sorted(set(sequence))

    def get_state(self) -> dict:
        """Get current monitor state."""
        return {
            "t": self.t,
            "baseline_locked": self.baseline_locked,
            "baseline_size": len(self.baseline_pits),
            "monitoring_size": len(self.monitoring_pits),
            "pits": self.baseline_pits + self.monitoring_pits,  # All PITs
            "ks_distance": self._compute_ks_distance() if self.t > 0 else None,
            "threshold": self._compute_threshold() if self.baseline_locked else None,
            "alarm_triggered": self.alarm_triggered,
            "alarm_time": self.alarm_time,
            "alpha": self.alpha,
        }

    def plot_diagnostics(self, figsize=(12, 8)):
        """
        Create diagnostic plots of monitor state.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
            Figure with diagnostic plots
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. Install with: pip install matplotlib"
            )

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        if self.t == 0:
            fig.suptitle("No data yet")
            return fig

        baseline_array = np.array(self.baseline_pits)
        monitoring_array = np.array(self.monitoring_pits)

        # Plot 1: PIT histogram (baseline vs monitoring)
        ax = axes[0, 0]
        if self.baseline_locked and len(monitoring_array) > 0:
            ax.hist(baseline_array, bins=15, density=True, alpha=0.5,
                   label="Baseline", color="blue", edgecolor="black")
            ax.hist(monitoring_array, bins=15, density=True, alpha=0.5,
                   label="Monitoring", color="orange", edgecolor="black")
        else:
            ax.hist(baseline_array, bins=20, density=True, alpha=0.7,
                   label="Baseline" if not self.baseline_locked else "PITs",
                   color="blue", edgecolor="black")
        ax.axhline(1.0, color="red", linestyle="--", label="Uniform", alpha=0.7)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title("PIT Distribution")
        ax.legend()

        # Plot 2: Empirical CDFs (baseline vs monitoring)
        ax = axes[0, 1]
        sorted_baseline = np.sort(baseline_array)
        baseline_cdf = np.arange(1, len(baseline_array) + 1) / len(baseline_array)
        ax.plot(sorted_baseline, baseline_cdf, "b-", label="Baseline CDF", linewidth=2, alpha=0.8)

        if self.baseline_locked and len(monitoring_array) > 0:
            sorted_monitoring = np.sort(monitoring_array)
            monitoring_cdf = np.arange(1, len(monitoring_array) + 1) / len(monitoring_array)
            ax.plot(sorted_monitoring, monitoring_cdf, "orange", label="Monitoring CDF", linewidth=2, alpha=0.8)

        ax.plot([0, 1], [0, 1], "r--", label="Uniform CDF", linewidth=2, alpha=0.7)
        ax.set_xlabel("u")
        ax.set_ylabel("F(u)")
        ax.set_title("Empirical CDFs")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Plot 3: Two-sample KS distance over monitoring time
        ax = axes[1, 0]

        if self.baseline_locked and len(monitoring_array) > 0:
            # Compute two-sample KS at each monitoring step
            ks_history = []
            threshold_history = []

            for k in range(1, len(monitoring_array) + 1):
                monitoring_segment = monitoring_array[:k]
                ks_dist = self._compute_ks_two_sample(baseline_array, monitoring_segment)
                ks_history.append(ks_dist)

                # Threshold at monitoring step k
                n_baseline = len(baseline_array)
                n_eff = (n_baseline * k) / (n_baseline + k)
                alpha_t = self.alpha / (np.pi**2 * k**2)
                threshold = np.sqrt(np.log(2 / alpha_t) / (2 * n_eff))
                threshold_history.append(threshold)

            monitoring_times = np.arange(self.baseline_size + 1, self.t + 1)
            ax.plot(monitoring_times, ks_history, "b-", label="Two-sample KS", linewidth=2)
            ax.plot(monitoring_times, threshold_history, "r--", label="Threshold", linewidth=2)

            if self.alarm_triggered:
                ax.axvline(self.alarm_time, color="orange", linestyle=":",
                          label=f"Alarm (t={self.alarm_time})", linewidth=2)

            ax.set_xlabel("Time")
            ax.set_ylabel("Two-sample KS distance")
            ax.set_title("Change Detection Monitoring")
            ax.legend()
            ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "Baseline collection in progress",
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("KS distance")
            ax.set_title("Monitoring (Not Started)")

        # Plot 4: PIT sequence over time
        ax = axes[1, 1]

        # Plot baseline PITs
        baseline_times = np.arange(1, len(baseline_array) + 1)
        ax.scatter(baseline_times, baseline_array, alpha=0.5, s=20,
                  color="blue", label="Baseline")

        # Plot monitoring PITs if available
        if self.baseline_locked and len(monitoring_array) > 0:
            monitoring_times = np.arange(len(baseline_array) + 1, self.t + 1)
            ax.scatter(monitoring_times, monitoring_array, alpha=0.5, s=20,
                      color="orange", label="Monitoring")

        # Mark baseline completion
        if self.baseline_locked:
            ax.axvline(self.baseline_size, color="green", linestyle="--",
                      alpha=0.7, linewidth=2, label="Baseline complete")

        # Mark alarm
        if self.alarm_triggered:
            ax.axvline(self.alarm_time, color="red", linestyle=":",
                      label=f"Alarm (t={self.alarm_time})", linewidth=2)

        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("PIT value")
        ax.set_title("PIT Sequence")
        ax.set_ylim([-0.05, 1.05])
        ax.legend()

        fig.suptitle(
            f"PIT Monitor Diagnostics (α={self.alpha}, baseline={self.baseline_size})",
            fontsize=14, y=1.00
        )
        plt.tight_layout()
        return fig

import bisect

import numpy as np
from dataclasses import dataclass


@dataclass
class AlarmInfo:
    """Information about a triggered alarm."""

    triggered: bool
    alarm_time: object = None
    martingale: object = None
    threshold: object = None
    diagnosis: object = None

    def __bool__(self):
        return self.triggered


class PITMonitor:
    """
    Sequential calibration monitor using conformal exchangeability testing.

    Detects distributional changes in PITs without requiring a baseline period.
    Each new PIT is ranked among all previous PITs; under stable calibration
    (exchangeability), these ranks are uniform regardless of the underlying
    distribution. A conformal test martingale accumulates evidence against
    exchangeability, alarming when it exceeds 1/alpha.

    This tolerates imperfect but stable calibration — a consistently
    miscalibrated model produces exchangeable PITs and will not alarm.
    Only changes in calibration trigger alarms.

    Parameters
    ----------
    false_alarm_rate : float, default=0.05
        Maximum probability of false alarm over the entire monitoring period.
        Guarantee is anytime-valid (holds at all stopping times).

    Attributes
    ----------
    t : int
        Total observations processed.
    pits : list
        All PIT values observed.
    alarm_triggered : bool
        Whether a calibration change was detected.
    alarm_time : int or None
        Time step when alarm was triggered.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> monitor = PITMonitor(false_alarm_rate=0.05)
    >>>
    >>> for prediction, outcome in data_stream:
    ...     alarm = monitor.update(prediction.cdf, outcome)
    ...     if alarm:
    ...         print(f"Calibration changed at t={monitor.t}")
    ...         cp = monitor.localize_changepoint()
    ...         print(f"Change began around t={cp}")
    ...         break

    Notes
    -----
    The monitor uses a conformal mixture e-process: for each PIT u_t, it
    computes the rank of u_t among all PITs seen so far. Under exchangeability,
    these ranks are uniform. A two-sided power betting function (epsilon=0.5)
    converts rank-based p-values into e-values. These are aggregated via a
    mixture over possible changepoint times with a slowly-decaying prior,
    forming a valid e-process. By Ville's inequality, P(sup M_t >= 1/alpha)
    <= alpha, giving exact anytime-valid false alarm control.
    """

    def __init__(self, false_alarm_rate=0.05):
        if not 0 < false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be in (0, 1)")

        self.alpha = false_alarm_rate

        # State
        self.t = 0
        self.pits = []
        self._sorted_pits = []
        self._mixture = 0.0  # mixture e-process value
        self._mixture_history = []
        self.alarm_triggered = False
        self.alarm_time = None
        self._alarm_info = None

    def update(self, predicted_cdf, outcome):
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
        """
        if self.alarm_triggered:
            return self._alarm_info

        u = predicted_cdf(outcome)
        return self._process_pit(u)

    def update_pit(self, pit_value):
        """
        Process one new observation using a pre-computed PIT value.

        Parameters
        ----------
        pit_value : float
            Pre-computed PIT value in [0, 1].

        Returns
        -------
        AlarmInfo
            Information about alarm status. Evaluates to True if alarm triggered.
        """
        if self.alarm_triggered:
            return self._alarm_info

        return self._process_pit(pit_value)

    def _process_pit(self, u):
        """Internal method to process a PIT value."""
        if not 0 <= u <= 1:
            raise ValueError(
                f"PIT value {u} outside [0,1]. Check that predicted_cdf "
                "is a valid CDF or that pit_value is valid."
            )

        self.t += 1
        self.pits.append(u)
        bisect.insort(self._sorted_pits, u)

        threshold = 1 / self.alpha

        if self.t == 1:
            self._mixture_history.append(0.0)
            return AlarmInfo(
                triggered=False, martingale=0.0, threshold=threshold
            )

        # Conformal p-value: rank among all observations
        # Using (t+1) denominator keeps p strictly in (0, 1)
        rank = bisect.bisect_right(self._sorted_pits, u)
        p = rank / (self.t + 1)

        # Two-sided power e-value (epsilon = 0.5)
        # f(p) = 0.25/sqrt(p) + 0.25/sqrt(1-p)
        e = 0.25 / np.sqrt(p) + 0.25 / np.sqrt(1 - p)

        # Mixture e-process with prior over changepoint times.
        # w_s = (1/log(2)) / ((s+2) * log^2(s+2)), which sums to 1
        # and decays slowly (~1/(s*log^2(s))), giving meaningful weight
        # to late changepoints.
        # M_t = e_t * (M_{t-1} + w_{t-1}) is a valid e-process:
        # the augmented process M_t + sum_{s>=t} w_s is a supermartingale,
        # so P(sup M_t >= 1/alpha) <= alpha by Ville's inequality.
        s = self.t - 1  # changepoint index (0-based)
        log_s2 = np.log(s + 2)
        w = 1.0 / (np.log(2) * (s + 2) * log_s2 ** 2)
        self._mixture = e * (self._mixture + w)
        self._mixture_history.append(self._mixture)

        if self._mixture >= threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t
            diagnosis = self._diagnose_deviation()

            self._alarm_info = AlarmInfo(
                triggered=True,
                alarm_time=self.t,
                martingale=self._mixture,
                threshold=threshold,
                diagnosis=diagnosis,
            )
            return self._alarm_info

        return AlarmInfo(
            triggered=False,
            martingale=self._mixture,
            threshold=threshold,
        )

    def localize_changepoint(self):
        """
        Estimate when calibration started changing.

        Uses binary segmentation: finds the split point that maximizes
        the scaled two-sample KS statistic between PITs before and after
        the candidate point. Pure estimation — no significance level needed.

        Returns
        -------
        int or None
            Estimated changepoint time (1-indexed), or None if no alarm.
        """
        if not self.alarm_triggered:
            return None

        n = len(self.pits)
        if n < 2:
            return 1

        pits_array = np.array(self.pits)
        best_stat = 0.0
        best_k = 1

        for k in range(1, n):
            left = pits_array[:k]
            right = pits_array[k:]
            ks_dist = self._compute_ks_two_sample(left, right)
            n_eff = (k * (n - k)) / n
            scaled_stat = np.sqrt(n_eff) * ks_dist

            if scaled_stat > best_stat:
                best_stat = scaled_stat
                best_k = k

        return best_k

    @staticmethod
    def _compute_ks_two_sample(pits1, pits2):
        """Compute two-sample KS distance. Used for changepoint localization."""
        n1, n2 = len(pits1), len(pits2)
        if n1 == 0 or n2 == 0:
            return 0.0

        sorted1 = np.sort(pits1)
        sorted2 = np.sort(pits2)

        all_vals = np.unique(np.concatenate([sorted1, sorted2]))

        max_dist = 0.0
        for u in all_vals:
            F1_u = np.sum(sorted1 <= u) / n1
            F2_u = np.sum(sorted2 <= u) / n2
            max_dist = max(max_dist, abs(F1_u - F2_u))

        return float(max_dist)

    def _diagnose_deviation(self):
        """Diagnose how the PIT distribution changed."""
        cp = self.localize_changepoint()
        if cp is None or cp >= len(self.pits):
            return "unknown"

        before = np.array(self.pits[:cp])
        after = np.array(self.pits[cp:])

        if len(before) == 0 or len(after) == 0:
            return "unknown"

        shift = np.mean(after) - np.mean(before)
        if abs(shift) > 0.1:
            direction = "higher" if shift > 0 else "lower"
            return f"PITs shifted {direction} after t={cp}"

        var_ratio = np.var(after) / max(np.var(before), 1e-10)
        if var_ratio > 1.5:
            return f"PITs became more dispersed after t={cp}"
        elif var_ratio < 1 / 1.5:
            return f"PITs became more concentrated after t={cp}"

        return f"PIT distribution shape changed after t={cp}"

    @property
    def evidence(self):
        """Current mixture e-process value (evidence against exchangeability)."""
        return self._mixture

    def get_state(self):
        """Get current monitor state."""
        return {
            "t": self.t,
            "pits": list(self.pits),
            "evidence": self._mixture,
            "threshold": 1 / self.alpha,
            "alarm_triggered": self.alarm_triggered,
            "alarm_time": self.alarm_time,
            "alpha": self.alpha,
        }

    def get_summary(self):
        """
        Get a comprehensive summary of monitor status.

        Returns
        -------
        dict
            Dictionary with monitoring summary statistics and status.
        """
        summary = {
            "status": "alarm" if self.alarm_triggered else "monitoring",
            "observations_processed": self.t,
            "evidence": self._mixture,
            "threshold": 1 / self.alpha,
            "parameters": {
                "false_alarm_rate": self.alpha,
            },
        }

        if self.alarm_triggered and self._alarm_info:
            summary["alarm"] = {
                "triggered_at": self.alarm_time,
                "estimated_changepoint": self.localize_changepoint(),
                "cusum": self._alarm_info.martingale,
                "diagnosis": self._alarm_info.diagnosis,
            }

        return summary

    def print_summary(self):
        """Print a human-readable summary of monitor status."""
        summary = self.get_summary()

        print("=" * 70)
        print("PITMonitor Summary")
        print("=" * 70)
        print(f"Status: {summary['status'].upper()}")
        print(f"Observations processed: {summary['observations_processed']}")
        print(f"Evidence: {summary['evidence']:.4f}")
        print(f"Threshold: {summary['threshold']:.1f}")
        print()

        if self.alarm_triggered:
            alarm = summary["alarm"]
            print("ALARM:")
            print(f"  Triggered at: t={alarm['triggered_at']}")
            print(f"  Estimated changepoint: t={alarm['estimated_changepoint']}")
            print(f"  Diagnosis: {alarm['diagnosis']}")

        print("=" * 70)

    def export_data(self):
        """
        Export all data for external analysis.

        Returns
        -------
        dict
            Dictionary with all PITs, martingale history, and metadata.
        """
        return {
            "metadata": {
                "version": "0.3.0",
                "parameters": {
                    "false_alarm_rate": self.alpha,
                },
            },
            "pits": list(self.pits),
            "evidence_history": list(self._mixture_history),
            "state": self.get_state(),
            "summary": self.get_summary(),
        }

    def plot_diagnostics(self, figsize=(12, 8)):
        """
        Create diagnostic plots of monitor state.

        Parameters
        ----------
        figsize : tuple, default=(12, 8)
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with diagnostic plots.
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

        pits_array = np.array(self.pits)
        evidence = np.array(self._mixture_history)
        times = np.arange(1, self.t + 1)

        # Plot 1: PIT histogram
        ax = axes[0, 0]
        if self.alarm_triggered:
            cp = self.localize_changepoint()
            ax.hist(
                pits_array[:cp], bins=15, density=True, alpha=0.5,
                label=f"Before t={cp}", color="blue", edgecolor="black",
            )
            ax.hist(
                pits_array[cp:], bins=15, density=True, alpha=0.5,
                label=f"After t={cp}", color="orange", edgecolor="black",
            )
        else:
            ax.hist(
                pits_array, bins=20, density=True, alpha=0.7,
                label="PITs", color="blue", edgecolor="black",
            )
        ax.axhline(1.0, color="red", linestyle="--", label="Uniform", alpha=0.7)
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title("PIT Distribution")
        ax.legend()

        # Plot 2: Empirical CDFs
        ax = axes[0, 1]
        if self.alarm_triggered:
            cp = self.localize_changepoint()
            before = np.sort(pits_array[:cp])
            after = np.sort(pits_array[cp:])
            ax.plot(
                before, np.arange(1, len(before) + 1) / len(before),
                "b-", label=f"Before t={cp}", linewidth=2, alpha=0.8,
            )
            ax.plot(
                after, np.arange(1, len(after) + 1) / len(after),
                color="orange", label=f"After t={cp}", linewidth=2, alpha=0.8,
            )
        else:
            sorted_pits = np.sort(pits_array)
            ax.plot(
                sorted_pits, np.arange(1, len(sorted_pits) + 1) / len(sorted_pits),
                "b-", label="Empirical CDF", linewidth=2, alpha=0.8,
            )
        ax.plot([0, 1], [0, 1], "r--", label="Uniform CDF", linewidth=2, alpha=0.7)
        ax.set_xlabel("u")
        ax.set_ylabel("F(u)")
        ax.set_title("Empirical CDFs")
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Plot 3: Evidence over time
        ax = axes[1, 0]
        ax.plot(times, evidence, "b-", label="Evidence", linewidth=2)
        ax.axhline(
            1 / self.alpha, color="red", linestyle="--",
            label=f"Threshold (1/\u03b1)", linewidth=2,
        )
        if self.alarm_triggered:
            ax.axvline(
                self.alarm_time, color="orange", linestyle=":",
                label=f"Alarm (t={self.alarm_time})", linewidth=2,
            )
            cp = self.localize_changepoint()
            ax.axvline(
                cp, color="green", linestyle="--",
                label=f"Changepoint (t={cp})", linewidth=2, alpha=0.7,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Mixture e-process")
        ax.set_title("Evidence Against Exchangeability")
        ax.legend()

        # Plot 4: PIT sequence over time
        ax = axes[1, 1]
        if self.alarm_triggered:
            cp = self.localize_changepoint()
            ax.scatter(
                times[:cp], pits_array[:cp], alpha=0.5, s=20,
                color="blue", label=f"Before t={cp}",
            )
            ax.scatter(
                times[cp:], pits_array[cp:], alpha=0.5, s=20,
                color="orange", label=f"After t={cp}",
            )
            ax.axvline(
                cp, color="green", linestyle="--",
                label=f"Changepoint (t={cp})", linewidth=2, alpha=0.7,
            )
        else:
            ax.scatter(
                times, pits_array, alpha=0.5, s=20, color="blue", label="PITs",
            )
        if self.alarm_triggered:
            ax.axvline(
                self.alarm_time, color="red", linestyle=":",
                label=f"Alarm (t={self.alarm_time})", linewidth=2,
            )
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
        ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Time")
        ax.set_ylabel("PIT value")
        ax.set_title("PIT Sequence")
        ax.set_ylim([-0.05, 1.05])
        ax.legend()

        fig.suptitle(
            f"PIT Monitor Diagnostics (\u03b1={self.alpha})",
            fontsize=14, y=1.00,
        )
        plt.tight_layout()
        return fig

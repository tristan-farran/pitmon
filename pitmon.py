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

    Tolerates imperfect but stable calibration — a consistently miscalibrated
    model produces exchangeable PITs and will not alarm. Only *changes* in
    calibration trigger alarms.

    Parameters
    ----------
    false_alarm_rate : float, default=0.05
        Maximum probability of false alarm over the entire monitoring period.
        Guarantee is anytime-valid (holds at all stopping times).

    Examples
    --------
    >>> from scipy.stats import norm
    >>> monitor = PITMonitor(false_alarm_rate=0.05)
    >>>
    >>> for prediction, outcome in data_stream:
    ...     alarm = monitor.update(prediction.cdf, outcome)
    ...     if alarm:
    ...         print(f"Calibration changed at t={monitor.t}")
    ...         print(f"Change began around t={monitor.localize_changepoint()}")
    ...         break
    """

    def __init__(self, false_alarm_rate=0.05):
        if not 0 < false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be in (0, 1)")

        self.alpha = false_alarm_rate
        self.t = 0
        self.pits = []
        self._sorted_pits = []
        self._mixture = 0.0
        self._mixture_history = []
        self.alarm_triggered = False
        self.alarm_time = None
        self._alarm_info = None

    def update(self, predicted_cdf, outcome):
        """
        Process one observation. Returns AlarmInfo (truthy if alarm triggered).

        Parameters
        ----------
        predicted_cdf : callable
            The model's CDF function F(y), mapping outcome values to [0, 1].
        outcome : float
            The observed outcome value.
        """
        if self.alarm_triggered:
            return self._alarm_info
        return self._process_pit(predicted_cdf(outcome))

    def update_pit(self, pit_value):
        """
        Process a pre-computed PIT value. Returns AlarmInfo.

        Parameters
        ----------
        pit_value : float
            Pre-computed PIT value in [0, 1].
        """
        if self.alarm_triggered:
            return self._alarm_info
        return self._process_pit(pit_value)

    def _process_pit(self, u):
        if not 0 <= u <= 1:
            raise ValueError(f"PIT value {u} outside [0,1]")

        self.t += 1
        self.pits.append(u)
        bisect.insort(self._sorted_pits, u)

        threshold = 1 / self.alpha

        if self.t == 1:
            self._mixture_history.append(0.0)
            return AlarmInfo(triggered=False, martingale=0.0, threshold=threshold)

        # Conformal p-value: rank of u among all observations
        rank = bisect.bisect_right(self._sorted_pits, u)
        p = rank / (self.t + 1)

        # Two-sided power e-value (epsilon = 0.5)
        e = 0.25 / np.sqrt(p) + 0.25 / np.sqrt(1 - p)

        # Mixture e-process with slowly-decaying prior over changepoint times.
        # w_s = 1/(ln2 * (s+2) * ln^2(s+2)) sums to 1, giving meaningful
        # weight to late changepoints. M_t = e_t * (M_{t-1} + w_{t-1}) is a
        # valid e-process; P(sup M_t >= 1/alpha) <= alpha by Ville's inequality.
        s = self.t - 1
        log_s2 = np.log(s + 2)
        w = 1.0 / (np.log(2) * (s + 2) * log_s2**2)
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

        return AlarmInfo(triggered=False, martingale=self._mixture, threshold=threshold)

    def localize_changepoint(self):
        """
        Estimate when calibration started changing via binary segmentation.

        Returns the split point maximizing the scaled two-sample KS statistic.
        Returns None if no alarm has been triggered.
        """
        if not self.alarm_triggered:
            return None

        n = len(self.pits)
        if n < 2:
            return 1

        pits = np.array(self.pits)
        best_stat, best_k = 0.0, 1

        for k in range(1, n):
            ks = self._ks_two_sample(pits[:k], pits[k:])
            scaled = np.sqrt(k * (n - k) / n) * ks
            if scaled > best_stat:
                best_stat, best_k = scaled, k

        return best_k

    @staticmethod
    def _ks_two_sample(a, b):
        """Two-sample Kolmogorov-Smirnov distance."""
        all_vals = np.unique(np.concatenate([a, b]))
        na, nb = len(a), len(b)
        return max(abs(np.sum(a <= v) / na - np.sum(b <= v) / nb) for v in all_vals)

    def _diagnose_deviation(self):
        cp = self.localize_changepoint()
        if cp is None or cp >= len(self.pits):
            return "unknown"

        before, after = np.array(self.pits[:cp]), np.array(self.pits[cp:])
        if len(before) == 0 or len(after) == 0:
            return "unknown"

        shift = np.mean(after) - np.mean(before)
        if abs(shift) > 0.1:
            return f"PITs shifted {'higher' if shift > 0 else 'lower'} after t={cp}"

        var_ratio = np.var(after) / max(np.var(before), 1e-10)
        if var_ratio > 1.5:
            return f"PITs became more dispersed after t={cp}"
        if var_ratio < 1 / 1.5:
            return f"PITs became more concentrated after t={cp}"

        return f"PIT distribution shape changed after t={cp}"

    @property
    def evidence(self):
        """Current e-process value (evidence against exchangeability)."""
        return self._mixture

    def plot(self, figsize=(12, 8)):
        """Create 4-panel diagnostic plot. Returns matplotlib Figure."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required: pip install matplotlib")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        if self.t == 0:
            fig.suptitle("No data yet")
            return fig

        pits = np.array(self.pits)
        evidence = np.array(self._mixture_history)
        times = np.arange(1, self.t + 1)
        cp = self.localize_changepoint() if self.alarm_triggered else None

        # 1: PIT histogram
        ax = axes[0, 0]
        if cp:
            ax.hist(
                pits[:cp],
                bins=15,
                density=True,
                alpha=0.5,
                label=f"Before t={cp}",
                color="blue",
                edgecolor="black",
            )
            ax.hist(
                pits[cp:],
                bins=15,
                density=True,
                alpha=0.5,
                label=f"After t={cp}",
                color="orange",
                edgecolor="black",
            )
        else:
            ax.hist(
                pits, bins=20, density=True, alpha=0.7, color="blue", edgecolor="black"
            )
        ax.axhline(1.0, color="red", linestyle="--", alpha=0.7, label="Uniform")
        ax.set(xlabel="PIT value", ylabel="Density", title="PIT Distribution")
        ax.legend()

        # 2: Empirical CDFs
        ax = axes[0, 1]
        if cp:
            before, after = np.sort(pits[:cp]), np.sort(pits[cp:])
            ax.plot(
                before,
                np.linspace(0, 1, len(before), endpoint=False) + 1 / len(before),
                "b-",
                label=f"Before t={cp}",
                lw=2,
                alpha=0.8,
            )
            ax.plot(
                after,
                np.linspace(0, 1, len(after), endpoint=False) + 1 / len(after),
                color="orange",
                label=f"After t={cp}",
                lw=2,
                alpha=0.8,
            )
        else:
            s = np.sort(pits)
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), "b-", lw=2, alpha=0.8)
        ax.plot([0, 1], [0, 1], "r--", lw=2, alpha=0.7, label="Uniform CDF")
        ax.set(
            xlabel="u", ylabel="F(u)", title="Empirical CDFs", xlim=[0, 1], ylim=[0, 1]
        )
        ax.legend()

        # 3: Evidence over time
        ax = axes[1, 0]
        ax.plot(times, evidence, "b-", lw=2, label="Evidence")
        ax.axhline(1 / self.alpha, color="red", linestyle="--", lw=2, label="Threshold")
        if cp:
            ax.axvline(
                self.alarm_time,
                color="orange",
                ls=":",
                lw=2,
                label=f"Alarm t={self.alarm_time}",
            )
            ax.axvline(
                cp, color="green", ls="--", lw=2, alpha=0.7, label=f"Changepoint t={cp}"
            )
        ax.set(
            xlabel="Time",
            ylabel="Mixture e-process",
            title="Evidence Against Exchangeability",
        )
        ax.legend()

        # 4: PIT sequence
        ax = axes[1, 1]
        if cp:
            ax.scatter(
                times[:cp],
                pits[:cp],
                alpha=0.5,
                s=20,
                color="blue",
                label=f"Before t={cp}",
            )
            ax.scatter(
                times[cp:],
                pits[cp:],
                alpha=0.5,
                s=20,
                color="orange",
                label=f"After t={cp}",
            )
            ax.axvline(cp, color="green", ls="--", lw=2, alpha=0.7)
        else:
            ax.scatter(times, pits, alpha=0.5, s=20, color="blue")
        ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set(
            xlabel="Time", ylabel="PIT value", title="PIT Sequence", ylim=[-0.05, 1.05]
        )
        ax.legend()

        fig.suptitle(
            f"PIT Monitor Diagnostics (alpha={self.alpha})", fontsize=14, y=1.00
        )
        plt.tight_layout()
        return fig

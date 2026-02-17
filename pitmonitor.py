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

        # Conformal p-value with randomized tie-breaking for exact uniformity.
        left = bisect.bisect_left(self._sorted_pits, u)
        right = bisect.bisect_right(self._sorted_pits, u)
        rank = left + np.random.uniform(0, right - left + 1)
        p = rank / (self.t + 1)

        # Two-sided e-value from Jeffreys prior: Beta(1/2, 1/2) vs Uniform.
        # This is a canonical, parameter-free choice with E[e]=1 under H0.
        e = 1.0 / (np.pi * np.sqrt(p * (1 - p)))

        # Mixture e-process with a universal prior over changepoint times.
        # w_s = 1/((s+1)(s+2)) is a canonical summable prior on integers (sums to 1).
        # M_t = e_t * (M_{t-1} + w_{t-1}) is valid; P(sup M_t >= 1/alpha) <= alpha.
        s = self.t - 1
        w = 1.0 / ((s + 1) * (s + 2))
        self._mixture = e * (self._mixture + w)
        self._mixture_history.append(self._mixture)

        if self._mixture >= threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t
            self._alarm_info = AlarmInfo(
                triggered=True,
                alarm_time=self.t,
                martingale=self._mixture,
                threshold=threshold,
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

    @property
    def evidence(self):
        """Current e-process value (evidence against exchangeability)."""
        return self._mixture

    @staticmethod
    def _ecdf(data):
        """Compute empirical CDF."""
        x = np.sort(data)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    def plot(self, figsize=(14, 8), evidence_log_scale=True):
        """Create 4-panel diagnostic plot with principled statistical display."""
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

        # 1: PIT histogram or ECDF based on sample size
        ax = axes[0, 0]
        if self.t >= 30:
            # Use histogram with uncertainty band
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
                    pits,
                    bins=20,
                    density=True,
                    alpha=0.7,
                    color="blue",
                    edgecolor="black",
                )
            ax.axhline(
                1.0,
                color="red",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Uniform",
            )
            ax.set(xlabel="PIT value", ylabel="Density", title="PIT Distribution")
        else:
            # Use ECDF for small samples
            if cp and len(pits[:cp]) > 0 and len(pits[cp:]) > 0:
                x1, y1 = self._ecdf(pits[:cp])
                x2, y2 = self._ecdf(pits[cp:])
                ax.plot(x1, y1, "b-", lw=2, alpha=0.8, label=f"Before t={cp}")
                ax.plot(x2, y2, color="orange", lw=2, alpha=0.8, label=f"After t={cp}")
            else:
                x, y = self._ecdf(pits)
                ax.plot(x, y, "b-", lw=2, alpha=0.8)
            ax.plot([0, 1], [0, 1], "r--", lw=2, alpha=0.7, label="Uniform CDF")
            ax.set(
                xlabel="u",
                ylabel="F(u)",
                title=f"Empirical CDF (n < 30)",
                xlim=[0, 1],
                ylim=[0, 1],
            )
        ax.legend()

        # 2: Empirical CDFs with KS confidence band
        ax = axes[0, 1]
        band_x = np.linspace(0, 1, 200)
        eps = 1.36 / np.sqrt(self.t)
        band_lower = np.clip(band_x - eps, 0, 1)
        band_upper = np.clip(band_x + eps, 0, 1)

        if cp and len(pits[:cp]) > 0 and len(pits[cp:]) > 0:
            x1, y1 = self._ecdf(pits[:cp])
            x2, y2 = self._ecdf(pits[cp:])
            ax.plot(x1, y1, "b-", lw=2, alpha=0.8, label=f"Before t={cp}")
            ax.plot(x2, y2, color="orange", lw=2, alpha=0.8, label=f"After t={cp}")
        else:
            x, y = self._ecdf(pits)
            ax.plot(x, y, "b-", lw=2, alpha=0.8)
        ax.plot([0, 1], [0, 1], "r--", lw=2, alpha=0.7, label="Uniform CDF")
        ax.set(
            xlabel="u",
            ylabel="F(u)",
            title="Empirical CDFs with KS band",
            xlim=[0, 1],
            ylim=[0, 1],
        )
        ax.legend()

        # 3: Evidence over time
        ax = axes[1, 0]
        ax.plot(times, evidence, "b-", lw=2, label="Evidence")
        ax.axhline(
            1 / self.alpha, color="red", linestyle="--", lw=2, label=f"Threshold (1/α)"
        )
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
        y_label = "e-process (log scale)" if evidence_log_scale else "e-process"
        ax.set(
            xlabel="Time",
            ylabel=y_label,
            title="Evidence Against Exchangeability",
        )
        if evidence_log_scale:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        # 4: PIT sequence with post-changepoint shading
        ax = axes[1, 1]
        if cp:
            # Shade post-changepoint region
            ax.axvspan(cp, self.t, color="orange", alpha=0.05)
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

        # Add running mean
        running_mean = np.cumsum(pits) / times
        ax.plot(
            times, running_mean, color="black", lw=2, alpha=0.6, label="Running mean"
        )
        ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
        ax.set(
            xlabel="Time",
            ylabel="PIT value",
            title="PIT Sequence with Running Mean",
            ylim=[-0.05, 1.05],
        )
        ax.legend()

        fig.suptitle(
            f"PIT Monitor Diagnostics (α={self.alpha}, n={self.t})", fontsize=14, y=1.00
        )
        plt.tight_layout()
        return fig

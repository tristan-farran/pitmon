"""
PIT Monitor: Model-agnostic sequential validation via Probability Integral Transform

Core implementation of online model validity testing using the probability integral
transform (PIT) and sequential Kolmogorov-Smirnov monitoring.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
from collections import namedtuple


@dataclass
class AlarmInfo:
    """Information about a triggered alarm."""
    triggered: bool
    alarm_time: Optional[int] = None
    changepoint_estimate: Optional[int] = None
    ks_distance: Optional[float] = None
    threshold: Optional[float] = None
    diagnosis: Optional[str] = None
    
    def __bool__(self):
        return self.triggered


class PITMonitor:
    """
    Monitor probabilistic model validity via sequential PIT uniformity testing.
    
    A model is considered valid as long as the Probability Integral Transform (PIT)
    of observed outcomes under the model's predictive distribution remains uniformly
    distributed on [0,1]. This monitor performs sequential testing of PIT uniformity
    using the Kolmogorov-Smirnov distance with anytime-valid thresholds.
    
    Parameters
    ----------
    false_alarm_rate : float, default=0.05
        Maximum probability of false alarm over the monitoring period.
        This is the only tuning parameter and represents your risk tolerance.
        Typical values: 0.01 (conservative), 0.05 (standard), 0.10 (liberal)
    
    method : str, default='alpha_spending'
        Threshold computation method:
        - 'alpha_spending': Simple union bound (√(log t / t) scaling)
        - 'stitching': Epoch-based stitching (√(log log t / t) scaling, tighter)
    
    changepoint_budget : float, default=0.5
        Fraction of false_alarm_rate reserved for changepoint localization.
        Only used if you call localize_changepoint() after an alarm.
    
    Attributes
    ----------
    t : int
        Current time step (number of observations)
    pits : np.ndarray
        Stored PIT values
    alarm_triggered : bool
        Whether an alarm has been triggered
    alarm_time : int or None
        Time step when alarm was triggered
    
    Examples
    --------
    >>> monitor = PITMonitor(false_alarm_rate=0.05)
    >>> for prediction, outcome in data_stream:
    ...     alarm = monitor.update(prediction, outcome)
    ...     if alarm:
    ...         print(f"Model broke at t={monitor.t}")
    ...         cp = monitor.localize_changepoint()
    ...         print(f"Change detected around t={cp}")
    
    Notes
    -----
    The PIT is computed as U = F(Y) where F is the model's CDF and Y is the outcome.
    Under a correctly specified model, U ~ Uniform(0,1). Persistent deviation from
    uniformity indicates model misspecification or regime change.
    
    References
    ----------
    - Rosenblatt (1952): Remarks on a multivariate transformation
    - Dvoretzky, Kiefer, Wolfowitz (1956): Asymptotic minimax character of the 
      sample distribution function (DKW inequality)
    - Ramdas et al. (2020): Sequential estimation of quantiles with applications
      to A/B testing and best-arm identification
    """
    
    def __init__(
        self,
        false_alarm_rate: float = 0.05,
        method: str = 'alpha_spending',
        changepoint_budget: float = 0.5
    ):
        if not 0 < false_alarm_rate < 1:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        if method not in ['alpha_spending', 'stitching']:
            raise ValueError("method must be 'alpha_spending' or 'stitching'")
        if not 0 < changepoint_budget < 1:
            raise ValueError("changepoint_budget must be in (0, 1)")
            
        self.alpha = false_alarm_rate
        self.method = method
        self.changepoint_budget = changepoint_budget
        
        # State
        self.t = 0
        self.pits = []
        self.alarm_triggered = False
        self.alarm_time = None
        self._alarm_info = None
        
    def update(self, predicted_cdf: Callable[[float], float], outcome: float) -> AlarmInfo:
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
        
        self.pits.append(u)
        self.t += 1
        
        # Compute test statistic and threshold
        ks_dist = self._compute_ks_distance()
        threshold = self._compute_threshold()
        
        # Check for alarm
        if ks_dist > threshold:
            self.alarm_triggered = True
            self.alarm_time = self.t
            diagnosis = self._diagnose_deviation()
            
            self._alarm_info = AlarmInfo(
                triggered=True,
                alarm_time=self.t,
                ks_distance=ks_dist,
                threshold=threshold,
                diagnosis=diagnosis
            )
            return self._alarm_info
        
        return AlarmInfo(triggered=False, ks_distance=ks_dist, threshold=threshold)
    
    def _compute_ks_distance(self) -> float:
        """
        Compute Kolmogorov-Smirnov distance: sup_u |F_empirical(u) - u|
        
        Returns
        -------
        float
            KS distance (supremum deviation from uniform CDF)
        """
        if self.t == 0:
            return 0.0
        
        sorted_pits = np.sort(self.pits)
        k = np.arange(1, self.t + 1)
        
        # KS distance is max over all u of |empirical CDF - uniform CDF|
        # The supremum occurs at the observed PIT values
        deviations = np.abs(k / self.t - sorted_pits)
        return float(np.max(deviations))
    
    def _compute_threshold(self) -> float:
        """
        Compute time-varying threshold for KS distance.
        
        Returns
        -------
        float
            Threshold value epsilon_t
        """
        if self.t == 0:
            return np.inf
        
        if self.method == 'alpha_spending':
            return self._threshold_alpha_spending()
        else:  # stitching
            return self._threshold_stitching()
    
    def _threshold_alpha_spending(self) -> float:
        """
        Simple alpha-spending threshold: sqrt(log(t) / t) scaling.
        
        Uses alpha_t = alpha / (pi^2 * t^2) and applies DKW inequality.
        """
        alpha_t = self.alpha / (np.pi**2 * self.t**2)
        return np.sqrt(np.log(2 / alpha_t) / (2 * self.t))
    
    def _threshold_stitching(self) -> float:
        """
        Stitched threshold: sqrt(log log(t) / t) scaling.
        
        Uses epoch-based alpha spending to achieve log-log scaling.
        """
        # Find current epoch j where t in [2^j, 2^{j+1})
        j = int(np.floor(np.log2(max(self.t, 1))))
        
        # Alpha spending per epoch: alpha_j = alpha / (pi^2 * j^2)
        # (for j >= 1; handle j=0 specially)
        if j == 0:
            alpha_j = self.alpha / 2
        else:
            alpha_j = self.alpha / (np.pi**2 * j**2)
        
        # Use threshold based on epoch start (2^j)
        epoch_start = 2**j
        return np.sqrt(np.log(2 / alpha_j) / (2 * epoch_start))
    
    def _diagnose_deviation(self) -> str:
        """
        Diagnose the type of deviation from uniformity.
        
        Returns
        -------
        str
            Human-readable diagnosis of the deviation pattern
        """
        sorted_pits = np.sort(self.pits)
        k = np.arange(1, self.t + 1)
        deviations = k / self.t - sorted_pits
        
        # Find where maximum deviation occurs
        max_idx = np.argmax(np.abs(deviations))
        max_deviation = deviations[max_idx]
        u_star = sorted_pits[max_idx]
        
        # Characterize the deviation
        parts = []
        
        # Location in distribution
        if u_star < 0.1:
            parts.append("lower tail")
        elif u_star > 0.9:
            parts.append("upper tail")
        else:
            parts.append("central region")
        
        # Direction of deviation
        if max_deviation > 0:
            parts.append("underconfident" if u_star < 0.5 else "overconfident")
            parts.append("(observed values more extreme than predicted)")
        else:
            parts.append("overconfident" if u_star < 0.5 else "underconfident")
            parts.append("(observed values less extreme than predicted)")
        
        return " - ".join(parts)
    
    def localize_changepoint(self, method: str = 'backward_scan') -> Optional[int]:
        """
        Estimate when the model started deviating (changepoint localization).
        
        Should be called after an alarm has been triggered.
        
        Parameters
        ----------
        method : str, default='backward_scan'
            Method for localization:
            - 'backward_scan': Scan backwards from alarm time
            - 'binary_search': Binary search over possible changepoints
            
        Returns
        -------
        int or None
            Estimated changepoint time, or None if no alarm triggered
            
        Notes
        -----
        Uses a reserved portion of the error budget (changepoint_budget * alpha)
        to perform valid post-alarm changepoint estimation.
        """
        if not self.alarm_triggered:
            return None
        
        if method == 'backward_scan':
            return self._localize_backward_scan()
        elif method == 'binary_search':
            return self._localize_binary_search()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _localize_backward_scan(self) -> int:
        """
        Backward scan for changepoint: find earliest segment with deviation.
        
        Scans geometrically spaced segment lengths for efficiency.
        """
        alpha_cp = self.alpha * self.changepoint_budget
        t_alarm = self.alarm_time
        pits_array = np.array(self.pits)
        
        # Geometric scan over segment lengths
        segment_lengths = self._geometric_sequence(1, t_alarm)
        
        for n in segment_lengths:
            s = t_alarm - n  # segment start
            if s < 0:
                continue
                
            # Compute KS distance on segment [s, t_alarm]
            segment_pits = pits_array[s:t_alarm]
            segment_ks = self._compute_ks_on_segment(segment_pits)
            
            # Threshold for this segment length
            alpha_n = alpha_cp / (np.pi**2 * int(np.log2(n) + 1)**2) if n > 1 else alpha_cp / 2
            threshold_n = np.sqrt(np.log(2 / alpha_n) / (2 * n))
            
            # If this segment shows deviation, we've found the changepoint region
            if segment_ks > threshold_n:
                return s + 1  # +1 for 1-indexed time
        
        # Default to alarm time if no earlier changepoint found
        return 1
    
    def _localize_binary_search(self) -> int:
        """
        Binary search for changepoint location.
        """
        # Simplified version: return midpoint between start and alarm
        # (Full implementation would do proper binary search with valid testing)
        return max(1, self.alarm_time // 2)
    
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
        """
        Get current monitor state for inspection or serialization.
        
        Returns
        -------
        dict
            Current state including PITs, statistics, and alarm status
        """
        return {
            't': self.t,
            'pits': self.pits.copy(),
            'ks_distance': self._compute_ks_distance() if self.t > 0 else None,
            'threshold': self._compute_threshold() if self.t > 0 else None,
            'alarm_triggered': self.alarm_triggered,
            'alarm_time': self.alarm_time,
            'alpha': self.alpha,
            'method': self.method
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
            raise ImportError("matplotlib required for plotting. Install with: pip install matplotlib")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        if self.t == 0:
            fig.suptitle("No data yet")
            return fig
        
        pits_array = np.array(self.pits)
        times = np.arange(1, self.t + 1)
        
        # Plot 1: PIT histogram
        ax = axes[0, 0]
        ax.hist(pits_array, bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='red', linestyle='--', label='Uniform')
        ax.set_xlabel('PIT value')
        ax.set_ylabel('Density')
        ax.set_title('PIT Distribution (should be uniform)')
        ax.legend()
        
        # Plot 2: Empirical CDF vs Uniform
        ax = axes[0, 1]
        sorted_pits = np.sort(pits_array)
        empirical_cdf = np.arange(1, self.t + 1) / self.t
        ax.plot(sorted_pits, empirical_cdf, 'b-', label='Empirical CDF', linewidth=2)
        ax.plot([0, 1], [0, 1], 'r--', label='Uniform CDF', linewidth=2)
        
        # Add confidence band
        threshold = self._compute_threshold()
        u_grid = np.linspace(0, 1, 100)
        ax.fill_between(u_grid, 
                        np.maximum(0, u_grid - threshold),
                        np.minimum(1, u_grid + threshold),
                        alpha=0.3, color='gray', label=f'{1-self.alpha:.0%} confidence band')
        ax.set_xlabel('u')
        ax.set_ylabel('F(u)')
        ax.set_title('Empirical CDF vs Uniform')
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Plot 3: KS distance over time
        ax = axes[1, 0]
        ks_history = []
        threshold_history = []
        for i in range(1, self.t + 1):
            temp_pits = pits_array[:i]
            ks_history.append(self._compute_ks_on_segment(temp_pits))
            
            # Recompute threshold at each time
            old_t = self.t
            self.t = i
            threshold_history.append(self._compute_threshold())
            self.t = old_t
        
        ax.plot(times, ks_history, 'b-', label='KS distance', linewidth=2)
        ax.plot(times, threshold_history, 'r--', label='Threshold', linewidth=2)
        if self.alarm_triggered:
            ax.axvline(self.alarm_time, color='orange', linestyle=':', 
                      label=f'Alarm (t={self.alarm_time})', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('KS distance')
        ax.set_title('Sequential Monitoring')
        ax.legend()
        ax.set_yscale('log')
        
        # Plot 4: PIT sequence over time
        ax = axes[1, 1]
        ax.scatter(times, pits_array, alpha=0.5, s=20)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        ax.axhline(0.0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(1.0, color='gray', linestyle='-', alpha=0.3)
        if self.alarm_triggered:
            ax.axvline(self.alarm_time, color='orange', linestyle=':', 
                      label=f'Alarm (t={self.alarm_time})', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('PIT value')
        ax.set_title('PIT Sequence')
        ax.set_ylim([-0.05, 1.05])
        if self.alarm_triggered:
            ax.legend()
        
        fig.suptitle(f'PIT Monitor Diagnostics (α={self.alpha}, method={self.method})', 
                    fontsize=14, y=1.00)
        plt.tight_layout()
        
        return fig

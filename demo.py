"""
Detecting changes in a latent process using PITs and a conformal e-process.

This demo simulates a return series with a volatility regime change. A fixed
GARCH(1,1) forecaster produces predictive distributions, and PIT values are
monitored for evidence of distributional change using a conformal e-process
(via PITMonitor). The forecaster is intentionally kept fixed to model a deployed
system that cannot adapt online.

Key steps:
1. Simulate returns with a regime change in volatility
2. Apply fixed GARCH(1,1) forecaster to get predictive distributions
3. Compute PIT values from predictions
4. Monitor PITs vs a CUSUM baseline
5. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt
from pitmonitor import PITMonitor


# ============================================================================
# 1. SIMULATE DATA
# ============================================================================


class GARCHRegimeSimulator:
    """
    Simulate returns with a regime change in GARCH(1,1) parameters.

    r_t = mu + sqrt(sigma2_t) * z_t, z_t ~ N(0,1)
    sigma2_t = omega + alpha * (r_{t-1} - mu)^2 + beta * sigma2_{t-1}
    """

    def __init__(
        self,
        omega_before=1e-4,
        alpha_before=0.05,
        beta_before=0.9,
        omega_after=4e-4,
        alpha_after=0.12,
        beta_after=0.82,
        mu_before=0.0,
        mu_after=0.0,
        t_changepoint=100,
        seed=42,
    ):
        self.omega_before = omega_before
        self.alpha_before = alpha_before
        self.beta_before = beta_before
        self.omega_after = omega_after
        self.alpha_after = alpha_after
        self.beta_after = beta_after
        self.mu_before = mu_before
        self.mu_after = mu_after
        self.t_changepoint = t_changepoint
        self.t = 0

        self._rng = np.random.RandomState(seed)
        self._sigma2 = self._uncond_var(
            self.omega_before, self.alpha_before, self.beta_before
        )
        self._prev_return = self.mu_before

    @staticmethod
    def _uncond_var(omega, alpha, beta):
        denom = max(1.0 - alpha - beta, 1e-6)
        return omega / denom

    def _params(self):
        if self.t >= self.t_changepoint:
            return (
                self.omega_after,
                self.alpha_after,
                self.beta_after,
                self.mu_after,
            )
        return (
            self.omega_before,
            self.alpha_before,
            self.beta_before,
            self.mu_before,
        )

    def step(self):
        omega, alpha, beta, mu = self._params()
        self._sigma2 = (
            omega + alpha * (self._prev_return - mu) ** 2 + beta * self._sigma2
        )
        shock = self._rng.randn()
        ret = mu + np.sqrt(self._sigma2) * shock
        self._prev_return = ret
        self.t += 1
        return ret, self._sigma2, mu

    def simulate(self, T):
        rets = np.zeros(T)
        sig2 = np.zeros(T)
        mu_series = np.zeros(T)
        for i in range(T):
            rets[i], sig2[i], mu_series[i] = self.step()
        return rets, sig2, mu_series


# ============================================================================
# 2. GARCH FORECASTER
# ============================================================================


class GARCHForecaster:
    """Fixed GARCH(1,1) forecaster using pre-change parameters only."""

    def __init__(self, omega, alpha, beta, mu=0.0):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.sigma2 = self._uncond_var(omega, alpha, beta)
        self.prev_return = mu

    @staticmethod
    def _uncond_var(omega, alpha, beta):
        denom = max(1.0 - alpha - beta, 1e-6)
        return omega / denom

    def predict_next_var(self):
        return (
            self.omega
            + self.alpha * (self.prev_return - self.mu) ** 2
            + self.beta * self.sigma2
        )

    def update(self, ret):
        self.sigma2 = (
            self.omega + self.alpha * (ret - self.mu) ** 2 + self.beta * self.sigma2
        )
        self.prev_return = ret


class CusumDetector:
    """One-sided CUSUM detector on standardized squared residuals."""

    def __init__(self, k=0.05, h=8.0):
        self.k = k
        self.h = h
        self.s = 0.0
        self.alarm_time = None
        self.history = []

    def update(self, z2, t):
        self.s = max(0.0, self.s + (z2 - 1.0 - self.k))
        self.history.append(self.s)
        if self.alarm_time is None and self.s > self.h:
            self.alarm_time = t
        return self.alarm_time


# ============================================================================
# 3. PIT COMPUTATION
# ============================================================================


def compute_pit(pred_mean, pred_var, observation):
    """
    Compute PIT (Probability Integral Transform) value.

    PIT = CDF of N(pred_mean, pred_var) evaluated at observation.

    Parameters
    ----------
    pred_mean : float
        Predicted mean.
    pred_var : float
        Predicted variance.
    observation : float
        Observed value.

    Returns
    -------
    pit : float
        PIT value in [0, 1].
    """
    pred_std = np.sqrt(pred_var)
    z = (observation - pred_mean) / pred_std
    pit = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    return pit


def ks_distance_uniform(samples):
    """Compute Kolmogorov-Smirnov distance to Uniform[0,1]."""
    if len(samples) == 0:
        return np.nan
    x = np.sort(samples)
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    return np.max(np.abs(ecdf - x))


# ============================================================================
# 4. MAIN DEMO
# ============================================================================


def main():
    print("=" * 80)
    print("CHANGEPOINT DETECTION VIA PITs AND CONFORMAL E-PROCESS")
    print("=" * 80)
    print()

    # ========================================================================
    # Configuration
    # ========================================================================

    T_total = 800  # Total time steps
    t_changepoint = 200  # When the regime shifts
    mu_before = 0.0
    mu_after = 0.0
    omega_before = 1e-4
    alpha_before = 0.05
    beta_before = 0.9
    omega_after = 4e-4
    alpha_after = 0.12
    beta_after = 0.82
    false_alarm_rate = 0.1
    cusum_k = 0.05
    cusum_h = 8.0
    seeds = list(range(20))

    print(f"Simulation Configuration:")
    print(f"  Total steps: {T_total}")
    print(f"  Changepoint: t={t_changepoint}")
    print(
        "  GARCH before: omega={:.1e}, alpha={:.2f}, beta={:.2f}".format(
            omega_before, alpha_before, beta_before
        )
    )
    print(
        "  GARCH after:  omega={:.1e}, alpha={:.2f}, beta={:.2f}".format(
            omega_after, alpha_after, beta_after
        )
    )
    print(f"  Mean before/after: {mu_before} / {mu_after}")
    print(f"  CUSUM k/h: {cusum_k} / {cusum_h}")
    print(f"  False alarm rate (alpha): {false_alarm_rate}")
    print(f"  Seeds: {seeds}")
    print()

    # ========================================================================
    # Step 1: Simulate data
    # ========================================================================

    def run_trial(seed, verbose=False):
        simulator = GARCHRegimeSimulator(
            omega_before=omega_before,
            alpha_before=alpha_before,
            beta_before=beta_before,
            omega_after=omega_after,
            alpha_after=alpha_after,
            beta_after=beta_after,
            mu_before=mu_before,
            mu_after=mu_after,
            t_changepoint=t_changepoint,
            seed=seed,
        )
        returns, true_sig2, mu_series = simulator.simulate(T_total)

        forecaster = GARCHForecaster(
            omega=omega_before,
            alpha=alpha_before,
            beta=beta_before,
            mu=mu_before,
        )
        monitor = PITMonitor(false_alarm_rate=false_alarm_rate)
        cusum = CusumDetector(k=cusum_k, h=cusum_h)

        pits_list = []
        pred_means = np.full(T_total, forecaster.mu)
        pred_vars = np.zeros(T_total)

        for t in range(T_total):
            pred_var = forecaster.predict_next_var()
            pred_vars[t] = pred_var
            pit = compute_pit(pred_means[t], pred_var, returns[t])
            pits_list.append(pit)

            if not monitor.alarm_triggered:
                alarm_info = monitor.update_pit(pit)
                if alarm_info.triggered and verbose:
                    print(f"  PIT alarm at t={t+1}")
                    print(f"    Evidence (e-process): {alarm_info.martingale:.2f}")
                    print(f"    Threshold (1/alpha): {alarm_info.threshold:.2f}")

            z2 = ((returns[t] - pred_means[t]) ** 2) / max(pred_var, 1e-12)
            cusum.update(z2, t + 1)

            forecaster.update(returns[t])

        pits_array = np.array(pits_list)
        return {
            "monitor": monitor,
            "cusum": cusum,
            "pred_means": pred_means,
            "pred_vars": pred_vars,
            "returns": returns,
            "pits": pits_array,
            "true_sig2": true_sig2,
        }

    print("Step 1-3: Running trials (PITMonitor + CUSUM)...")
    trial_results = []
    for seed in seeds:
        trial_results.append(run_trial(seed, verbose=False))
    print(f"  Completed {len(trial_results)} trials")
    print()

    primary = trial_results[0]
    monitor = primary["monitor"]
    returns = primary["returns"]
    pits_array = primary["pits"]
    pred_means = primary["pred_means"]
    pred_vars = primary["pred_vars"]
    true_sig2 = primary["true_sig2"]
    cusum = primary["cusum"]

    print()
    print("Results (seed 0):")
    print(f"  PIT alarm: {monitor.alarm_time}")
    print(f"  CUSUM alarm: {cusum.alarm_time}")
    if monitor.alarm_triggered:
        changepoint_est = monitor.localize_changepoint()
        print(f"  Estimated changepoint (PIT): t={changepoint_est}")
        print(f"  True changepoint: t={t_changepoint}")
        if changepoint_est:
            error = abs(changepoint_est - t_changepoint)
            print(f"  Estimation error: {error} steps")
    print()

    pit_alarm_times = [
        result["monitor"].alarm_time
        for result in trial_results
        if result["monitor"].alarm_time
    ]
    cusum_alarm_times = [
        result["cusum"].alarm_time
        for result in trial_results
        if result["cusum"].alarm_time
    ]
    pit_detection_rate = len(pit_alarm_times) / len(trial_results)
    cusum_detection_rate = len(cusum_alarm_times) / len(trial_results)
    pit_median_alarm = int(np.median(pit_alarm_times)) if pit_alarm_times else None
    cusum_median_alarm = (
        int(np.median(cusum_alarm_times)) if cusum_alarm_times else None
    )
    pit_delays = [t - t_changepoint for t in pit_alarm_times]
    cusum_delays = [t - t_changepoint for t in cusum_alarm_times]
    pit_median_delay = int(np.median(pit_delays)) if pit_delays else None
    cusum_median_delay = int(np.median(cusum_delays)) if cusum_delays else None
    print("Results (all seeds):")
    print("  Detector        rate   median_t   median_delay")
    print(
        "  PITMonitor      {:>4.2f}   {:>8}   {:>12}".format(
            pit_detection_rate, str(pit_median_alarm), str(pit_median_delay)
        )
    )
    print(
        "  CUSUM(z^2-1)    {:>4.2f}   {:>8}   {:>12}".format(
            cusum_detection_rate, str(cusum_median_alarm), str(cusum_median_delay)
        )
    )
    print()

    # ========================================================================
    # Step 4: Diagnostics
    # ========================================================================

    print("Step 4: Computing diagnostics...")

    mse = (returns - pred_means) ** 2
    before_mask = np.arange(T_total) < t_changepoint
    after_mask = np.arange(T_total) >= t_changepoint
    mse_before = np.mean(mse[before_mask]) if np.any(before_mask) else np.nan
    mse_after = np.mean(mse[after_mask]) if np.any(after_mask) else np.nan
    print("  Mean squared error before/after:")
    print(f"    {mse_before:.4f}  {mse_after:.4f}")
    print()

    pits_before = pits_array[:t_changepoint]
    pits_after = pits_array[t_changepoint:]
    ks_before = ks_distance_uniform(pits_before)
    ks_after = ks_distance_uniform(pits_after)
    print("  K-S distance to uniform:")
    print(f"    Before changepoint: {ks_before:.4f}")
    print(f"    After changepoint: {ks_after:.4f}")
    print()

    # ========================================================================
    # Step 5: Visualization
    # ========================================================================

    print("Step 5: Creating visualizations...")
    times = np.arange(1, T_total + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(times, returns, color="black", lw=0.8, alpha=0.7)
    ax.axvline(t_changepoint, color="red", ls="--", lw=2, label="Regime shift")
    ax.set(title="Returns", xlabel="Time", ylabel="Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, np.sqrt(true_sig2), color="orange", lw=1.5, label="True sigma")
    ax.plot(
        times, np.sqrt(pred_vars), color="blue", lw=1.5, alpha=0.8, label="Model sigma"
    )
    ax.axvline(t_changepoint, color="red", ls="--", lw=2)
    ax.set(title="Volatility: True vs Model", xlabel="Time", ylabel="Sigma")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(times, pits_array, s=15, alpha=0.5, color="blue")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.6)
    ax.axvline(t_changepoint, color="red", ls="--", lw=2)
    ax.set(title="PIT sequence", xlabel="Time", ylabel="PIT")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    evidence = np.array(monitor._mixture_history)
    if len(evidence) > 0:
        ax.plot(
            np.arange(1, len(evidence) + 1),
            evidence,
            color="blue",
            lw=2,
            label="PIT e-process",
        )
        ax.axhline(1 / monitor.alpha, color="red", ls="--", lw=2, label="PIT threshold")
    ax.plot(times, cusum.history, color="purple", lw=1.5, label="CUSUM(z^2-1)")
    ax.axhline(cusum_h, color="purple", ls=":", lw=1.5, label="CUSUM threshold")
    ax.axvline(t_changepoint, color="red", ls="--", lw=2)
    ax.set(title="Detectors", xlabel="Time", ylabel="Statistic")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Regime Change Detection via PITs vs CUSUM", fontsize=14)
    plt.tight_layout()
    plt.show()

    print()
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

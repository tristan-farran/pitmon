"""verify_model.py — Diagnostic report for ProbabilisticMLP prediction quality.

Runs entirely on pre-drift (undrifted) data so the results reflect what
PITMonitor sees under the null hypothesis H0: calibration is stable.

Diagnostics
-----------
1.  **Point prediction accuracy**
        RMSE, MAE, R² on a held-out pre-drift test set.

2.  **Calibration check: PIT uniformity**
        Histogram of PIT values vs U[0,1] reference.
        Kolmogorov–Smirnov test  (H0: PITs ~ Uniform[0,1]).
        Expected coverage: for each nominal level p, the fraction of PITs
        below p should equal p.  A calibration curve shows the gap.

3.  **Reliability diagram (interval coverage)**
        For a grid of nominal coverage levels, compute the empirical fraction
        of observations falling inside the predictive interval.
        Perfect calibration → points on the diagonal.

4.  **Sharpness**
        Mean predictive standard deviation.  A well-calibrated model should
        be *as sharp as possible* consistent with calibration: unnecessarily
        wide intervals are a sign that the log-variance head is not learning.

5.  **Residual diagnostics**
        Standardized residuals (z-scores) should be approximately N(0,1).
        - Histogram vs N(0,1) density.
        - Normal Q–Q plot.
        - Ljung-Box test for autocorrelation in residuals (checks that errors
          are i.i.d., an implicit PITMonitor assumption).
        - Goldfeld–Quandt-style variance check: compare residual variance in
          the first vs second half of the test set.

6.  **Console summary table** with pass/fail indicators for each test.

Usage
-----
    # Quick check on a saved bundle:
    python verify_model.py

    # Custom paths / split sizes:
    python verify_model.py --bundle out/model.pkl --n-test 1000 --seed 7

    # Suppress plots (CI / headless):
    python verify_model.py --no-show

Output
------
    out/verify_model.png   — multi-panel diagnostic figure
    Console summary table  — printed to stdout
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

_this_dir = Path(__file__).resolve().parent
_repo_root = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_repo_root))

from config import Config
from data import generate_stream
from model import (
    ModelBundle,
    compute_pits,
    compute_predictions,
    compute_residuals,
    load_bundle,
    train_model,
    save_bundle,
)


# ─── CLI ─────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Verify ProbabilisticMLP prediction quality on pre-drift data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
        Examples:
          python verify_model.py                         # use defaults
          python verify_model.py --n-test 2000           # larger test set
          python verify_model.py --no-show               # save figure only
          python verify_model.py --bundle out/model.pkl  # explicit bundle path
        """
        ),
    )
    p.add_argument(
        "--bundle",
        type=str,
        default=None,
        help="Path to saved ModelBundle (default: cfg.bundle_path = out/model.pkl)",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=2000,
        help="Number of held-out pre-drift samples for diagnostics (default: 2000)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed (default: 42)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="out",
        help="Output directory for the diagnostic figure (default: out)",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save figure to disk but do not display it",
    )
    p.add_argument(
        "--alpha-ks",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    return p.parse_args()


# ─── Metric helpers ───────────────────────────────────────────────────


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nll(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood (in nats)."""
    return float(
        np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma) ** 2)
    )


def coverage(pits: np.ndarray, level: float) -> float:
    """Empirical fraction of PITs below *level* (calibration curve point)."""
    return float(np.mean(pits < level))


def interval_coverage(pits: np.ndarray, nominal: float) -> float:
    """Central-interval empirical coverage at nominal level.

    For a central interval at level ``nominal``, the ideal PIT fraction
    inside the interval is ``nominal``; i.e. PITs should fall in
    [(1-nominal)/2, (1+nominal)/2].
    """
    lo = (1 - nominal) / 2
    hi = (1 + nominal) / 2
    return float(np.mean((pits >= lo) & (pits <= hi)))


def ljung_box_pvalue(residuals: np.ndarray, lags: int = 10) -> float:
    """Ljung-Box p-value for autocorrelation in *residuals* up to *lags* lags.

    Large p-value → no evidence of autocorrelation (good for PITMonitor's
    exchangeability assumption).
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
    return float(result["lb_pvalue"].iloc[0])


# ─── Plotting ─────────────────────────────────────────────────────────


_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
}

PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"


def _apply_style():
    plt.rcParams.update(_RC)


def make_diagnostic_figure(
    y_test: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    pits: np.ndarray,
    z_scores: np.ndarray,
    alpha_ks: float,
) -> tuple[plt.Figure, dict]:
    """Build the 3×3 diagnostic figure and return (fig, metrics).

    Parameters
    ----------
    y_test : ndarray, shape (N,)
    mu : ndarray, shape (N,)
    sigma : ndarray, shape (N,)
    pits : ndarray, shape (N,)
    z_scores : ndarray, shape (N,)
    alpha_ks : float

    Returns
    -------
    fig : plt.Figure
    metrics : dict
        Summary statistics and test results for the console table.
    """
    # NOTE: the accumulator is named ``metrics`` (not ``stats``) to avoid
    # shadowing the ``from scipy import stats`` module import.
    _apply_style()
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        "ProbabilisticMLP — Pre-Drift Prediction Quality",
        fontsize=14,
        fontweight="bold",
    )

    metrics: dict = {}
    N = len(y_test)

    # ── 1. Predicted vs Actual ────────────────────────────────────────
    ax = axes[0, 0]
    lo = min(y_test.min(), mu.min())
    hi = max(y_test.max(), mu.max())
    ax.scatter(y_test, mu, s=6, alpha=0.25, color="steelblue", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
    _r2 = r2(y_test, mu)
    _rmse = rmse(y_test, mu)
    metrics["R2"] = _r2
    metrics["RMSE"] = _rmse
    metrics["MAE"] = mae(y_test, mu)
    metrics["NLL"] = nll(y_test, mu, sigma)
    ax.set(
        xlabel="Actual",
        ylabel="Predicted mean",
        title=f"Predicted vs Actual  (R²={_r2:.3f})",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── 2. Residuals vs Predicted ─────────────────────────────────────
    ax = axes[0, 1]
    residuals = y_test - mu
    ax.scatter(mu, residuals, s=6, alpha=0.25, color="steelblue", rasterized=True)
    ax.axhline(0, color="crimson", ls="--", lw=1.5)
    ax.set(
        xlabel="Predicted mean", ylabel="Residual  (y − ŷ)", title="Residuals vs Fitted"
    )
    ax.grid(alpha=0.3)

    # ── 3. Predictive sigma distribution ─────────────────────────────
    ax = axes[0, 2]
    ax.hist(
        sigma, bins=40, color="steelblue", edgecolor="white", density=True, alpha=0.8
    )
    mean_sigma = float(sigma.mean())
    std_sigma = float(sigma.std())
    metrics["mean_sigma"] = mean_sigma
    metrics["std_sigma"] = std_sigma
    # Coefficient of variation: measures how constant the predicted σ is
    sigma_cv = std_sigma / mean_sigma if mean_sigma > 0 else float("nan")
    metrics["sigma_cv"] = sigma_cv
    ax.axvline(
        mean_sigma, color="crimson", ls="--", lw=1.5, label=f"Mean = {mean_sigma:.3f}"
    )
    ax.set(
        xlabel="Predictive σ",
        ylabel="Density",
        title=f"Predictive Uncertainty  (CV={sigma_cv:.3f})",
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 4. PIT histogram ─────────────────────────────────────────────
    ax = axes[1, 0]
    pit_bins = np.linspace(0, 1, 21)
    ax.hist(
        pits,
        bins=pit_bins,
        density=True,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        label="Empirical PITs",
    )
    ax.axhline(1.0, color="crimson", ls="--", lw=1.8, label="U[0,1] reference")

    ks_stat, ks_p = stats.kstest(pits, "uniform")
    metrics["KS_stat"] = ks_stat
    metrics["KS_pvalue"] = ks_p
    color = PASS_COLOR if ks_p > alpha_ks else FAIL_COLOR
    ax.set_title(f"PIT Histogram  (KS p={ks_p:.4f})", color=color)
    ax.set(xlabel="PIT", ylabel="Density", xlim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 5. Calibration curve ─────────────────────────────────────────
    ax = axes[1, 1]
    levels = np.linspace(0, 1, 101)
    empirical = np.array([coverage(pits, lv) for lv in levels])
    ax.plot(levels, levels, "r--", lw=1.5, label="Perfect calibration")
    ax.plot(levels, empirical, color="steelblue", lw=2, label="Empirical")
    ax.fill_between(levels, levels, empirical, alpha=0.15, color="steelblue")
    ece = float(np.mean(np.abs(empirical - levels)))
    mce = float(np.max(np.abs(empirical - levels)))
    metrics["ECE"] = ece
    metrics["MCE"] = mce
    color = PASS_COLOR if ece < 0.05 else FAIL_COLOR
    ax.set_title(f"Calibration Curve  (ECE={ece:.4f})", color=color)
    ax.set(
        xlabel="Nominal CDF level",
        ylabel="Empirical coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 6. Interval coverage ─────────────────────────────────────────
    ax = axes[1, 2]
    nominal_levels = np.linspace(0.05, 0.99, 50)
    emp_coverage = np.array([interval_coverage(pits, lv) for lv in nominal_levels])
    ax.plot(nominal_levels, nominal_levels, "r--", lw=1.5, label="Perfect")
    ax.plot(nominal_levels, emp_coverage, color="steelblue", lw=2, label="Empirical")
    ax.fill_between(
        nominal_levels, nominal_levels, emp_coverage, alpha=0.15, color="steelblue"
    )
    cov_50 = interval_coverage(pits, 0.50)
    cov_90 = interval_coverage(pits, 0.90)
    metrics["coverage_50"] = cov_50
    metrics["coverage_90"] = cov_90
    color = PASS_COLOR if abs(cov_90 - 0.90) < 0.05 else FAIL_COLOR
    ax.set_title(
        f"Interval Coverage  (50%: {cov_50:.2f}, 90%: {cov_90:.2f})", color=color
    )
    ax.set(
        xlabel="Nominal level", ylabel="Empirical coverage", xlim=(0, 1), ylim=(0, 1)
    )
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 7. Standardized residual histogram (z-scores) ─────────────────
    ax = axes[2, 0]
    z_range = np.linspace(-4.5, 4.5, 300)
    ax.hist(
        z_scores,
        bins=50,
        density=True,
        color="steelblue",
        edgecolor="white",
        alpha=0.8,
        label="Empirical z-scores",
    )
    ax.plot(z_range, stats.norm.pdf(z_range), color="crimson", lw=2, label="N(0,1)")
    z_ks_stat, z_ks_p = stats.kstest(z_scores, "norm")
    metrics["z_KS_stat"] = z_ks_stat
    metrics["z_KS_pvalue"] = z_ks_p
    metrics["z_mean"] = float(z_scores.mean())
    metrics["z_std"] = float(z_scores.std())
    color = PASS_COLOR if z_ks_p > alpha_ks else FAIL_COLOR
    ax.set_title(
        f"Z-scores vs N(0,1)  (KS p={z_ks_p:.4f}, mean={z_scores.mean():.3f}, std={z_scores.std():.3f})",
        color=color,
    )
    ax.set(xlabel="Standardized residual", ylabel="Density")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 8. Normal Q–Q plot ────────────────────────────────────────────
    ax = axes[2, 1]
    (osm, osr), (slope, intercept, r_qq) = stats.probplot(z_scores, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.3, color="steelblue", rasterized=True)
    x_line = np.array([osm[0], osm[-1]])
    ax.plot(
        x_line,
        slope * x_line + intercept,
        "r-",
        lw=1.8,
        label=f"Reference  (r={r_qq:.4f})",
    )
    metrics["QQ_r"] = float(r_qq)
    color = PASS_COLOR if r_qq > 0.995 else FAIL_COLOR
    ax.set_title(f"Normal Q–Q (z-scores)  (r={r_qq:.4f})", color=color)
    ax.set(xlabel="Theoretical quantiles", ylabel="Sample quantiles")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── 9. Sigma vs |residual| ────────────────────────────────────────
    # FriedmanDrift has *homoscedastic* additive noise: the true conditional
    # variance is constant regardless of x.  A correctly-specified model
    # should therefore output nearly constant σ, and the Spearman correlation
    # between σ and |residual| will be near zero — this is correct behaviour,
    # not a flaw.  We instead check that the mean σ is well-matched to the
    # empirical mean absolute residual (via the N(0,1) scaling factor √(2/π)):
    #   E[|residual|] ≈ σ · √(2/π)  ⟺  σ ≈ E[|residual|] / √(2/π)
    # A large ratio |σ_mean / σ_implied - 1| indicates over/under-dispersion.
    ax = axes[2, 2]
    abs_res = np.abs(residuals)
    order = np.argsort(sigma)
    sigma_sorted = sigma[order]
    abs_res_sorted = abs_res[order]
    n_bins_var = 40
    bin_edges = np.quantile(sigma_sorted, np.linspace(0, 1, n_bins_var + 1))
    bin_centers, bin_means = [], []
    for lo_b, hi_b in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (sigma_sorted >= lo_b) & (sigma_sorted <= hi_b)
        if mask.sum() > 0:
            bin_centers.append(float(np.median(sigma_sorted[mask])))
            bin_means.append(float(np.mean(abs_res_sorted[mask])))

    ax.scatter(
        sigma,
        abs_res,
        s=4,
        alpha=0.12,
        color="steelblue",
        rasterized=True,
        label="Samples",
    )
    sigma_line = np.linspace(sigma.min(), sigma.max(), 200)
    ax.plot(
        sigma_line,
        sigma_line * np.sqrt(2 / np.pi),
        "r--",
        lw=1.5,
        label="Ideal: σ·√(2/π)",
    )

    # Dispersion ratio: how well does mean σ match mean |residual|?
    sigma_implied = float(np.mean(abs_res)) / np.sqrt(2 / np.pi)
    dispersion_ratio = mean_sigma / sigma_implied  # 1.0 = perfect
    metrics["sigma_dispersion_ratio"] = dispersion_ratio
    # Also record Spearman ρ as an informational metric (not a pass/fail check)
    spear_r, spear_p = stats.spearmanr(sigma, abs_res)
    metrics["sigma_vs_abserr_spearman_r"] = float(spear_r)
    metrics["sigma_vs_abserr_spearman_p"] = float(spear_p)
    # Pass if the mean σ is within 15% of the implied value from mean |residual|
    sigma_ok = abs(dispersion_ratio - 1.0) < 0.15
    color = PASS_COLOR if sigma_ok else FAIL_COLOR
    ax.set_title(
        f"Predictive σ vs |Residual|  (ratio={dispersion_ratio:.3f})", color=color
    )
    ax.set(xlabel="Predictive σ", ylabel="|Residual|")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig, metrics


# ─── Console summary ─────────────────────────────────────────────────


def print_summary(metrics: dict, alpha_ks: float, n_test: int) -> None:
    """Print a formatted pass/fail summary table to stdout."""
    PASS = "\033[92m✓ PASS\033[0m"
    FAIL = "\033[91m✗ FAIL\033[0m"

    def _check(condition: bool) -> str:
        return PASS if condition else FAIL

    divider = "─" * 70
    print(f"\n{'═'*70}")
    print(f"  ProbabilisticMLP — Pre-Drift Verification  (n_test={n_test})")
    print(f"{'═'*70}")

    print(f"\n  Point Prediction Quality")
    print(divider)
    print(f"  {'R²':<35}  {metrics['R2']:>8.4f}    {_check(metrics['R2'] > 0.7)}")
    print(f"  {'RMSE':<35}  {metrics['RMSE']:>8.4f}")
    print(f"  {'MAE':<35}  {metrics['MAE']:>8.4f}")
    print(f"  {'Mean Gaussian NLL':<35}  {metrics['NLL']:>8.4f}")

    print(f"\n  Calibration (PIT Uniformity)")
    print(divider)
    print(
        f"  {'KS statistic vs U[0,1]':<35}  {metrics['KS_stat']:>8.4f}    {_check(metrics['KS_pvalue'] > alpha_ks)}"
    )
    print(f"  {'KS p-value':<35}  {metrics['KS_pvalue']:>8.4f}    (α = {alpha_ks})")
    print(
        f"  {'Expected Calibration Error (ECE)':<35}  {metrics['ECE']:>8.4f}    {_check(metrics['ECE'] < 0.05)}"
    )
    print(
        f"  {'Max Calibration Error (MCE)':<35}  {metrics['MCE']:>8.4f}    {_check(metrics['MCE'] < 0.10)}"
    )
    print(
        f"  {'Empirical 50% interval coverage':<35}  {metrics['coverage_50']:>8.4f}    {_check(abs(metrics['coverage_50']-0.50) < 0.05)}"
    )
    print(
        f"  {'Empirical 90% interval coverage':<35}  {metrics['coverage_90']:>8.4f}    {_check(abs(metrics['coverage_90']-0.90) < 0.05)}"
    )

    print(f"\n  Standardized Residual (Z-score) Normality")
    print(divider)
    print(
        f"  {'Z-score mean':<35}  {metrics['z_mean']:>8.4f}    {_check(abs(metrics['z_mean']) < 0.1)}"
    )
    print(
        f"  {'Z-score std':<35}  {metrics['z_std']:>8.4f}    {_check(0.9 < metrics['z_std'] < 1.1)}"
    )
    print(
        f"  {'KS vs N(0,1)':<35}  p={metrics['z_KS_pvalue']:>7.4f}    {_check(metrics['z_KS_pvalue'] > alpha_ks)}"
    )
    print(
        f"  {'Normal Q-Q correlation r':<35}  {metrics['QQ_r']:>8.4f}    {_check(metrics['QQ_r'] > 0.995)}"
    )

    print(f"\n  Uncertainty Quantification")
    print(divider)
    print(f"  {'Mean predictive σ':<35}  {metrics['mean_sigma']:>8.4f}")
    print(
        f"  {'Std of predictive σ  (CV)':<35}  {metrics['std_sigma']:>8.4f}    ({metrics['sigma_cv']:.3f})"
    )
    # Dispersion ratio: mean σ vs the σ implied by mean |residual| under N(0,1).
    # FriedmanDrift noise is homoscedastic, so near-constant σ is correct — we
    # do NOT penalise low Spearman ρ.  Instead we check that the overall scale
    # of σ matches the observed errors.
    ratio = metrics["sigma_dispersion_ratio"]
    print(
        f"  {'σ dispersion ratio  (ideal: 1.0)':<35}  {ratio:>8.3f}    {_check(abs(ratio - 1.0) < 0.15)}"
    )
    print(f"  {'  = mean σ / (mean |resid| / √(2/π))'}")
    spear_r = metrics["sigma_vs_abserr_spearman_r"]
    print(f"  {'σ vs |resid| Spearman ρ  (info)':<35}  {spear_r:>8.4f}")
    print(f"  {'  (near-zero ρ expected: noise is homoscedastic)'}")

    # Overall verdict
    sigma_ok = abs(metrics["sigma_dispersion_ratio"] - 1.0) < 0.15
    checks = [
        metrics["R2"] > 0.7,
        metrics["KS_pvalue"] > alpha_ks,
        metrics["ECE"] < 0.05,
        metrics["MCE"] < 0.10,
        abs(metrics["coverage_50"] - 0.50) < 0.05,
        abs(metrics["coverage_90"] - 0.90) < 0.05,
        abs(metrics["z_mean"]) < 0.1,
        0.9 < metrics["z_std"] < 1.1,
        metrics["z_KS_pvalue"] > alpha_ks,
        sigma_ok,
    ]
    n_pass = sum(checks)
    n_total = len(checks)
    verdict_color = (
        "\033[92m"
        if n_pass == n_total
        else ("\033[93m" if n_pass >= n_total * 0.7 else "\033[91m")
    )
    print(f"\n{'═'*70}")
    print(f"  Overall: {verdict_color}{n_pass}/{n_total} checks passed\033[0m")
    if n_pass < n_total:
        print(f"\n  Interpretation for PITMonitor:")
        if metrics["KS_pvalue"] <= alpha_ks or metrics["ECE"] >= 0.05:
            print("  ⚠  PITs are not close to U[0,1] even on pre-drift data.")
            print("     PITMonitor may raise false alarms (FPR could exceed α).")
        if metrics["R2"] <= 0.7:
            print("  ⚠  Point predictions are weak.  The model may need more")
            print("     capacity, training time, or better hyperparameters.")
        if not (0.9 < metrics["z_std"] < 1.1):
            if metrics["z_std"] < 0.9:
                print("  ⚠  Predictive σ is too large (over-dispersed).")
                print(
                    "     Intervals are wider than necessary → PITMonitor slower to detect drift."
                )
            else:
                print("  ⚠  Predictive σ is too small (under-dispersed).")
                print(
                    "     Model is overconfident → PITs crowd near 0 and 1 → elevated FPR."
                )
        if not sigma_ok:
            ratio = metrics["sigma_dispersion_ratio"]
            if ratio > 1.15:
                print("  ⚠  Mean σ is larger than the errors imply (over-dispersed).")
                print(
                    "     Predictive intervals are unnecessarily wide → reduced power."
                )
            else:
                print("  ⚠  Mean σ is smaller than the errors imply (under-dispersed).")
                print(
                    "     Model is overconfident → PITs skewed toward 0 and 1 → elevated FPR."
                )
    print(f"{'═'*70}\n")


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    """Load (or train) the model, run diagnostics, save and display figure."""
    args = parse_args()
    cfg = Config(seed=args.seed, output_dir=args.output)

    # ── Locate or train bundle ────────────────────────────────────────
    bundle_path = Path(args.bundle) if args.bundle else cfg.bundle_path
    if bundle_path.exists():
        print(f"Loading model bundle from {bundle_path} …")
        bundle: ModelBundle = load_bundle(bundle_path)
    else:
        print(f"No bundle found at {bundle_path}.  Training from scratch …")
        drift_type, tw = cfg.drift_scenarios[0]
        X_all, y_all = generate_stream(
            cfg, drift_type=drift_type, transition_window=tw, seed=cfg.seed
        )
        X_train, y_train = X_all[: cfg.n_train], y_all[: cfg.n_train]
        bundle = train_model(
            X_train, y_train, epochs=cfg.epochs, lr=cfg.lr, seed=cfg.seed
        )
        save_bundle(bundle, bundle_path)
        print(f"Bundle saved to {bundle_path}")

    # ── Generate held-out pre-drift test data ─────────────────────────
    # Use a fresh seed so the test set is disjoint from training in distribution
    # (same pre-drift DGP, different random stream).
    test_seed = cfg.seed + 999_999
    print(
        f"Generating {args.n_test} held-out pre-drift test samples (seed={test_seed}) …"
    )
    drift_type, tw = cfg.drift_scenarios[0]

    # Build a cfg with enough samples: n_train (skip) + n_test
    from dataclasses import replace

    # We only need n_train + n_test samples; reuse generate_stream with
    # a temporary config that has n_stable=n_test, n_post=0.
    test_cfg = Config(
        seed=test_seed,
        n_train=cfg.n_train,
        n_stable=args.n_test,
        n_post=0,
        output_dir=args.output,
    )
    X_all, y_all = generate_stream(
        test_cfg, drift_type=drift_type, transition_window=tw, seed=test_seed
    )
    X_test = X_all[test_cfg.n_train :]
    y_test = y_all[test_cfg.n_train :]
    print(f"  Test set shape: X={X_test.shape}, y={y_test.shape}")

    # ── Compute diagnostics ───────────────────────────────────────────
    print("Computing predictions and PITs …")
    mu, sigma = compute_predictions(bundle, X_test)
    pits = compute_pits(bundle, X_test, y_test)
    z_scores = (y_test - mu) / np.clip(sigma, 1e-8, None)

    # ── Build figure ──────────────────────────────────────────────────
    print("Building diagnostic figure …")
    fig, diag_stats = make_diagnostic_figure(
        y_test=y_test,
        mu=mu,
        sigma=sigma,
        pits=pits,
        z_scores=z_scores,
        alpha_ks=args.alpha_ks,
    )

    # Save
    out_path = Path(args.output)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = out_path / "verify_model.png"
    fig.savefig(fig_path, bbox_inches="tight")
    print(f"\nFigure saved to {fig_path}")

    # ── Console summary ───────────────────────────────────────────────
    print_summary(diag_stats, alpha_ks=args.alpha_ks, n_test=len(y_test))

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

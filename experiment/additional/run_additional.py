"""Targeted additional experiments for PITMonitor theory and assumptions.

This module adds lightweight, reproducible checks for:
1. Detection-power propositions in Section 3 (asymptotic limit, finite-time mean,
   and warmup bound under adverse prior alignment).
2. Stationary but dependent PIT streams.
3. Mildly non-stationary PIT streams.
4. Seasonal PIT streams.

Usage
-----
python run_additional.py
python run_additional.py --profile quick
python run_additional.py --profile publication
python run_additional.py --output out/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _THIS_DIR.parent
_REPO_ROOT = _EXPERIMENT_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

from pitmon import PITMonitor
from export_additional_macros import build_macros


@dataclass(frozen=True)
class VerificationConfig:
    """Configuration for the verification experiments."""

    seed: int = 123
    alpha: float = 0.05
    n_bins: int = 100

    # Proposition verification parameters
    m_pre: int = 2_500
    n_grid_max: int = 5_000
    n_grid_points: int = 18
    n_mc_formula: int = 12_000

    # Stream robustness parameters
    n_trials_stream: int = 2_000
    n_steps_stream: int = 5_000

    # Multi-step drift localization parameters
    n_trials_localization: int = 2_000
    n_steps_localization: int = 5_000
    onset_step: int = 2_001
    mid_step: int = 2_901
    max_step: int = 3_701


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a Bernoulli proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def make_theta_nonuniform(n_bins: int, sharpness: float = 1.6) -> np.ndarray:
    """Create a smooth non-uniform bin distribution over [0, 1]."""
    x = np.linspace(0.0, 1.0, n_bins, endpoint=False) + 0.5 / n_bins
    # Symmetric, mildly U-shaped profile to avoid overfitting to one edge.
    raw = 1.0 + sharpness * np.abs(x - 0.5)
    theta = raw / raw.sum()
    return theta.astype(np.float64)


def formula_expected_e(
    n: int,
    n_bins: int,
    m_pre: int,
    gamma: float,
    delta: float,
) -> float:
    """Finite-time formula from Proposition (Finite-Sample Mean Gain)."""
    denom = n_bins + m_pre + n
    return 1.0 + ((n_bins + m_pre) / denom) * gamma + (n / denom) * (n_bins * delta)


def empirical_expected_e(
    rng: np.random.Generator,
    theta: np.ndarray,
    A: np.ndarray,
    n_bins: int,
    m_pre: int,
    n: int,
    n_mc: int,
) -> dict:
    """Monte-Carlo estimate of E[e_t | A] at post-shift lag n."""
    probs = theta
    denom = n_bins + m_pre + n

    # Draw C in bulk then draw the next bin b_t.
    C = rng.multinomial(n=n, pvals=probs, size=n_mc)
    b_t = rng.choice(n_bins, size=n_mc, p=probs)
    numer = 1.0 + A[b_t] + C[np.arange(n_mc), b_t]
    e_vals = (n_bins * numer) / denom

    return {
        "mean": float(np.mean(e_vals)),
        "std": float(np.std(e_vals, ddof=1)),
        "se": float(np.std(e_vals, ddof=1) / math.sqrt(n_mc)),
    }


def build_pre_counts(theta: np.ndarray, m_pre: int, mode: str) -> np.ndarray:
    """Construct pre-shift bin counts A for aligned/misaligned cases."""
    n_bins = len(theta)
    if mode == "aligned":
        p = theta
    elif mode == "misaligned":
        # Intentionally anti-aligned but not degenerate (keeps MC variance sane).
        p = 1.0 / np.maximum(theta, 1e-12)
        p = p / p.sum()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rng_local = np.random.default_rng(2024 if mode == "aligned" else 2025)
    A = rng_local.multinomial(n=m_pre, pvals=p)
    return A.astype(np.int64)


def verify_detection_power_propositions(cfg: VerificationConfig) -> dict:
    """Verify asymptotic limit, finite-time mean formula, and warmup bound."""
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_bins
    m = cfg.m_pre
    theta = make_theta_nonuniform(B)

    delta = float(np.sum(theta**2) - 1.0 / B)
    asymptotic_target = float(B * np.sum(theta**2))

    cases = {}
    for mode in ("aligned", "misaligned"):
        A = build_pre_counts(theta, m, mode)
        q_hat = (1.0 + A) / (B + m)
        gamma = float(B * np.sum(theta * q_hat) - 1.0)

        n_grid = np.unique(
            np.round(np.geomspace(1, cfg.n_grid_max, num=cfg.n_grid_points)).astype(int)
        )

        rows = []
        for n in n_grid:
            formula_mean = formula_expected_e(
                n=n,
                n_bins=B,
                m_pre=m,
                gamma=gamma,
                delta=delta,
            )
            emp = empirical_expected_e(
                rng=rng,
                theta=theta,
                A=A,
                n_bins=B,
                m_pre=m,
                n=n,
                n_mc=cfg.n_mc_formula,
            )
            rows.append(
                {
                    "n_post_before_t": int(n),
                    "formula_mean_e": float(formula_mean),
                    "empirical_mean_e": emp["mean"],
                    "empirical_se": emp["se"],
                    "abs_error": float(abs(emp["mean"] - formula_mean)),
                }
            )

        max_abs_error = max(r["abs_error"] for r in rows)
        tail = [r for r in rows if r["n_post_before_t"] >= int(0.7 * cfg.n_grid_max)]
        tail_emp_mean = float(np.mean([r["empirical_mean_e"] for r in tail]))

        # Warmup threshold n*: formula predicts when E[e_t | A] crosses 1.
        if gamma < 0 and delta > 0:
            n_star = ((B + m) * abs(gamma)) / (B * delta)
            warmup_bound = m / (B * delta)
        else:
            n_star = 0.0
            warmup_bound = m / (B * delta) if delta > 0 else float("inf")

        # Empirical crossing index (first n where lower 95% MC CI is > 1).
        n_emp_cross = None
        for r in rows:
            if (r["empirical_mean_e"] - 1.96 * r["empirical_se"]) > 1.0:
                n_emp_cross = r["n_post_before_t"]
                break

        cases[mode] = {
            "gamma": gamma,
            "delta": delta,
            "asymptotic_target_Bsumtheta2": asymptotic_target,
            "tail_empirical_mean_e": tail_emp_mean,
            "max_abs_formula_error": float(max_abs_error),
            "n_star_formula": float(n_star),
            "warmup_bound_m_over_Bdelta": float(warmup_bound),
            "empirical_crossing_n": (
                int(n_emp_cross) if n_emp_cross is not None else None
            ),
            "rows": rows,
        }

    # Check warmup bound n* <= m/(B*delta) under many prior alignments.
    n_random = 300
    warmup_ratios = []
    for _ in range(n_random):
        A = rng.multinomial(n=m, pvals=rng.dirichlet(np.ones(B)))
        q_hat = (1.0 + A) / (B + m)
        gamma = float(B * np.sum(theta * q_hat) - 1.0)
        if gamma >= 0 or delta <= 0:
            continue
        n_star = ((B + m) * abs(gamma)) / (B * delta)
        bound = m / (B * delta)
        warmup_ratios.append(n_star / bound)

    # Include crafted, strongly misaligned priors to probe near worst-case ratios.
    p_concentrated = np.zeros(B, dtype=np.float64)
    p_concentrated[int(np.argmin(theta))] = 1.0
    A_worst = rng.multinomial(n=m, pvals=p_concentrated)
    q_worst = (1.0 + A_worst) / (B + m)
    gamma_worst = float(B * np.sum(theta * q_worst) - 1.0)
    n_star_worst = ((B + m) * abs(gamma_worst)) / (B * delta)
    bound = m / (B * delta)

    p_inverse = 1.0 / np.maximum(theta, 1e-12)
    p_inverse = p_inverse / p_inverse.sum()
    A_inverse = rng.multinomial(n=m, pvals=p_inverse)
    q_inverse = (1.0 + A_inverse) / (B + m)
    gamma_inverse = float(B * np.sum(theta * q_inverse) - 1.0)
    n_star_inverse = ((B + m) * abs(gamma_inverse)) / (B * delta)

    return {
        "settings": {
            "n_bins": B,
            "m_pre": m,
            "n_grid_max": cfg.n_grid_max,
            "n_grid_points": cfg.n_grid_points,
            "n_mc_formula": cfg.n_mc_formula,
        },
        "cases": cases,
        "warmup_ratio_random_priors": {
            "n_valid": len(warmup_ratios),
            "max_ratio": (
                float(np.max(warmup_ratios)) if warmup_ratios else float("nan")
            ),
            "mean_ratio": (
                float(np.mean(warmup_ratios)) if warmup_ratios else float("nan")
            ),
        },
        "warmup_ratio_constructed_priors": {
            "inverse_theta_ratio": float(n_star_inverse / bound),
            "concentrated_min_theta_ratio": float(n_star_worst / bound),
        },
    }


def normal_cdf_scalar(x: float) -> float:
    """Standard normal CDF via erf, scalar and dependency-free."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def stream_iid_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=n)


def stream_stationary_dependent_ar1(
    rng: np.random.Generator,
    n: int,
    phi: float = 0.7,
) -> np.ndarray:
    """Stationary dependent stream with uniform margins via Gaussian copula."""
    z = np.empty(n, dtype=np.float64)
    z[0] = rng.normal()
    sigma_eps = math.sqrt(max(1e-12, 1.0 - phi * phi))
    for t in range(1, n):
        z[t] = phi * z[t - 1] + sigma_eps * rng.normal()
    u = np.array([normal_cdf_scalar(v) for v in z], dtype=np.float64)
    return np.clip(u, 1e-12, 1.0 - 1e-12)


def stream_mild_nonstationary(
    rng: np.random.Generator,
    n: int,
    k_min: float = 0.9,
    k_max: float = 1.25,
) -> np.ndarray:
    """Mildly non-stationary stream with slowly drifting symmetric Beta shape."""
    t = np.linspace(0.0, 1.0, n)
    k = k_min + (k_max - k_min) * t
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rng.beta(k[i], k[i])
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def stream_seasonal(
    rng: np.random.Generator,
    n: int,
    period: int = 300,
    amp: float = 0.55,
) -> np.ndarray:
    """Seasonal stream via periodic Beta concentration around uniform."""
    idx = np.arange(n)
    k = 1.0 + amp * np.sin(2.0 * math.pi * idx / period)
    k = np.clip(k, 0.35, 1.65)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rng.beta(k[i], k[i])
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def stream_multistep_piecewise(
    rng: np.random.Generator,
    n: int,
    onset_step: int,
    mid_step: int,
    max_step: int,
    k_levels: tuple[float, float, float, float] = (1.0, 0.85, 0.65, 0.45),
) -> np.ndarray:
    """Piecewise-constant PIT stream with several step changes in drift intensity.

    A smaller symmetric-Beta shape parameter implies stronger non-uniformity and
    therefore stronger exchangeability violation intensity for PITMonitor.
    """
    idx = np.arange(1, n + 1)
    k = np.full(n, k_levels[0], dtype=np.float64)
    k[idx >= onset_step] = k_levels[1]
    k[idx >= mid_step] = k_levels[2]
    k[idx >= max_step] = k_levels[3]

    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = rng.beta(k[i], k[i])
    return np.clip(out, 1e-12, 1.0 - 1e-12)


def lag1_autocorr(x: np.ndarray) -> float:
    """Sample lag-1 autocorrelation."""
    x0 = x[:-1]
    x1 = x[1:]
    x0c = x0 - x0.mean()
    x1c = x1 - x1.mean()
    denom = np.sqrt(np.sum(x0c**2) * np.sum(x1c**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(x0c * x1c) / denom)


def uniform_ks_statistic(x: np.ndarray) -> float:
    """One-sample KS statistic against Uniform(0,1), no scipy dependency."""
    xs = np.sort(x)
    n = len(xs)
    i = np.arange(1, n + 1)
    d_plus = np.max(i / n - xs)
    d_minus = np.max(xs - (i - 1) / n)
    return float(max(d_plus, d_minus))


def evaluate_stream_regime(
    cfg: VerificationConfig,
    name: str,
    generator: Callable[[np.random.Generator, int], np.ndarray],
) -> dict:
    """Estimate alarm behaviour of PITMonitor under one stream regime."""
    # Stable across Python sessions (unlike hash(name)).
    name_offset = int(sum(name.encode("utf-8")))
    rng = np.random.default_rng(cfg.seed + name_offset)
    alarm_times: list[int] = []

    # Diagnostic stream for dependence and marginal shape checks.
    diag = generator(rng, cfg.n_steps_stream)
    diagnostics = {
        "mean": float(np.mean(diag)),
        "var": float(np.var(diag)),
        "lag1_autocorr": lag1_autocorr(diag),
        "ks_uniform": uniform_ks_statistic(diag),
    }

    for _ in range(cfg.n_trials_stream):
        pits = generator(rng, cfg.n_steps_stream)
        mon = PITMonitor(alpha=cfg.alpha, n_bins=cfg.n_bins, rng=rng.integers(1 << 31))

        tripped = False
        for t, pit in enumerate(pits, start=1):
            alarm = mon.update(float(pit))
            if alarm.triggered:
                alarm_times.append(t)
                tripped = True
                break

        if not tripped:
            alarm_times.append(-1)

    n = cfg.n_trials_stream
    n_alarm = int(sum(t > 0 for t in alarm_times))
    alarm_rate = n_alarm / n
    ci_lo, ci_hi = wilson_ci(n_alarm, n)

    positive_times = [t for t in alarm_times if t > 0]
    median_time = float(np.median(positive_times)) if positive_times else float("nan")

    return {
        "name": name,
        "settings": {
            "n_trials": cfg.n_trials_stream,
            "n_steps": cfg.n_steps_stream,
            "alpha": cfg.alpha,
            "n_bins": cfg.n_bins,
        },
        "diagnostics": diagnostics,
        "alarm_rate": alarm_rate,
        "alarm_rate_ci95": [ci_lo, ci_hi],
        "n_alarms": n_alarm,
        "median_alarm_time_conditional": median_time,
    }


def evaluate_multistep_localization(cfg: VerificationConfig) -> dict:
    """Report changepoint error to onset and maximal-intensity phase.

    The stream has three step changes in intensity at known times:
    onset_step < mid_step < max_step. We report localization errors of
    PITMonitor's changepoint estimate against both onset and max_step.
    """
    name_offset = int(sum(b"multistep_localization"))
    rng = np.random.default_rng(cfg.seed + name_offset)

    onset = cfg.onset_step
    max_step = cfg.max_step

    cp_estimates: list[int] = []
    alarm_times: list[int] = []

    for _ in range(cfg.n_trials_localization):
        pits = stream_multistep_piecewise(
            rng=rng,
            n=cfg.n_steps_localization,
            onset_step=cfg.onset_step,
            mid_step=cfg.mid_step,
            max_step=cfg.max_step,
        )
        mon = PITMonitor(alpha=cfg.alpha, n_bins=cfg.n_bins, rng=rng.integers(1 << 31))

        alarm_time = -1
        for t, pit in enumerate(pits, start=1):
            alarm = mon.update(float(pit))
            if alarm.triggered:
                alarm_time = t
                break

        if alarm_time > 0:
            cp = mon.changepoint()
            if cp is not None:
                cp_estimates.append(int(cp))

        alarm_times.append(alarm_time)

    n = cfg.n_trials_localization
    n_alarm = int(sum(t > 0 for t in alarm_times))
    alarm_rate = n_alarm / n
    ci_lo, ci_hi = wilson_ci(n_alarm, n)

    signed_onset = [cp - onset for cp in cp_estimates]
    signed_max = [cp - max_step for cp in cp_estimates]

    abs_onset = [abs(v) for v in signed_onset]
    abs_max = [abs(v) for v in signed_max]

    n_cp = len(cp_estimates)
    n_closer_onset = int(sum(abs_onset[i] < abs_max[i] for i in range(n_cp)))
    n_closer_max = int(sum(abs_max[i] < abs_onset[i] for i in range(n_cp)))
    n_ties = n_cp - n_closer_onset - n_closer_max

    return {
        "name": "multistep_step_drift_localization",
        "settings": {
            "n_trials": cfg.n_trials_localization,
            "n_steps": cfg.n_steps_localization,
            "alpha": cfg.alpha,
            "n_bins": cfg.n_bins,
            "onset_step": cfg.onset_step,
            "mid_step": cfg.mid_step,
            "max_step": cfg.max_step,
            "k_levels": [1.0, 0.85, 0.65, 0.45],
        },
        "alarm_rate": alarm_rate,
        "alarm_rate_ci95": [ci_lo, ci_hi],
        "n_alarms": n_alarm,
        "n_with_changepoint_estimate": n_cp,
        "mae_to_onset": float(np.mean(abs_onset)) if abs_onset else float("nan"),
        "mae_to_max_phase": float(np.mean(abs_max)) if abs_max else float("nan"),
        "median_signed_error_to_onset": (
            float(np.median(signed_onset)) if signed_onset else float("nan")
        ),
        "median_signed_error_to_max_phase": (
            float(np.median(signed_max)) if signed_max else float("nan")
        ),
        "closer_reference_counts": {
            "onset": n_closer_onset,
            "max_phase": n_closer_max,
            "tie": n_ties,
        },
        "signed_error_to_onset": signed_onset,
        "signed_error_to_max_phase": signed_max,
    }


def run_all(cfg: VerificationConfig) -> dict:
    """Run all requested verification experiments."""
    proposition = verify_detection_power_propositions(cfg)

    stream_results = [
        evaluate_stream_regime(cfg, "iid_uniform_baseline", stream_iid_uniform),
        evaluate_stream_regime(
            cfg,
            "stationary_dependent_ar1_uniform_margin",
            lambda r, n: stream_stationary_dependent_ar1(r, n, phi=0.7),
        ),
        evaluate_stream_regime(
            cfg,
            "mild_nonstationary_symmetric_beta_trend",
            lambda r, n: stream_mild_nonstationary(r, n, k_min=0.9, k_max=1.25),
        ),
        evaluate_stream_regime(
            cfg,
            "seasonal_symmetric_beta",
            lambda r, n: stream_seasonal(r, n, period=300, amp=0.55),
        ),
    ]

    localization = evaluate_multistep_localization(cfg)

    # Compact takeaways to make paper integration easy.
    baseline = next(r for r in stream_results if r["name"] == "iid_uniform_baseline")
    dep = next(
        r
        for r in stream_results
        if r["name"] == "stationary_dependent_ar1_uniform_margin"
    )
    nonstat = next(
        r
        for r in stream_results
        if r["name"] == "mild_nonstationary_symmetric_beta_trend"
    )
    seasonal = next(r for r in stream_results if r["name"] == "seasonal_symmetric_beta")

    summary = {
        "proposition_formula_accuracy": {
            mode: proposition["cases"][mode]["max_abs_formula_error"]
            for mode in ("aligned", "misaligned")
        },
        "asymptotic_empirical_minus_target": {
            mode: proposition["cases"][mode]["tail_empirical_mean_e"]
            - proposition["cases"][mode]["asymptotic_target_Bsumtheta2"]
            for mode in ("aligned", "misaligned")
        },
        "warmup_bound_ratio_max": proposition["warmup_ratio_random_priors"][
            "max_ratio"
        ],
        "alarm_rate_by_regime": {
            "iid_uniform": baseline["alarm_rate"],
            "stationary_dependent": dep["alarm_rate"],
            "mild_nonstationary": nonstat["alarm_rate"],
            "seasonal": seasonal["alarm_rate"],
        },
        "multistep_localization": {
            "mae_to_onset": localization["mae_to_onset"],
            "mae_to_max_phase": localization["mae_to_max_phase"],
            "alarm_rate": localization["alarm_rate"],
        },
    }

    return {
        "config": {
            "seed": cfg.seed,
            "alpha": cfg.alpha,
            "n_bins": cfg.n_bins,
            "m_pre": cfg.m_pre,
            "n_grid_max": cfg.n_grid_max,
            "n_grid_points": cfg.n_grid_points,
            "n_mc_formula": cfg.n_mc_formula,
            "n_trials_stream": cfg.n_trials_stream,
            "n_steps_stream": cfg.n_steps_stream,
            "n_trials_localization": cfg.n_trials_localization,
            "n_steps_localization": cfg.n_steps_localization,
            "onset_step": cfg.onset_step,
            "mid_step": cfg.mid_step,
            "max_step": cfg.max_step,
        },
        "proposition_verification": proposition,
        "stream_regime_verification": stream_results,
        "multistep_localization_verification": localization,
        "summary": summary,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PITMonitor additional experiments")
    p.add_argument("--output", type=str, default="out/results.json")
    p.add_argument(
        "--profile",
        type=str,
        choices=["quick", "standard", "publication"],
        default="standard",
        help=(
            "Experiment preset: quick (smoke), standard (default research), "
            "publication (higher precision)."
        ),
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--bins", type=int, default=100)
    p.add_argument(
        "--macros-output",
        type=str,
        default="out/additional_macros.tex",
        help="Where to write auto-generated LaTeX macros for appendix tables.",
    )
    p.add_argument(
        "--no-export-macros",
        action="store_true",
        help="Skip LaTeX macro export after writing JSON results.",
    )
    return p.parse_args()


def config_for_profile(args: argparse.Namespace) -> VerificationConfig:
    """Construct a config from the selected profile plus CLI overrides."""
    base = VerificationConfig(seed=args.seed, alpha=args.alpha, n_bins=args.bins)

    if args.profile == "quick":
        return VerificationConfig(
            seed=args.seed,
            alpha=args.alpha,
            n_bins=args.bins,
            m_pre=1_500,
            n_grid_max=2_500,
            n_grid_points=12,
            n_mc_formula=2_500,
            n_trials_stream=300,
            n_steps_stream=2_000,
            n_trials_localization=300,
            n_steps_localization=3_500,
            onset_step=1_201,
            mid_step=2_001,
            max_step=2_801,
        )

    if args.profile == "publication":
        return VerificationConfig(
            seed=args.seed,
            alpha=args.alpha,
            n_bins=args.bins,
            m_pre=3_000,
            n_grid_max=7_500,
            n_grid_points=22,
            n_mc_formula=20_000,
            n_trials_stream=5_000,
            n_steps_stream=5_000,
            n_trials_localization=5_000,
            n_steps_localization=5_000,
            onset_step=2_001,
            mid_step=2_901,
            max_step=3_701,
        )

    return base


def main() -> None:
    args = parse_args()
    cfg = config_for_profile(args)

    results = run_all(cfg)
    results["config"]["profile"] = args.profile

    out_path = (
        (_THIS_DIR / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if not args.no_export_macros:
        macros_path = (
            (_THIS_DIR / args.macros_output).resolve()
            if not Path(args.macros_output).is_absolute()
            else Path(args.macros_output)
        )
        macros_path.parent.mkdir(parents=True, exist_ok=True)
        macros_path.write_text(build_macros(results), encoding="utf-8")
        print(f"Wrote LaTeX macros to {macros_path}")

    print(f"Saved additional experiment results to {out_path}")
    print("Summary:")
    for k, v in results["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

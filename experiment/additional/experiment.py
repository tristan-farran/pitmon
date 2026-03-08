"""Verification computations for PITMonitor additional experiments."""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from config import VerificationConfig
from data import build_pre_counts, make_theta_nonuniform, stream_multistep_piecewise
from pitmon import PITMonitor


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a Bernoulli proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = k / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z * z) / (4.0 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


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


# ---------------------------------------------------------------------------
# Top-level worker functions (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _proposition_grid_worker(
    seed: int,
    n: int,
    theta: np.ndarray,
    A: np.ndarray,
    n_bins: int,
    m_pre: int,
    gamma: float,
    delta: float,
    n_mc: int,
) -> dict:
    """Single grid-point MC worker for proposition verification."""
    rng = np.random.default_rng(seed)
    formula_mean = formula_expected_e(
        n=n, n_bins=n_bins, m_pre=m_pre, gamma=gamma, delta=delta
    )
    emp = empirical_expected_e(
        rng=rng, theta=theta, A=A, n_bins=n_bins, m_pre=m_pre, n=n, n_mc=n_mc
    )
    return {
        "n_post_before_t": int(n),
        "formula_mean_e": float(formula_mean),
        "empirical_mean_e": emp["mean"],
        "empirical_se": emp["se"],
        "abs_error": float(abs(emp["mean"] - formula_mean)),
    }


def _localization_trial(
    seed: int,
    onset_step: int,
    mid_step: int,
    max_step: int,
    n_steps: int,
    alpha: float,
    n_bins: int,
) -> tuple[int, int | None]:
    """Single localization trial; returns (alarm_time, changepoint_or_None)."""
    rng = np.random.default_rng(seed)
    pits = stream_multistep_piecewise(
        rng=rng,
        n=n_steps,
        onset_step=onset_step,
        mid_step=mid_step,
        max_step=max_step,
    )
    mon = PITMonitor(alpha=alpha, n_bins=n_bins, rng=rng.integers(1 << 31))
    alarm_time = -1
    for t, pit in enumerate(pits, start=1):
        alarm = mon.update(float(pit))
        if alarm.triggered:
            alarm_time = t
            break
    cp: int | None = None
    if alarm_time > 0:
        raw = mon.changepoint()
        if raw is not None:
            cp = int(raw)
    return alarm_time, cp


# ---------------------------------------------------------------------------
# Experiment functions
# ---------------------------------------------------------------------------


def verify_detection_power_propositions(cfg: VerificationConfig) -> dict:
    """Verify asymptotic limit, finite-time mean formula, and warmup bound."""
    rng = np.random.default_rng(cfg.seed)
    B = cfg.n_bins
    m = cfg.m_pre
    theta = make_theta_nonuniform(B)

    delta = float(np.sum(theta**2) - 1.0 / B)
    asymptotic_target = float(B * np.sum(theta**2))

    # Pre-compute per-mode data so we can submit all grid jobs in one pool.
    mode_data: dict[str, dict] = {}
    for mode in ("aligned", "misaligned"):
        A = build_pre_counts(theta, m, mode)
        q_hat = (1.0 + A) / (B + m)
        gamma = float(B * np.sum(theta * q_hat) - 1.0)
        n_grid = np.unique(
            np.round(np.geomspace(1, cfg.n_grid_max, num=cfg.n_grid_points)).astype(int)
        )
        mode_data[mode] = {"A": A, "gamma": gamma, "n_grid": n_grid}

    # --- Parallel grid computation ---
    seed_rng = np.random.default_rng(cfg.seed + 999)
    total_jobs = sum(len(md["n_grid"]) for md in mode_data.values())
    print(
        f"[proposition] submitting {total_jobs} grid-point jobs "
        f"({cfg.n_mc_formula:,} MC samples each, 2 modes) ..."
    )

    all_rows: dict[str, dict[int, dict]] = {"aligned": {}, "misaligned": {}}
    completed = 0

    with ProcessPoolExecutor() as executor:
        future_to_mode: dict = {}
        for mode, md in mode_data.items():
            for n_val in md["n_grid"]:
                seed = int(seed_rng.integers(0, 1 << 31))
                fut = executor.submit(
                    _proposition_grid_worker,
                    seed=seed,
                    n=int(n_val),
                    theta=theta,
                    A=md["A"],
                    n_bins=B,
                    m_pre=m,
                    gamma=md["gamma"],
                    delta=delta,
                    n_mc=cfg.n_mc_formula,
                )
                future_to_mode[fut] = mode

        for future in as_completed(future_to_mode):
            mode = future_to_mode[future]
            row = future.result()
            all_rows[mode][row["n_post_before_t"]] = row
            completed += 1
            print(
                f"[proposition] {completed}/{total_jobs}  [{mode:>10}]  "
                f"n={row['n_post_before_t']:>6,}  "
                f"formula={row['formula_mean_e']:.4f}  "
                f"empirical={row['empirical_mean_e']:.4f}  "
                f"err={row['abs_error']:.2e}"
            )

    print(f"[proposition] all {total_jobs} grid jobs complete")

    # --- Derive per-mode statistics ---
    cases = {}
    for mode, md in mode_data.items():
        rows = [all_rows[mode][int(n_val)] for n_val in sorted(all_rows[mode])]
        gamma = md["gamma"]

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

    # --- Warmup bound check (serial: 10k cheap numpy iterations) ---
    n_random = 10_000
    print(f"[proposition] warmup bound check: {n_random:,} random priors ...")
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
    print(
        f"[proposition] warmup bound check done "
        f"({len(warmup_ratios):,} valid cases, "
        f"max ratio={max(warmup_ratios, default=float('nan')):.4f})"
    )

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


def evaluate_multistep_localization(cfg: VerificationConfig) -> dict:
    """Report changepoint error to onset and maximal-intensity phase.

    The stream has three step changes in intensity at known times:
    onset_step < mid_step < max_step. We report localization errors of
    PITMonitor's changepoint estimate against both onset and max_step.
    """
    name_offset = int(sum(b"multistep_localization"))
    ss = np.random.SeedSequence(cfg.seed + name_offset)
    child_seeds = [
        int(s.generate_state(1)[0]) for s in ss.spawn(cfg.n_trials_localization)
    ]

    n_trials = cfg.n_trials_localization
    onset = cfg.onset_step
    max_step = cfg.max_step

    print(
        f"[localization] submitting {n_trials:,} trials "
        f"({cfg.n_steps_localization:,} steps each) ..."
    )
    report_every = max(1, n_trials // 10)

    cp_estimates: list[int] = []
    alarm_times: list[int] = []
    completed = 0

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                _localization_trial,
                seed,
                cfg.onset_step,
                cfg.mid_step,
                cfg.max_step,
                cfg.n_steps_localization,
                cfg.alpha,
                cfg.n_bins,
            ): i
            for i, seed in enumerate(child_seeds)
        }
        for future in as_completed(futures):
            alarm_time, cp = future.result()
            alarm_times.append(alarm_time)
            if alarm_time > 0 and cp is not None:
                cp_estimates.append(cp)
            completed += 1
            if completed % report_every == 0 or completed == n_trials:
                n_alarm_so_far = sum(t > 0 for t in alarm_times)
                print(
                    f"[localization] {completed:>{len(str(n_trials))}}/{n_trials:,}  "
                    f"({100 * completed // n_trials:3d}%)  "
                    f"alarms so far: {n_alarm_so_far:,}"
                )

    print(f"[localization] all {n_trials:,} trials complete")

    n_alarm = int(sum(t > 0 for t in alarm_times))
    alarm_rate = n_alarm / n_trials
    ci_lo, ci_hi = wilson_ci(n_alarm, n_trials)

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
    print("=" * 60)
    print("Stage 1/2  proposition verification")
    print("=" * 60)
    proposition = verify_detection_power_propositions(cfg)
    print()

    print("=" * 60)
    print("Stage 2/2  multistep localization")
    print("=" * 60)
    localization = evaluate_multistep_localization(cfg)
    print()

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
            "n_trials_localization": cfg.n_trials_localization,
            "n_steps_localization": cfg.n_steps_localization,
            "onset_step": cfg.onset_step,
            "mid_step": cfg.mid_step,
            "max_step": cfg.max_step,
        },
        "proposition_verification": proposition,
        "multistep_localization_verification": localization,
        "summary": summary,
    }

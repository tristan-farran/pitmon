"""Core experiment: run all detectors across trials and scenarios.

Each trial:
    1. Generate a FriedmanDrift stream  (data.py)
    2. Train MLP on pre-drift data      (model.py)
    3. Compute PITs + residuals on the monitoring window
    4. Feed every detector               (detectors.py)
    5. Collect results
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import Config
from data import generate_stream
from detectors import (
    ALL_DETECTOR_NAMES,
    DetectorResult,
    build_all_detectors,
)
from model import compute_pits, compute_residuals, train_model


# ─── Single trial ────────────────────────────────────────────────────


def run_single_trial(
    cfg: Config,
    drift_type: str,
    transition_window: int,
    trial_idx: int,
) -> dict[str, dict]:
    """Run one trial and return {detector_name: result_dict}."""
    seed = cfg.seed + trial_idx * 1000

    # 1. Generate data
    X, y = generate_stream(cfg, drift_type, transition_window, seed=seed)

    X_train, y_train = X[: cfg.n_train], y[: cfg.n_train]
    X_mon = X[cfg.n_train :]
    y_mon = y[cfg.n_train :]

    # 2. Train model
    gbr, sigma_hat = train_model(X_train, y_train, cfg.n_cal_frac, seed=seed)

    # 3. Compute monitoring signals
    pits = compute_pits(gbr, sigma_hat, X_mon, y_mon)
    residuals = compute_residuals(gbr, X_mon, y_mon)

    sq_residuals = residuals**2

    # Binary errors: 1 if |residual| exceeds pre-drift median absolute error.
    pre_drift_abs = np.abs(residuals[: cfg.n_stable])
    threshold = float(np.median(pre_drift_abs))
    binary_errors = (np.abs(residuals) > threshold).astype(np.float64)

    # 4. Run all detectors
    detectors = build_all_detectors(
        alpha=cfg.alpha,
        n_monitor_bins=cfg.n_monitor_bins,
        seed=seed,
    )
    out: dict[str, dict] = {}
    for det in detectors:
        det.feed(pits, sq_residuals, binary_errors, cfg.n_stable)
        out[det.name] = asdict(det.result)

    return out


# ─── Aggregation helpers ─────────────────────────────────────────────


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def aggregate_results(
    trial_results: list[dict[str, dict]],
    n_stable: int,
) -> dict[str, dict]:
    """Aggregate per-trial detector results into summary statistics.

    Returns {detector_name: summary_dict}.
    """
    summaries: dict[str, dict] = {}

    for det_name in ALL_DETECTOR_NAMES:
        rows = [trial[det_name] for trial in trial_results if det_name in trial]
        n = len(rows)
        if n == 0:
            continue

        n_fired = sum(r["alarm_fired"] for r in rows)
        n_false = sum(r["false_alarm"] for r in rows)
        n_true_detect = sum(r["alarm_fired"] and not r["false_alarm"] for r in rows)
        delays = [
            r["detection_delay"] for r in rows if r["detection_delay"] is not None
        ]

        tpr = n_true_detect / n
        fpr = n_false / n
        tpr_ci = _wilson_ci(n_true_detect, n)
        fpr_ci = _wilson_ci(n_false, n)

        summaries[det_name] = {
            "n_trials": n,
            "alarm_rate": n_fired / n,
            "tpr": tpr,
            "tpr_ci": tpr_ci,
            "fpr": fpr,
            "fpr_ci": fpr_ci,
            "median_delay": float(np.median(delays)) if delays else float("nan"),
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "std_delay": float(np.std(delays)) if delays else float("nan"),
            "n_detections": n_true_detect,
            "delays": delays,
        }
    return summaries


# ─── Full experiment ─────────────────────────────────────────────────


def run_experiment(cfg: Config) -> dict:
    """Run the complete experiment: all scenarios × all trials × all detectors.

    Returns a JSON-serialisable results dictionary.
    """
    print(
        f"Running experiment: {cfg.n_trials} trials × "
        f"{len(cfg.drift_scenarios)} scenarios × "
        f"{len(ALL_DETECTOR_NAMES)} detectors"
    )
    print(f"  n_train={cfg.n_train}, n_stable={cfg.n_stable}, n_post={cfg.n_post}")

    all_results: dict[str, dict] = {}
    t0 = time.time()

    for drift_type, tw in cfg.drift_scenarios:
        scenario_key = f"{drift_type}_tw{tw}"
        print(f"\n── Scenario: {scenario_key} ──")

        trial_results: list[dict[str, dict]] = []

        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(run_single_trial, cfg, drift_type, tw, trial_idx): trial_idx
                for trial_idx in range(cfg.n_trials)
            }
            for i, future in enumerate(as_completed(futures), 1):
                trial_results.append(future.result())
                if i % 50 == 0 or i == cfg.n_trials:
                    print(f"  Completed {i}/{cfg.n_trials} trials")

        summary = aggregate_results(trial_results, cfg.n_stable)
        all_results[scenario_key] = summary

        # Quick console summary
        for name in ALL_DETECTOR_NAMES:
            s = summary.get(name)
            if s is None:
                continue
            print(
                f"    {name:>12s}  TPR={s['tpr']:.2%}  "
                f"FPR={s['fpr']:.2%}  "
                f"median_delay={s['median_delay']:.0f}"
            )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "config": {
            "seed": cfg.seed,
            "n_train": cfg.n_train,
            "n_stable": cfg.n_stable,
            "n_post": cfg.n_post,
            "alpha": cfg.alpha,
            "n_trials": cfg.n_trials,
            "drift_scenarios": list(cfg.drift_scenarios),
        },
        "results": all_results,
        "elapsed_seconds": elapsed,
    }


def save_results(results: dict, path: Path) -> None:
    """Save results as JSON (delays are serialised as lists)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_default)


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

"""Core experiment: run all detectors across trials and scenarios.

Each Monte-Carlo trial:
    1. Generate a fresh FriedmanDrift monitoring stream (new seed).
    2. Load the pre-trained ``ModelBundle`` from disk (shared across trials).
    3. Compute PITs, squared residuals, and binary errors on the monitoring window.
    4. Feed every detector and collect results.

The model is trained *once* (by ``train_model.py``) and shared across all
trials.  This correctly models the scenario of a fixed deployed model whose
calibration is being monitored over time.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from config import Config
from data import generate_stream
from detectors import (
    ALL_DETECTOR_NAMES,
    build_all_detectors,
)
from model import (
    ModelBundle,
    compute_pits,
    compute_predictions,
    compute_residuals,
    load_bundle,
)


# ─── Single trial ────────────────────────────────────────────────────


def run_single_trial(
    cfg: Config,
    bundle: ModelBundle,
    drift_type: str,
    transition_window: int,
    trial_idx: int,
    all_n_bins: tuple[int, ...],
    binary_threshold: float,
) -> dict:
    """Run one Monte-Carlo trial for all n_bins values simultaneously.

    Generates the monitoring stream and computes model signals once, then
    feeds each PITMonitor n_bins variant and all baseline detectors.  The
    baseline detectors (ADWIN, DDM, …) do not depend on n_bins so they are
    run once and their results are shared across all bin sizes.

    Parameters
    ----------
    cfg : Config
        Experiment configuration.
    bundle : ModelBundle
        Pre-trained model bundle (shared and read-only across trials).
    drift_type : str
        FriedmanDrift variant (``'gra'``, ``'gsg'``, or ``'lea'``).
    transition_window : int
        Gradual-drift transition width in samples (0 = abrupt).
    trial_idx : int
        Trial index, used to derive a unique per-trial seed.
    all_n_bins : tuple of int
        All PITMonitor bin sizes to evaluate.  ``all_n_bins[0]`` is the
        canonical value used in the main comparison table.
    binary_threshold : float
        Median absolute residual from the training set, used to binarize
        residuals for DDM / EDDM / HDDM_A / HDDM_W.

    Returns
    -------
    dict
        Structure::

            {
                "by_n_bins": {
                    100: {"PITMonitor": result_dict, "ADWIN": result_dict, ...},
                    ...
                }
            }

        Baseline detector results are identical across bin sizes and are
        stored under every key for a uniform access pattern.
    """
    seed = cfg.seed + trial_idx * 1000

    # ── Generate stream and compute signals once ─────────────────────
    X, y = generate_stream(cfg, drift_type, transition_window, seed=seed)
    X_mon = X[cfg.n_train :]
    y_mon = y[cfg.n_train :]

    pits = compute_pits(bundle, X_mon, y_mon)
    residuals = compute_residuals(bundle, X_mon, y_mon)
    sq_residuals = residuals**2

    # Binary error threshold from training data (no leakage)
    binary_errors = (np.abs(residuals) > binary_threshold).astype(np.float64)

    # ── Run baseline detectors once (they don't depend on n_bins) ────
    baseline_detectors = build_all_detectors(
        alpha=cfg.alpha,
        delta=cfg.delta,
        n_monitor_bins=all_n_bins[0],  # n_bins irrelevant for baselines
        seed=seed,
    )
    baseline_results: dict = {}
    for det in baseline_detectors:
        if det.name == "PITMonitor":
            continue  # handled per-n_bins below
        det.feed(pits, sq_residuals, binary_errors, cfg.n_stable)
        baseline_results[det.name] = asdict(det.result)

    # ── Run one PITMonitor per n_bins value ───────────────────────────
    from pitmon import PITMonitor

    pit_results: dict[int, dict] = {}
    for n_bins in all_n_bins:
        mon = PITMonitor(alpha=cfg.alpha, n_bins=n_bins, rng=seed)
        alarm_idx: Optional[int] = None
        for i, pit in enumerate(pits):
            alarm = mon.update(float(pit))
            if alarm.triggered and alarm_idx is None:
                alarm_idx = i
                break
        fired = alarm_idx is not None
        false_alarm = fired and alarm_idx < cfg.n_stable
        delay = (alarm_idx - cfg.n_stable) if (fired and not false_alarm) else None

        # Changepoint estimation (only meaningful if alarm fired)
        cp = mon.changepoint()
        # Convert PITMonitor's 1-based changepoint to 0-based monitoring index
        cp_idx = (cp - 1) if cp is not None else None

        pit_results[n_bins] = {
            "name": "PITMonitor",
            "alarm_fired": fired,
            "alarm_index": alarm_idx,
            "false_alarm": false_alarm,
            "detection_delay": delay,
            "changepoint_estimate": cp_idx,
        }

    # ── Assemble per-n_bins output ────────────────────────────────────
    by_n_bins: dict[int, dict] = {}
    for n_bins in all_n_bins:
        by_n_bins[n_bins] = {"PITMonitor": pit_results[n_bins], **baseline_results}

    return {"by_n_bins": by_n_bins}


# ─── Single-run artifact collection ──────────────────────────────────


def collect_single_run(
    cfg: Config,
    bundle: ModelBundle,
    drift_type: str,
    transition_window: int,
    trial_idx: int = 0,
) -> dict:
    """Run one trial and collect detailed per-step data for visualization.

    The evidence trace records PITMonitor's running e-process value at every
    step, even after an alarm fires, so the full trajectory can be plotted.
    Uses ``cfg.n_monitor_bins`` (the canonical bin size).

    Parameters
    ----------
    cfg : Config
    bundle : ModelBundle
    drift_type : str
    transition_window : int
    trial_idx : int, default=0

    Returns
    -------
    dict
        Keys: ``true_shift_point``, ``true_labels``, ``predictions``,
        ``pits``, ``evidence_trace``, ``alarm_fired``, ``alarm_time``,
        ``monitor_alpha``, ``changepoint``, ``scenario_key``.
    """
    from pitmon import PITMonitor

    seed = cfg.seed + trial_idx * 1000
    X, y = generate_stream(cfg, drift_type, transition_window, seed=seed)
    X_mon = X[cfg.n_train :]
    y_mon = y[cfg.n_train :]

    pits = compute_pits(bundle, X_mon, y_mon)
    mu, _ = compute_predictions(bundle, X_mon)

    mon = PITMonitor(alpha=cfg.alpha, n_bins=cfg.n_monitor_bins, rng=seed)
    evidence_trace = []
    alarm_time = None
    for i, pit in enumerate(pits):
        alarm = mon.update(float(pit))
        evidence_trace.append(float(alarm.evidence))
        if alarm.triggered and alarm_time is None:
            alarm_time = i + 1  # 1-based index within monitoring stream

    cp = mon.changepoint()
    scenario_key = f"{drift_type}_tw{transition_window}"

    return {
        "true_shift_point": cfg.n_stable + 1,  # 1-based within monitoring stream
        "true_labels": y_mon.tolist(),
        "predictions": mu.tolist(),
        "pits": pits.tolist(),
        "evidence_trace": evidence_trace,
        "alarm_fired": alarm_time is not None,
        "alarm_time": alarm_time,
        "monitor_alpha": cfg.alpha,
        "changepoint": int(cp) if cp is not None else None,
        "scenario_key": scenario_key,
    }


# ─── Aggregation helpers ─────────────────────────────────────────────


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.
    z : float, default=1.96
        Normal quantile for the desired coverage (1.96 → 95%).

    Returns
    -------
    (lower, upper) : tuple of float
        Bounds clipped to [0, 1].  Both are NaN when n == 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def aggregate_results(trial_results_for_bins: list[dict], n_stable: int) -> dict:
    """Aggregate per-trial detector results into summary statistics.

    Parameters
    ----------
    trial_results_for_bins : list of dict
        Each element is ``{detector_name: result_dict}`` for a fixed n_bins
        (i.e. already sliced from ``trial["by_n_bins"][n_bins]``).
    n_stable : int
        Number of pre-drift monitoring samples.

    Returns
    -------
    dict
        ``{detector_name: summary_dict}`` with keys: n_trials, alarm_rate,
        tpr, tpr_ci, fpr, fpr_ci, mean_delay, std_delay, median_delay,
        n_detections, delays, mean_cp_error (PITMonitor only).
    """
    summaries = {}
    for det_name in ALL_DETECTOR_NAMES:
        rows = [t[det_name] for t in trial_results_for_bins if det_name in t]
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

        summary = {
            "n_trials": n,
            "alarm_rate": n_fired / n,
            "tpr": tpr,
            "tpr_ci": _wilson_ci(n_true_detect, n),
            "fpr": fpr,
            "fpr_ci": _wilson_ci(n_false, n),
            "mean_delay": float(np.mean(delays)) if delays else float("nan"),
            "std_delay": float(np.std(delays)) if delays else float("nan"),
            "median_delay": float(np.median(delays)) if delays else float("nan"),
            "n_detections": n_true_detect,
            "delays": delays,
        }

        # Changepoint error for PITMonitor (mean absolute error vs true drift)
        if det_name == "PITMonitor":
            cp_errors = []
            for r in rows:
                cp = r.get("changepoint_estimate")
                if cp is not None and r["alarm_fired"] and not r["false_alarm"]:
                    # True drift is at index n_stable (0-based)
                    cp_errors.append(abs(cp - n_stable))
            summary["mean_cp_error"] = (
                float(np.mean(cp_errors)) if cp_errors else float("nan")
            )
            summary["median_cp_error"] = (
                float(np.median(cp_errors)) if cp_errors else float("nan")
            )
            summary["n_cp_estimates"] = len(cp_errors)

        summaries[det_name] = summary
    return summaries


# ─── Full experiment ─────────────────────────────────────────────────


def run_experiment(
    cfg: Config,
    bundle: Optional[ModelBundle] = None,
) -> dict:
    """Run the complete experiment: all scenarios × all trials × all detectors.

    This is the single entry point for all computation.  It:

    1. Computes the binary error threshold from training data (once).
    2. Runs ``cfg.n_trials`` Monte-Carlo trials per drift scenario.
    3. Within each trial, evaluates *all* ``cfg.n_bins_list`` PITMonitor
       variants and all baseline detectors using the same data stream.
    4. Collects one detailed single-run artifact per scenario for visualization.
    5. Aggregates results separately for every n_bins value.

    The main comparison table uses ``cfg.n_monitor_bins`` (= ``n_bins_list[0]``).
    The n_bins sensitivity data is stored alongside under ``"bins_sweep"``.

    If *bundle* is None, the model is loaded from ``cfg.bundle_path``.
    Raises ``FileNotFoundError`` if no saved model exists.

    Parameters
    ----------
    cfg : Config
    bundle : ModelBundle or None
        Pre-trained model.  If None, loaded from ``cfg.bundle_path``.

    Returns
    -------
    dict
        JSON-serializable results with keys:

        ``"config"``
            A copy of the relevant config parameters.
        ``"results"``
            Main comparison table: ``{scenario_key: {detector: summary}}``,
            computed using ``cfg.n_monitor_bins``.
        ``"single_runs"``
            Per-step visualization data: ``{scenario_key: artifact_dict}``.
        ``"bins_sweep"``
            n_bins sensitivity data.
        ``"elapsed_seconds"``
            Wall-clock time for the full experiment.
    """
    if bundle is None:
        if not cfg.bundle_path.exists():
            raise FileNotFoundError(
                f"No model bundle found at {cfg.bundle_path}.\n"
                "Run `python train_model.py` first to train and save the model."
            )
        bundle = load_bundle(cfg.bundle_path)

    all_n_bins = tuple(cfg.n_bins_list)  # evaluated together in every trial
    canonical_bins = cfg.n_monitor_bins

    # ── Compute binary error threshold from TRAINING data ────────────
    # This avoids information leakage: the threshold is determined before
    # any monitoring data is seen, exactly as in a real deployment.
    drift_type_0, tw_0 = cfg.drift_scenarios[0]
    X_for_threshold, y_for_threshold = generate_stream(
        cfg, drift_type=drift_type_0, transition_window=tw_0, seed=cfg.seed
    )
    X_train = X_for_threshold[: cfg.n_train]
    y_train = y_for_threshold[: cfg.n_train]
    train_residuals = compute_residuals(bundle, X_train, y_train)
    binary_threshold = float(np.median(np.abs(train_residuals)))
    print(
        f"Binary error threshold (training-data median |residual|): "
        f"{binary_threshold:.4f}"
    )

    print(
        f"Running experiment: {cfg.n_trials} trials × "
        f"{len(cfg.drift_scenarios)} scenarios × "
        f"{len(ALL_DETECTOR_NAMES)} detectors"
    )
    print(f"  n_bins sweep: {list(all_n_bins)}  (canonical: {canonical_bins})")
    print(f"  n_train={cfg.n_train}, n_stable={cfg.n_stable}, n_post={cfg.n_post}")

    all_results: dict = {}  # main table (canonical n_bins)
    all_single_runs: dict = {}  # per-scenario visualization artifacts
    bins_sweep_scenarios: dict = {}  # n_bins sensitivity data
    t0 = time.time()

    for drift_type, tw in cfg.drift_scenarios:
        scenario_key = f"{drift_type}_tw{tw}"
        print(f"\n── Scenario: {scenario_key} ──")

        # ── Single-run visualization artifact ─────────────────────────
        all_single_runs[scenario_key] = collect_single_run(
            cfg, bundle, drift_type, tw, trial_idx=0
        )

        # ── Monte-Carlo trials ────────────────────────────────────────
        raw_trial_results: list[dict] = []
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
            futures = {
                pool.submit(
                    run_single_trial,
                    cfg,
                    bundle,
                    drift_type,
                    tw,
                    trial_idx,
                    all_n_bins,
                    binary_threshold,
                ): trial_idx
                for trial_idx in range(cfg.n_trials)
            }
            for i, future in enumerate(as_completed(futures), 1):
                raw_trial_results.append(future.result())
                if i % 500 == 0 or i == cfg.n_trials:
                    print(f"  Completed {i}/{cfg.n_trials} trials")

        # ── Aggregate: canonical n_bins (main table) ──────────────────
        canonical_slice = [t["by_n_bins"][canonical_bins] for t in raw_trial_results]
        summary = aggregate_results(canonical_slice, cfg.n_stable)
        all_results[scenario_key] = summary

        for name in ALL_DETECTOR_NAMES:
            s = summary.get(name)
            if s is None:
                continue
            cp_str = ""
            if name == "PITMonitor" and not np.isnan(
                s.get("mean_cp_error", float("nan"))
            ):
                cp_str = f"  mean_cp_err={s['mean_cp_error']:.0f}"
            print(
                f"    {name:>12s}  TPR={s['tpr']:.2%}  "
                f"FPR={s['fpr']:.2%}  "
                f"mean_delay={s['mean_delay']:.0f}{cp_str}"
            )

        # ── Aggregate: every n_bins (sweep) ───────────────────────────
        bins_sweep_scenarios[scenario_key] = {}
        for n_bins in all_n_bins:
            bins_slice = [t["by_n_bins"][n_bins] for t in raw_trial_results]
            bins_summary = aggregate_results(bins_slice, cfg.n_stable)
            pit_s = bins_summary.get("PITMonitor", {})
            bins_sweep_scenarios[scenario_key][n_bins] = {
                "tpr": pit_s.get("tpr", float("nan")),
                "tpr_ci": pit_s.get("tpr_ci", (float("nan"), float("nan"))),
                "fpr": pit_s.get("fpr", float("nan")),
                "fpr_ci": pit_s.get("fpr_ci", (float("nan"), float("nan"))),
                "mean_delay": pit_s.get("mean_delay", float("nan")),
                "median_delay": pit_s.get("median_delay", float("nan")),
            }

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    return {
        "config": {
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "lr": cfg.lr,
            "n_train": cfg.n_train,
            "n_stable": cfg.n_stable,
            "n_post": cfg.n_post,
            "alpha": cfg.alpha,
            "delta": cfg.delta,
            "n_bins": canonical_bins,
            "n_bins_list": list(all_n_bins),
            "n_trials": cfg.n_trials,
            "drift_scenarios": list(cfg.drift_scenarios),
            "binary_threshold": binary_threshold,
        },
        "results": all_results,
        "single_runs": all_single_runs,
        "bins_sweep": {
            "n_bins_list": list(all_n_bins),
            "scenarios": bins_sweep_scenarios,
        },
        "elapsed_seconds": elapsed,
    }


# ─── I/O helpers ─────────────────────────────────────────────────────


def save_results(results: dict, path: Path) -> None:
    """Save a results dictionary as indented JSON.

    Handles numpy scalar types transparently so that ``json.dump`` never raises
    a ``TypeError`` on the values produced by ``aggregate_results``.

    Parameters
    ----------
    results : dict
    path : Path
        Destination file; parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_default)


def load_results(path: Path) -> dict:
    """Load a JSON results file previously written by ``save_results``.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
    """
    with open(path) as f:
        return json.load(f)

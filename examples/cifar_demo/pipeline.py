from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pitmon import PITMonitor

from .config import CifarDemoConfig
from .core import (
    build_true_prob_reference,
    confidence_pits_from_reference,
    load_cifar10_train_test,
    load_cifar10c_corruption,
    predict_proba_safe,
    run_baselines_one_pass,
    run_pitmonitor_trial,
    summarize_trials,
    train_classifier,
)


def _build_reference_pool(
    x_test: np.ndarray,
    y_test: np.ndarray,
    n_ref_cal: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(len(x_test), size=n_ref_cal, replace=False)
    test_pool_idx = np.setdiff1d(np.arange(len(x_test)), ref_idx)
    return (
        x_test[ref_idx],
        y_test[ref_idx],
        x_test[test_pool_idx],
        y_test[test_pool_idx],
    )


def _h0_trial(
    severity_level: int,
    trial_idx: int,
    cfg: CifarDemoConfig,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    classifier,
    scaler,
    pit_reference: np.ndarray,
) -> dict:
    base_seed = cfg.seed + severity_level * 1_000_000
    rng = np.random.default_rng(base_seed + 50_000 + trial_idx)
    stable_idx = rng.choice(len(x_pool), size=cfg.n_stable_power, replace=False)
    shifted_idx = rng.choice(len(x_pool), size=cfg.n_shifted_power, replace=False)

    x_stable = x_pool[stable_idx]
    y_stable = y_pool[stable_idx]
    x_shifted = x_pool[shifted_idx]
    y_shifted = y_pool[shifted_idx]

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.int64, copy=False)

    probs = predict_proba_safe(classifier, x_all, scaler).astype(np.float32, copy=False)
    return run_pitmonitor_trial(
        probs,
        y_all,
        pit_reference,
        alpha=cfg.alpha_power,
        n_bins=cfg.n_bins,
        n_stable=cfg.n_stable_power,
        pit_seed=base_seed + 150_000 + trial_idx,
    )


def _h1_trial(
    severity_level: int,
    trial_idx: int,
    cfg: CifarDemoConfig,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_corr: np.ndarray,
    y_corr: np.ndarray,
    classifier,
    scaler,
    pit_reference: np.ndarray,
) -> dict:
    rng = np.random.default_rng(cfg.seed + severity_level * 10_000 + trial_idx)
    stable_idx = rng.choice(len(x_pool), size=cfg.n_stable_power, replace=False)
    shifted_idx = rng.choice(len(x_corr), size=cfg.n_shifted_power, replace=False)

    x_stable = x_pool[stable_idx]
    y_stable = y_pool[stable_idx]
    x_shifted = x_corr[shifted_idx]
    y_shifted = y_corr[shifted_idx]

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.int64, copy=False)

    probs = predict_proba_safe(classifier, x_all, scaler).astype(np.float32, copy=False)
    pred_labels = probs.argmax(axis=1).astype(np.int16, copy=False)
    error_stream = (pred_labels != y_all).astype(np.uint8, copy=False)

    pit_trial = run_pitmonitor_trial(
        probs,
        y_all,
        pit_reference,
        alpha=cfg.alpha_power,
        n_bins=cfg.n_bins,
        n_stable=cfg.n_stable_power,
        pit_seed=cfg.seed + severity_level * 100_000 + trial_idx,
    )

    return {
        "pit_trial": pit_trial,
        "error_stream": error_stream,
        "n_stable": cfg.n_stable_power,
    }


def _baseline_h0_error_stream_trial(
    trial_idx: int,
    cfg: CifarDemoConfig,
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    classifier,
    scaler,
) -> np.ndarray:
    """Generate an error stream for a clean→clean (H0) run.

    Both the "stable" and "shifted" segments are sampled from the same clean pool,
    so there is no distribution shift; any alarm from a baseline detector is a false alarm
    in this experiment.
    """
    rng = np.random.default_rng(cfg.seed + 90_000 + trial_idx)
    stable_idx = rng.choice(len(x_pool), size=cfg.n_stable_power, replace=False)
    shifted_idx = rng.choice(len(x_pool), size=cfg.n_shifted_power, replace=False)

    x_stable = x_pool[stable_idx]
    y_stable = y_pool[stable_idx]
    x_shifted = x_pool[shifted_idx]
    y_shifted = y_pool[shifted_idx]

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.int64, copy=False)

    probs = predict_proba_safe(classifier, x_all, scaler).astype(np.float32, copy=False)
    pred_labels = probs.argmax(axis=1).astype(np.int16, copy=False)
    return (pred_labels != y_all).astype(np.uint8, copy=False)


def compute_all(cfg: CifarDemoConfig) -> dict:
    cfg = cfg.normalized()
    t0 = time.time()

    x_train, y_train, x_test, y_test = load_cifar10_train_test(cfg.data_dir)
    rng_train = np.random.default_rng(cfg.seed)
    train_idx = rng_train.choice(len(x_train), size=cfg.train_size, replace=False)
    x_train_sub = x_train[train_idx]
    y_train_sub = y_train[train_idx]

    classifier, scaler = train_classifier(x_train_sub, y_train_sub, cfg.seed)

    n_ref = min(cfg.n_ref_cal, len(x_test) // 2)
    x_ref, y_ref, x_pool, y_pool = _build_reference_pool(
        x_test, y_test, n_ref, cfg.seed + 40_000
    )
    probs_ref = predict_proba_safe(classifier, x_ref, scaler).astype(
        np.float32, copy=False
    )
    pit_reference = build_true_prob_reference(probs_ref, y_ref)

    x_corr_demo, y_corr_demo = load_cifar10c_corruption(
        cfg.data_dir,
        cfg.corruption,
        cfg.severity_demo,
    )
    rng_demo = np.random.default_rng(cfg.seed + 1_111)
    stable_demo_idx = rng_demo.choice(len(x_pool), size=cfg.n_stable, replace=False)
    shifted_demo_idx = rng_demo.choice(
        len(x_corr_demo), size=cfg.n_shifted, replace=False
    )
    x_demo_stable = x_pool[stable_demo_idx]
    y_demo_stable = y_pool[stable_demo_idx]
    x_demo_shifted = x_corr_demo[shifted_demo_idx]
    y_demo_shifted = y_corr_demo[shifted_demo_idx]

    x_demo_all = np.vstack([x_demo_stable, x_demo_shifted]).astype(
        np.float32, copy=False
    )
    y_demo_all = np.concatenate([y_demo_stable, y_demo_shifted]).astype(
        np.int64, copy=False
    )
    probs_demo = predict_proba_safe(classifier, x_demo_all, scaler).astype(
        np.float32, copy=False
    )
    pred_demo = probs_demo.argmax(axis=1).astype(np.int16, copy=False)

    single_summary = run_pitmonitor_trial(
        probs_demo,
        y_demo_all,
        pit_reference,
        alpha=cfg.alpha,
        n_bins=cfg.n_bins,
        n_stable=cfg.n_stable,
        pit_seed=cfg.seed + 3_333,
    )

    pits_stream_values = confidence_pits_from_reference(
        probs_demo,
        y_demo_all,
        pit_reference,
        np.random.default_rng(cfg.seed + 3_333),
    ).astype(np.float32, copy=False)
    monitor_demo = PITMonitor(alpha=cfg.alpha, n_bins=cfg.n_bins)
    evidence_trace = np.empty(len(pits_stream_values), dtype=np.float32)
    for index, pit in enumerate(pits_stream_values):
        update = monitor_demo.update(float(pit))
        evidence_trace[index] = float(update.evidence)
    demo_monitor_summary = monitor_demo.summary()

    h0_by_severity: dict[int, dict] = {}
    h1_cache: dict[int, list[dict]] = {}
    h1_by_severity: dict[int, dict] = {}

    for severity_level in cfg.severity_levels:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            h0_trials = list(
                executor.map(
                    lambda idx: _h0_trial(
                        severity_level,
                        idx,
                        cfg,
                        x_pool,
                        y_pool,
                        classifier,
                        scaler,
                        pit_reference,
                    ),
                    range(cfg.n_trials),
                )
            )
        h0_by_severity[severity_level] = summarize_trials(h0_trials, cfg.n_trials)

        x_corr, y_corr = load_cifar10c_corruption(
            cfg.data_dir, cfg.corruption, severity_level
        )
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            h1_trials = list(
                executor.map(
                    lambda idx: _h1_trial(
                        severity_level,
                        idx,
                        cfg,
                        x_pool,
                        y_pool,
                        x_corr,
                        y_corr,
                        classifier,
                        scaler,
                        pit_reference,
                    ),
                    range(cfg.n_trials),
                )
            )
        h1_cache[severity_level] = h1_trials
        h1_by_severity[severity_level] = summarize_trials(
            [trial["pit_trial"] for trial in h1_trials],
            cfg.n_trials,
        )

    # Baseline detectors under H0: clean → clean (no distribution shift).
    # This evaluates how often DDM/EDDM/ADWIN/KSWIN raise (early) alarms when
    # there is in fact no change, on the same horizon used for power trials.
    compare_methods = ["PITMonitor", "DDM", "EDDM", "ADWIN", "KSWIN"]
    compare_baselines = ["DDM", "EDDM", "ADWIN", "KSWIN"]

    baseline_h0_trials: dict[str, list[dict]] = {name: [] for name in compare_baselines}
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        error_streams_h0 = list(
            executor.map(
                lambda idx: _baseline_h0_error_stream_trial(
                    idx,
                    cfg,
                    x_pool,
                    y_pool,
                    classifier,
                    scaler,
                ),
                range(cfg.n_trials),
            )
        )

    for error_stream in error_streams_h0:
        det_runs = run_baselines_one_pass(error_stream, cfg.n_stable_power)
        for name in compare_baselines:
            result = det_runs[name]
            baseline_h0_trials[name].append(
                {
                    "alarm_fired": result.alarm_time is not None,
                    # Under H0 (clean→clean), *any* alarm is a false alarm,
                    # regardless of when it occurs.
                    "false_alarm": result.alarm_time is not None,
                    "detection_delay": None,
                    "final_evidence": float("nan"),
                }
            )

    baseline_h0_results_by_method = {
        name: summarize_trials(trials, cfg.n_trials)
        for name, trials in baseline_h0_trials.items()
    }
    comparison_rows = []

    for severity_level in cfg.severity_levels:
        trials = h1_cache[severity_level]
        method_trials = {method: [] for method in compare_methods}

        for trial in trials:
            pit_trial = trial["pit_trial"]
            n_stable = int(trial["n_stable"])
            error_stream = np.asarray(trial["error_stream"], dtype=np.float32)
            det_runs = run_baselines_one_pass(error_stream, n_stable)

            method_trials["PITMonitor"].append(pit_trial)
            for detector in compare_baselines:
                result = det_runs[detector]
                method_trials[detector].append(
                    {
                        "alarm_fired": result.alarm_time is not None,
                        "false_alarm": result.false_alarm,
                        "detection_delay": result.detection_delay,
                        "final_evidence": float("nan"),
                    }
                )

        for method in compare_methods:
            summary = summarize_trials(method_trials[method], cfg.n_trials)
            comparison_rows.append(
                {
                    "severity": severity_level,
                    "method": method,
                    "false_alarm_rate": summary["false_alarm_rate"],
                    "tpr": summary["tpr"],
                    "median_delay": summary["median_delay"],
                }
            )

    return {
        "config": cfg.to_dict(),
        "train_accuracy": float(
            classifier.score(scaler.transform(x_train_sub), y_train_sub)
        ),
        "single_run": {
            "severity": cfg.severity_demo,
            "true_shift_point": cfg.n_stable + 1,
            "pred_labels": pred_demo,
            "true_labels": y_demo_all.astype(np.int16, copy=False),
            "pits": pits_stream_values,
            "evidence_trace": evidence_trace,
            "alarm_fired": bool(monitor_demo.alarm_triggered),
            "alarm_time": monitor_demo.alarm_time,
            "final_evidence": float(monitor_demo.evidence),
            "monitor_alpha": cfg.alpha,
            "changepoint": demo_monitor_summary.get("changepoint"),
            "pit_trial_summary": single_summary,
        },
        "h0_results_by_severity": h0_by_severity,
        "power_results_by_severity": h1_by_severity,
        "baseline_h0_results_by_method": baseline_h0_results_by_method,
        "comparison_rows": comparison_rows,
        "shared_h1_trial_cache": h1_cache,
        "shared_trial_cache_meta": {
            "corruption": cfg.corruption,
            "n_stable": cfg.n_stable_power,
            "n_shifted": cfg.n_shifted_power,
            "n_trials": cfg.n_trials,
            "severities": list(cfg.severity_levels),
            "pit_method": "confidence_ecdf_reference",
        },
        "elapsed_seconds": time.time() - t0,
    }


def save_artifacts(artifacts: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(artifacts, file_handle)


def load_artifacts(path: Path) -> dict:
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)


def augment_artifacts_with_baseline_h0(
    artifacts: dict,
    cfg: CifarDemoConfig,
) -> dict:
    """Reuse an existing artifact and only compute missing baseline H0 results.

    This avoids re-running the expensive PITMonitor H0/H1 trials when they are
    already present. We only simulate clean→clean error streams and run the
    river detectors to obtain their empirical H0 false-alarm rates.
    """
    # Basic compatibility check: same corruption and trial count.
    art_cfg = artifacts.get("config", {})
    if (
        art_cfg.get("corruption") != cfg.corruption
        or art_cfg.get("n_trials") != cfg.n_trials
    ):
        return artifacts

    # Reload minimal state needed to simulate new H0 error streams.
    try:
        x_train, y_train, x_test, y_test = load_cifar10_train_test(cfg.data_dir)
    except FileNotFoundError:
        # If CIFAR-10 data is not available, skip adding baseline H0 results
        # but still allow reuse of the existing artifact.
        return artifacts

    rng_train = np.random.default_rng(cfg.seed)
    train_idx = rng_train.choice(len(x_train), size=cfg.train_size, replace=False)
    x_train_sub = x_train[train_idx]
    y_train_sub = y_train[train_idx]
    classifier, scaler = train_classifier(x_train_sub, y_train_sub, cfg.seed)

    n_ref = min(cfg.n_ref_cal, len(x_test) // 2)
    _, _, x_pool, y_pool = _build_reference_pool(
        x_test, y_test, n_ref, cfg.seed + 40_000
    )

    baseline_h0_trials: dict[str, list[dict]] = {
        name: [] for name in ["DDM", "EDDM", "ADWIN", "KSWIN"]
    }
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        error_streams_h0 = list(
            executor.map(
                lambda idx: _baseline_h0_error_stream_trial(
                    idx,
                    cfg,
                    x_pool,
                    y_pool,
                    classifier,
                    scaler,
                ),
                range(cfg.n_trials),
            )
        )

    from .core import run_baselines_one_pass  # local import to avoid cycles

    for error_stream in error_streams_h0:
        det_runs = run_baselines_one_pass(error_stream, cfg.n_stable_power)
        for name in baseline_h0_trials:
            result = det_runs[name]
            baseline_h0_trials[name].append(
                {
                    "alarm_fired": result.alarm_time is not None,
                    # Under H0 (clean→clean), any alarm is a false alarm.
                    "false_alarm": result.alarm_time is not None,
                    "detection_delay": None,
                    "final_evidence": float("nan"),
                }
            )

    artifacts["baseline_h0_results_by_method"] = {
        name: summarize_trials(trials, cfg.n_trials)
        for name, trials in baseline_h0_trials.items()
    }
    return artifacts

from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from pitmon import PITMonitor

from config import DeliveryDemoConfig
from core import (
    batch_predictive_cdf,
    build_bin_edges,
    generate_features,
    make_mlp_classifier,
    predict_mean_batch,
    report_ks_calibration,
    run_baselines_one_pass,
    run_pitmonitor_trial,
    summarize_trials,
    true_delivery_time_regime,
    true_delivery_time_shift,
    y_to_bin_index,
)


def _fit_delivery_model(cfg: DeliveryDemoConfig) -> dict:
    rng_train = np.random.default_rng(cfg.seed + 10)
    x_train = generate_features(cfg.n_train, rng_train)
    y_train = true_delivery_time_regime(x_train, rng_train, regime="before")

    rng_cal = np.random.default_rng(cfg.seed + 20)
    x_cal = generate_features(cfg.n_cal, rng_cal)
    y_cal = true_delivery_time_regime(x_cal, rng_cal, regime="before")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_cal_scaled = scaler.transform(x_cal)

    bin_edges = build_bin_edges(y_train, cfg.n_y_bins)
    n_bins_y = len(bin_edges) - 1
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    y_train_bins = y_to_bin_index(y_train, bin_edges)
    y_cal_bins = y_to_bin_index(y_cal, bin_edges)

    nn_model = make_mlp_classifier(cfg.seed)
    nn_model.fit(x_train_scaled, y_train_bins)

    return {
        "scaler": scaler,
        "nn": nn_model,
        "bin_edges": bin_edges,
        "n_bins_y": n_bins_y,
        "bin_centers": bin_centers,
        "classes_seen": nn_model.classes_.astype(np.int32, copy=False),
        "train_accuracy": float(nn_model.score(x_train_scaled, y_train_bins)),
        "cal_accuracy": float(nn_model.score(x_cal_scaled, y_cal_bins)),
    }


def _single_run_demo(cfg: DeliveryDemoConfig, model: dict) -> dict:
    scaler = model["scaler"]
    nn_model = model["nn"]
    classes_seen = model["classes_seen"]
    n_bins_y = model["n_bins_y"]
    bin_edges = model["bin_edges"]
    bin_centers = model["bin_centers"]

    rng_stable = np.random.default_rng(cfg.seed + 100)
    x_stable = generate_features(cfg.n_stable, rng_stable)
    y_stable = true_delivery_time_regime(x_stable, rng_stable, regime="before")

    rng_shifted = np.random.default_rng(cfg.seed + 200)
    x_shifted = generate_features(cfg.n_shifted, rng_shifted)
    y_shifted = true_delivery_time_regime(x_shifted, rng_shifted, regime="after")

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.float32, copy=False)

    pits = batch_predictive_cdf(
        x_all,
        y_all,
        scaler,
        nn_model,
        classes_seen,
        n_bins_y,
        bin_edges,
    ).astype(np.float32, copy=False)

    preds = predict_mean_batch(
        x_all,
        scaler,
        nn_model,
        classes_seen,
        n_bins_y,
        bin_centers,
    ).astype(np.float32, copy=False)

    monitor = PITMonitor(alpha=cfg.alpha, n_bins=cfg.n_monitor_bins)
    evidence_trace = np.empty(len(pits), dtype=np.float32)
    for i, pit in enumerate(pits):
        alarm = monitor.update(float(pit))
        evidence_trace[i] = float(alarm.evidence)

    summary = monitor.summary()
    ks = report_ks_calibration(pits[: cfg.n_stable])

    return {
        "true_shift_point": cfg.n_stable + 1,
        "predictions": preds,
        "true_labels": y_all,
        "pits": pits,
        "evidence_trace": evidence_trace,
        "alarm_fired": bool(summary["alarm_triggered"]),
        "alarm_time": summary["alarm_time"],
        "final_evidence": float(summary["evidence"]),
        "changepoint": summary.get("changepoint"),
        "monitor_alpha": cfg.alpha,
        "pit_trial_summary": monitor.trial_summary(cfg.n_stable),
        "ks_calibration_pre_shift": ks,
    }


def _power_trial(
    cfg: DeliveryDemoConfig,
    model: dict,
    shift_fraction: float,
    trial_seed: int,
) -> dict:
    rng = np.random.default_rng(trial_seed)
    x_stable = generate_features(cfg.n_stable_power, rng)
    y_stable = true_delivery_time_shift(x_stable, rng, shift_fraction=0.0)
    x_shifted = generate_features(cfg.n_shifted_power, rng)
    y_shifted = true_delivery_time_shift(x_shifted, rng, shift_fraction=shift_fraction)

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.float32, copy=False)

    pits = batch_predictive_cdf(
        x_all,
        y_all,
        model["scaler"],
        model["nn"],
        model["classes_seen"],
        model["n_bins_y"],
        model["bin_edges"],
    ).astype(np.float32, copy=False)

    return run_pitmonitor_trial(
        pits,
        alpha=cfg.alpha_power,
        n_bins=cfg.n_monitor_bins,
        n_stable=cfg.n_stable_power,
    )


def _comparison_trial(
    cfg: DeliveryDemoConfig,
    model: dict,
    shift_fraction: float,
    trial_seed: int,
) -> dict[str, dict]:
    rng = np.random.default_rng(trial_seed)
    x_stable = generate_features(cfg.n_stable_power, rng)
    y_stable = true_delivery_time_shift(x_stable, rng, shift_fraction=0.0)
    x_shifted = generate_features(cfg.n_shifted_power, rng)
    y_shifted = true_delivery_time_shift(x_shifted, rng, shift_fraction=shift_fraction)

    x_all = np.vstack([x_stable, x_shifted]).astype(np.float32, copy=False)
    y_all = np.concatenate([y_stable, y_shifted]).astype(np.float32, copy=False)

    pits = batch_predictive_cdf(
        x_all,
        y_all,
        model["scaler"],
        model["nn"],
        model["classes_seen"],
        model["n_bins_y"],
        model["bin_edges"],
    ).astype(np.float32, copy=False)

    pit_trial = run_pitmonitor_trial(
        pits,
        alpha=cfg.alpha_power,
        n_bins=cfg.n_monitor_bins,
        n_stable=cfg.n_stable_power,
    )

    preds = predict_mean_batch(
        x_all,
        model["scaler"],
        model["nn"],
        model["classes_seen"],
        model["n_bins_y"],
        model["bin_centers"],
    ).astype(np.float32, copy=False)
    abs_err = np.abs(preds - y_all).astype(np.float32, copy=False)
    threshold_err = float(np.quantile(abs_err[: cfg.n_stable_power], 0.70))
    error_stream = (abs_err > threshold_err).astype(np.float32, copy=False)

    baselines = run_baselines_one_pass(error_stream, cfg.n_stable_power)
    out: dict[str, dict] = {"PITMonitor": pit_trial}
    for name in ["DDM", "EDDM", "ADWIN", "KSWIN"]:
        run = baselines[name]
        out[name] = {
            "alarm_fired": run.alarm_time is not None,
            "false_alarm": run.false_alarm,
            "detection_delay": run.detection_delay,
            "final_evidence": float("nan"),
        }
    return out


def compute_all(cfg: DeliveryDemoConfig) -> dict:
    cfg = cfg.normalized()
    start_time = time.time()

    model = _fit_delivery_model(cfg)
    single_run = _single_run_demo(cfg, model)

    power_results_by_shift: dict[float, dict] = {}
    for shift_fraction in cfg.shift_levels:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            trials = list(
                executor.map(
                    lambda idx, sf=shift_fraction: _power_trial(
                        cfg,
                        model,
                        sf,
                        cfg.seed + int(sf * 10_000) * 10_000 + idx,
                    ),
                    range(cfg.n_trials),
                )
            )
        power_results_by_shift[shift_fraction] = summarize_trials(trials, cfg.n_trials)

    compare_methods = ["PITMonitor", "DDM", "EDDM", "ADWIN", "KSWIN"]
    comparison_rows: list[dict] = []
    for shift_fraction in cfg.compare_shift_levels:
        method_trials = {name: [] for name in compare_methods}
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            trial_outputs = list(
                executor.map(
                    lambda idx, sf=shift_fraction: _comparison_trial(
                        cfg,
                        model,
                        sf,
                        700_000 + int(sf * 10_000) * 10_000 + idx,
                    ),
                    range(cfg.n_trials_compare),
                )
            )

        for trial in trial_outputs:
            for method in compare_methods:
                method_trials[method].append(trial[method])

        for method in compare_methods:
            summary = summarize_trials(method_trials[method], cfg.n_trials_compare)
            comparison_rows.append(
                {
                    "shift": shift_fraction,
                    "method": method,
                    "false_alarm_rate": summary["false_alarm_rate"],
                    "tpr": summary["tpr"],
                    "median_delay": summary["median_delay"],
                }
            )

    return {
        "config": cfg.to_dict(),
        "model_diagnostics": {
            "train_accuracy": model["train_accuracy"],
            "cal_accuracy": model["cal_accuracy"],
            "n_output_bins": model["n_bins_y"],
        },
        "single_run": single_run,
        "power_results_by_shift": power_results_by_shift,
        "comparison_rows": comparison_rows,
        "elapsed_seconds": time.time() - start_time,
    }


def save_artifacts(artifacts: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as file_handle:
        pickle.dump(artifacts, file_handle)


def load_artifacts(path: Path) -> dict:
    with open(path, "rb") as file_handle:
        return pickle.load(file_handle)

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river.drift import ADWIN, KSWIN
from river.drift.binary import DDM, EDDM
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pitmon import PITMonitor


@dataclass
class BaselineRun:
    warning_time: int | None
    alarm_time: int | None
    false_alarm: bool
    detection_delay: int | None


def generate_features(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    distance = rng.lognormal(mean=4.5, sigma=0.6, size=n_samples).astype(np.float32)
    weight = (rng.exponential(scale=5, size=n_samples) + 0.5).astype(np.float32)
    complexity = rng.beta(2, 5, size=n_samples).astype(np.float32)
    time_of_day = rng.uniform(0, 2 * np.pi, size=n_samples).astype(np.float32)
    return np.column_stack([distance, weight, complexity, time_of_day]).astype(
        np.float32,
        copy=False,
    )


def _delivery_components(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance, weight, complexity, tod = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    base = 8 + 0.15 * distance + 0.08 * distance * complexity + 0.3 * weight
    rush_hour = 3.0 * np.exp(-((tod - 2.5) ** 2) / 0.5)
    noise_std = 1.5 + 0.02 * distance
    return (
        base.astype(np.float32),
        rush_hour.astype(np.float32),
        noise_std.astype(np.float32),
    )


def true_delivery_time_regime(
    X: np.ndarray,
    rng: np.random.Generator,
    regime: str = "before",
) -> np.ndarray:
    base, rush_hour, noise_std = _delivery_components(X)
    noise = rng.normal(0.0, noise_std, size=len(X)).astype(np.float32)

    if regime == "after":
        highway_benefit = 0.30 * base * (1.0 / (1.0 + np.exp(-0.03 * (X[:, 0] - 50))))
        return (base + rush_hour - highway_benefit + noise * 0.7).astype(
            np.float32,
            copy=False,
        )
    return (base + rush_hour + noise).astype(np.float32, copy=False)


def true_delivery_time_shift(
    X: np.ndarray,
    rng: np.random.Generator,
    shift_fraction: float,
) -> np.ndarray:
    base, rush_hour, noise_std = _delivery_components(X)
    noise = rng.normal(0.0, noise_std, size=len(X)).astype(np.float32)

    if shift_fraction > 0:
        benefit = shift_fraction * base * (1.0 / (1.0 + np.exp(-0.03 * (X[:, 0] - 50))))
        return (
            base + rush_hour - benefit + noise * (1.0 - 0.3 * shift_fraction)
        ).astype(
            np.float32,
            copy=False,
        )
    return (base + rush_hour + noise).astype(np.float32, copy=False)


def make_mlp_classifier(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        learning_rate_init=0.001,
    )


def build_bin_edges(y: np.ndarray, n_bins: int) -> np.ndarray:
    inner = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    inner = np.unique(inner)
    if len(inner) < 3:
        raise RuntimeError("Not enough unique target values to build bins")
    pad = max(float(np.std(y)), 1.0)
    return np.concatenate(([inner[0] - pad], inner[1:-1], [inner[-1] + pad])).astype(
        np.float32,
        copy=False,
    )


def y_to_bin_index(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    return np.digitize(y, bin_edges[1:-1], right=False)


def expand_probs_to_full_bins(
    probs_seen: np.ndarray,
    classes_seen: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    probs_full = np.zeros(n_bins, dtype=np.float32)
    probs_full[classes_seen.astype(int)] = probs_seen
    total = probs_full.sum()
    if total <= 0:
        probs_full[:] = 1.0 / n_bins
    else:
        probs_full /= total
    return probs_full


def cdf_from_edges(probs: np.ndarray, y: float, edges: np.ndarray) -> float:
    if y <= edges[0]:
        return 0.0
    if y >= edges[-1]:
        return 1.0

    n_local_bins = len(edges) - 1
    index = int(np.searchsorted(edges, y, side="right") - 1)
    index = int(np.clip(index, 0, n_local_bins - 1))

    cdf_left = float(probs[:index].sum()) if index > 0 else 0.0
    width = float(edges[index + 1] - edges[index])
    frac = 0.0 if width <= 0 else float((y - edges[index]) / width)
    frac = float(np.clip(frac, 0.0, 1.0))

    return float(np.clip(cdf_left + probs[index] * frac, 0.0, 1.0))


def batch_predictive_cdf(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    scaler: StandardScaler,
    nn_model: MLPClassifier,
    classes_seen: np.ndarray,
    n_bins: int,
    bin_edges: np.ndarray,
) -> np.ndarray:
    x_scaled = scaler.transform(np.asarray(X_batch, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )
    probs_seen_batch = nn_model.predict_proba(x_scaled).astype(np.float32, copy=False)
    cdfs = np.empty(len(y_batch), dtype=np.float32)

    for i, y_value in enumerate(y_batch):
        probs = expand_probs_to_full_bins(probs_seen_batch[i], classes_seen, n_bins)
        cdfs[i] = cdf_from_edges(probs, float(y_value), bin_edges)
    return np.clip(cdfs, 0.0, 1.0).astype(np.float32, copy=False)


def predict_mean_batch(
    X_batch: np.ndarray,
    scaler: StandardScaler,
    nn_model: MLPClassifier,
    classes_seen: np.ndarray,
    n_bins: int,
    bin_centers: np.ndarray,
) -> np.ndarray:
    x_scaled = scaler.transform(np.asarray(X_batch, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )
    probs_seen_batch = nn_model.predict_proba(x_scaled).astype(np.float32, copy=False)
    means = np.empty(len(X_batch), dtype=np.float32)
    for i, row in enumerate(probs_seen_batch):
        probs_full = expand_probs_to_full_bins(row, classes_seen, n_bins)
        means[i] = float(np.dot(probs_full, bin_centers))
    return means


def run_pitmonitor_trial(
    pits: np.ndarray,
    alpha: float,
    n_bins: int,
    n_stable: int,
) -> dict:
    monitor = PITMonitor(alpha=alpha, n_bins=n_bins)
    monitor.update_many(np.asarray(pits, dtype=np.float32), stop_on_alarm=True)
    return monitor.trial_summary(n_stable)


def summarize_trials(rows: list[dict], n_trials: int) -> dict:
    n_alarm = sum(row["alarm_fired"] for row in rows)
    n_false = sum(row["false_alarm"] for row in rows)
    n_detect = sum(1 for row in rows if row["alarm_fired"] and not row["false_alarm"])
    delays = [
        row["detection_delay"] for row in rows if row.get("detection_delay") is not None
    ]

    def wilson_ci(
        k_success: int, n_total: int, z_score: float = 1.96
    ) -> tuple[float, float]:
        if n_total <= 0:
            return float("nan"), float("nan")
        p_hat = k_success / n_total
        denom = 1 + z_score**2 / n_total
        center = (p_hat + z_score**2 / (2 * n_total)) / denom
        radius = (
            z_score
            * np.sqrt((p_hat * (1 - p_hat) + z_score**2 / (4 * n_total)) / n_total)
            / denom
        )
        return float(max(0.0, center - radius)), float(min(1.0, center + radius))

    tpr_ci_low, tpr_ci_high = wilson_ci(n_detect, n_trials)
    fpr_ci_low, fpr_ci_high = wilson_ci(n_false, n_trials)

    return {
        "n_trials": n_trials,
        "alarm_rate": n_alarm / n_trials,
        "false_alarm_rate": n_false / n_trials,
        "tpr": n_detect / n_trials,
        "tpr_ci_low": tpr_ci_low,
        "tpr_ci_high": tpr_ci_high,
        "fpr_ci_low": fpr_ci_low,
        "fpr_ci_high": fpr_ci_high,
        "median_delay": float(np.median(delays)) if delays else float("nan"),
        "mean_delay": float(np.mean(delays)) if delays else float("nan"),
        "delays": delays,
        "evidences": [row.get("final_evidence", float("nan")) for row in rows],
    }


def make_baseline_detectors() -> dict[str, object]:
    return {
        "DDM": DDM(),
        "EDDM": EDDM(),
        "ADWIN": ADWIN(),
        "KSWIN": KSWIN(),
    }


def run_baselines_one_pass(
    error_stream: np.ndarray,
    n_stable: int,
) -> dict[str, BaselineRun]:
    detectors = make_baseline_detectors()
    warning_time = {name: None for name in detectors}
    alarm_time = {name: None for name in detectors}

    for t, value in enumerate(np.asarray(error_stream, dtype=np.float32), start=1):
        for name, detector in detectors.items():
            if alarm_time[name] is not None:
                continue
            detector.update(float(value))
            if warning_time[name] is None and bool(
                getattr(detector, "warning_detected", False)
            ):
                warning_time[name] = t
            if bool(getattr(detector, "drift_detected", False)):
                alarm_time[name] = t

    out = {}
    for name in detectors:
        alarm = alarm_time[name]
        false_alarm = alarm is not None and alarm <= n_stable
        delay = alarm - n_stable if alarm is not None and alarm > n_stable else None
        out[name] = BaselineRun(
            warning_time=warning_time[name],
            alarm_time=alarm,
            false_alarm=false_alarm,
            detection_delay=delay,
        )
    return out


def report_ks_calibration(pits: np.ndarray) -> dict:
    ks_stat, ks_pvalue = stats.kstest(np.asarray(pits, dtype=np.float32), "uniform")
    return {
        "ks_stat": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }

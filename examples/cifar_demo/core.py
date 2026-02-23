from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from river.drift import ADWIN, KSWIN
from river.drift.binary import DDM, EDDM
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from pitmon import PITMonitor


@dataclass
class BaselineRun:
    warning_time: int | None
    alarm_time: int | None
    false_alarm: bool
    detection_delay: int | None


def _load_cifar_batch(batch_file: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(batch_file, "rb") as file_handle:
        payload = pickle.load(file_handle, encoding="bytes")
    x_data = payload[b"data"]
    y_data = np.array(payload[b"labels"], dtype=np.int64)
    return x_data, y_data


def load_cifar10_train_test(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = data_dir / "CIFAR-10"
    if not root.exists():
        raise FileNotFoundError(f"Missing CIFAR-10 directory: {root}")

    x_train_all, y_train_all = [], []
    for batch_index in range(1, 6):
        x_batch, y_batch = _load_cifar_batch(root / f"data_batch_{batch_index}")
        x_train_all.append(x_batch)
        y_train_all.append(y_batch)

    x_train = np.vstack(x_train_all).astype(np.float32) / 255.0
    y_train = np.concatenate(y_train_all)
    x_test, y_test = _load_cifar_batch(root / "test_batch")
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, y_train, x_test, y_test


def load_cifar10c_corruption(
    data_dir: Path,
    corruption: str,
    severity: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 1 <= severity <= 5:
        raise ValueError("severity must be in {1,2,3,4,5}")

    root = data_dir / "CIFAR-10-C"
    if not root.exists():
        raise FileNotFoundError(f"Missing CIFAR-10-C directory: {root}")

    x_all = np.load(root / f"{corruption}.npy")
    y_all = np.load(root / "labels.npy")

    start_index = (severity - 1) * 10_000
    end_index = severity * 10_000
    x_subset = x_all[start_index:end_index]
    if x_subset.ndim == 4:
        x_subset = x_subset.reshape(len(x_subset), -1)
    x_subset = x_subset.astype(np.float32) / 255.0
    y_subset = y_all[start_index:end_index].astype(np.int64)
    return x_subset, y_subset


def make_mlp_classifier(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=800,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
        learning_rate_init=0.001,
    )


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[MLPClassifier, StandardScaler]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    classifier = make_mlp_classifier(seed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        classifier.fit(x_scaled, y_train)
    return classifier, scaler


def sanitize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.array(probs, dtype=np.float64, copy=True)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    row_sums = probs.sum(axis=1, keepdims=True)
    invalid_rows = ~np.isfinite(row_sums[:, 0]) | (row_sums[:, 0] <= 0.0)
    if np.any(invalid_rows):
        probs[invalid_rows] = 1.0 / probs.shape[1]
        row_sums = probs.sum(axis=1, keepdims=True)

    probs /= row_sums
    return probs.astype(np.float32)


def predict_proba_safe(
    classifier: MLPClassifier,
    x_data: np.ndarray,
    scaler: StandardScaler,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        probs = classifier.predict_proba(scaler.transform(x_data))
    return sanitize_probabilities(probs)


def randomized_classification_pit(
    probs: np.ndarray,
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard randomized PIT for classification (Eq. 1 in the paper).

    For each sample i with predicted class probabilities (p_1, ..., p_K)
    and true class y:
        U = sum_{j < y} p_j  +  V * p_y,   V ~ Uniform(0, 1)

    Under perfect calibration, U ~ Uniform(0, 1).
    No reference pool required.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    n = len(y)

    cumsum = np.cumsum(probs, axis=1)
    idx = np.arange(n)
    cdf_below = np.where(y > 0, cumsum[idx, y - 1], 0.0)
    p_true = probs[idx, y]

    v = rng.random(n)
    pits = cdf_below + v * p_true
    return np.clip(pits, 0.0, 1.0).astype(np.float32, copy=False)


def summarize_trials(rows: list[dict], n_trials: int) -> dict:
    n_alarm = sum(row["alarm_fired"] for row in rows)
    n_false = sum(row["false_alarm"] for row in rows)
    n_detect = sum(1 for row in rows if row["alarm_fired"] and not row["false_alarm"])
    delays = [
        row["detection_delay"] for row in rows if row["detection_delay"] is not None
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
        "evidences": [row["final_evidence"] for row in rows],
    }


def run_pitmonitor_trial(
    probs: np.ndarray,
    y_all: np.ndarray,
    alpha: float,
    n_bins: int,
    n_stable: int,
    pit_seed: int,
) -> dict:
    pits = randomized_classification_pit(
        probs,
        y_all,
        np.random.default_rng(pit_seed),
    )
    monitor = PITMonitor(alpha=alpha, n_bins=n_bins)
    monitor.update_many(pits, stop_on_alarm=True)
    return monitor.trial_summary(n_stable)


def make_baseline_detectors() -> dict[str, object]:
    return {
        "DDM": DDM(),
        "EDDM": EDDM(),
        "ADWIN": ADWIN(),
        "KSWIN": KSWIN(),
    }


def run_baselines_one_pass(
    error_stream: np.ndarray, n_stable: int
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
        at = alarm_time[name]
        false_alarm = at is not None and at <= n_stable
        delay = at - n_stable if at is not None and at > n_stable else None
        out[name] = BaselineRun(warning_time[name], at, false_alarm, delay)
    return out

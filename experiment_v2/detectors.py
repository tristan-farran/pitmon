"""Unified drift detector interface.

All detectors are wrapped so they expose a common API:
    detector.name     → str
    detector.feed(...)  → None  (processes the full monitoring stream)
    detector.result   → DetectorResult

This keeps the experiment loop clean and makes adding new detectors trivial.

Input conventions:
    - PITMonitor ← PIT values  (distributional)
    - ADWIN / KSWIN / PageHinkley ← squared residuals  (continuous)
    - DDM / EDDM / HDDM_A / HDDM_W ← binary errors  (thresholded)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, HDDM_A, HDDM_W

from pitmon import PITMonitor


# ─── Result container ────────────────────────────────────────────────


@dataclass
class DetectorResult:
    """Standardised output from any drift detector."""

    name: str
    alarm_fired: bool
    alarm_index: int | None  # index within the monitoring stream
    false_alarm: bool  # alarm before the true drift point
    detection_delay: int | None  # samples after drift; None if missed


# ─── Protocol for type-checking ──────────────────────────────────────


class Detector(Protocol):
    name: str

    def feed(
        self,
        pits: np.ndarray,
        sq_residuals: np.ndarray,
        binary_errors: np.ndarray,
        n_stable: int,
    ) -> None: ...

    @property
    def result(self) -> DetectorResult: ...


# ─── PITMonitor wrapper ─────────────────────────────────────────────


class PITMonitorDetector:
    """Wraps our PITMonitor in the common Detector interface."""

    def __init__(self, alpha: float = 0.05, n_bins: int = 10, seed: int = 42):
        self.name = "PITMonitor"
        self._alpha = alpha
        self._n_bins = n_bins
        self._seed = seed
        self._result: DetectorResult | None = None

    def feed(
        self,
        pits: np.ndarray,
        sq_residuals: np.ndarray,
        binary_errors: np.ndarray,
        n_stable: int,
    ) -> None:
        mon = PITMonitor(alpha=self._alpha, n_bins=self._n_bins, rng=self._seed)
        alarm_idx = None
        for i, pit in enumerate(pits):
            alarm = mon.update(float(pit))
            if alarm.triggered and alarm_idx is None:
                alarm_idx = i
                break  # stop on first alarm

        fired = alarm_idx is not None
        false_alarm = fired and alarm_idx < n_stable
        delay = (alarm_idx - n_stable) if (fired and not false_alarm) else None
        self._result = DetectorResult(
            name=self.name,
            alarm_fired=fired,
            alarm_index=alarm_idx,
            false_alarm=false_alarm,
            detection_delay=delay,
        )

    @property
    def result(self) -> DetectorResult:
        assert self._result is not None
        return self._result


# ─── Generic River detector wrapper ─────────────────────────────────


class RiverDetector:
    """Wraps any River drift detector.

    Parameters
    ----------
    name : str
        Human-readable name for reporting.
    detector_factory : callable
        Zero-argument callable that returns a fresh detector instance.
    input_kind : str
        One of "continuous" or "binary".
    """

    def __init__(self, name: str, detector_factory, input_kind: str):
        self.name = name
        self._factory = detector_factory
        self._input_kind = input_kind
        self._result: DetectorResult | None = None

    def feed(
        self,
        pits: np.ndarray,
        sq_residuals: np.ndarray,
        binary_errors: np.ndarray,
        n_stable: int,
    ) -> None:
        det = self._factory()
        stream = sq_residuals if self._input_kind == "continuous" else binary_errors
        alarm_idx = None

        for i, val in enumerate(stream):
            v = int(val) if self._input_kind == "binary" else float(val)
            det.update(v)
            if det.drift_detected:
                alarm_idx = i
                break  # first detection only

        fired = alarm_idx is not None
        false_alarm = fired and alarm_idx < n_stable
        delay = (alarm_idx - n_stable) if (fired and not false_alarm) else None
        self._result = DetectorResult(
            name=self.name,
            alarm_fired=fired,
            alarm_index=alarm_idx,
            false_alarm=false_alarm,
            detection_delay=delay,
        )

    @property
    def result(self) -> DetectorResult:
        assert self._result is not None
        return self._result


# ─── Factory: build all detectors ────────────────────────────────────


def build_all_detectors(
    alpha: float = 0.05,
    n_monitor_bins: int = 10,
    seed: int = 42,
) -> list[Detector]:
    """Instantiate one of every detector type for a single trial."""
    return [
        PITMonitorDetector(alpha=alpha, n_bins=n_monitor_bins, seed=seed),
        RiverDetector("ADWIN", ADWIN, "continuous"),
        RiverDetector("KSWIN", KSWIN, "continuous"),
        RiverDetector("PageHinkley", PageHinkley, "continuous"),
        RiverDetector("DDM", DDM, "binary"),
        RiverDetector("EDDM", EDDM, "binary"),
        RiverDetector("HDDM_A", HDDM_A, "binary"),
        RiverDetector("HDDM_W", HDDM_W, "binary"),
    ]


ALL_DETECTOR_NAMES = [
    "PITMonitor",
    "ADWIN",
    "KSWIN",
    "PageHinkley",
    "DDM",
    "EDDM",
    "HDDM_A",
    "HDDM_W",
]

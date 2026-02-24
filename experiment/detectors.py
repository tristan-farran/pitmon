"""Unified drift detector interface.

All detectors expose a common API:
    detector.name          → str
    detector.feed(...)     → None  (processes the full monitoring stream)
    detector.result        → DetectorResult

This design keeps the experiment loop clean and makes adding new detectors
trivial.

Input conventions
-----------------
Different detector families expect different input types:

    PITMonitor ← PIT values in [0, 1]  (distributional)
    ADWIN / KSWIN / PageHinkley  ← squared residuals  (continuous, non-negative)
    DDM / EDDM / HDDM_A / HDDM_W ← binary errors 0/1  (thresholded by median)

All three signal streams are computed once per trial by the experiment loop
and passed to every detector's ``feed`` method; each detector takes only the
stream relevant to its family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from river.drift import ADWIN, KSWIN, PageHinkley
from river.drift.binary import DDM, EDDM, HDDM_A, HDDM_W

from pitmon import PITMonitor


# ─── Result container ────────────────────────────────────────────────


@dataclass
class DetectorResult:
    """Standardised output from any drift detector after one trial.

    Attributes
    ----------
    name : str
        Detector identifier.
    alarm_fired : bool
        Whether the detector raised any alarm during the monitoring window.
    alarm_index : int or None
        0-based index within the monitoring stream at which the first alarm
        fired.  ``None`` if no alarm fired.
    false_alarm : bool
        ``True`` if the alarm fired before the true drift point (i.e. during
        the stable pre-drift window).
    detection_delay : int or None
        Number of samples between the true drift onset and the alarm.
        ``None`` if no true-positive detection occurred.
    """

    name: str
    alarm_fired: bool
    alarm_index: Optional[int]
    false_alarm: bool
    detection_delay: Optional[int]


# ─── PITMonitor wrapper ──────────────────────────────────────────────


class PITMonitorDetector:
    """Wraps PITMonitor in the common Detector interface.

    Parameters
    ----------
    alpha : float, default=0.05
        Anytime-valid false alarm level.
    n_bins : int, default=10
        Histogram bins for the e-value density estimator.
    seed : int, default=42
        RNG seed passed to PITMonitor for tie-randomized p-values.
    """

    def __init__(self, alpha: float = 0.05, n_bins: int = 10, seed: int = 42):
        self.name = "PITMonitor"
        self._alpha = alpha
        self._n_bins = n_bins
        self._seed = seed
        self._result: Optional[DetectorResult] = None

    def feed(
        self,
        pits: np.ndarray,
        sq_residuals: np.ndarray,
        binary_errors: np.ndarray,
        n_stable: int,
    ) -> None:
        """Process the full monitoring stream and store the result.

        Parameters
        ----------
        pits : ndarray, shape (n_monitor,)
            PIT values in [0, 1]; only this stream is consumed.
        sq_residuals : ndarray
            Ignored (provided for API uniformity).
        binary_errors : ndarray
            Ignored (provided for API uniformity).
        n_stable : int
            Number of pre-drift samples in the monitoring window.
        """
        mon = PITMonitor(alpha=self._alpha, n_bins=self._n_bins, rng=self._seed)
        alarm_idx: Optional[int] = None
        for i, pit in enumerate(pits):
            alarm = mon.update(float(pit))
            if alarm.triggered and alarm_idx is None:
                alarm_idx = i
                break

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
        """The result of the last ``feed`` call."""
        if self._result is None:
            raise RuntimeError("Call feed() before accessing result.")
        return self._result


# ─── Generic River detector wrapper ─────────────────────────────────


class RiverDetector:
    """Wraps any River drift detector in the common Detector interface.

    Parameters
    ----------
    name : str
        Human-readable name for reporting.
    detector_factory : callable
        Zero-argument callable that returns a fresh detector instance.
    input_kind : str
        One of ``'continuous'`` (uses squared residuals) or
        ``'binary'`` (uses binary error stream).
    """

    def __init__(self, name: str, detector_factory, input_kind: str):
        self.name = name
        self._factory = detector_factory
        self._input_kind = input_kind
        self._result: Optional[DetectorResult] = None

    def feed(
        self,
        pits: np.ndarray,
        sq_residuals: np.ndarray,
        binary_errors: np.ndarray,
        n_stable: int,
    ) -> None:
        """Process the full monitoring stream and store the result.

        Parameters
        ----------
        pits : ndarray
            Ignored (provided for API uniformity).
        sq_residuals : ndarray, shape (n_monitor,)
            Squared residuals; used when ``input_kind == 'continuous'``.
        binary_errors : ndarray, shape (n_monitor,)
            Binary error flags; used when ``input_kind == 'binary'``.
        n_stable : int
            Number of pre-drift samples in the monitoring window.
        """
        det = self._factory()
        stream = sq_residuals if self._input_kind == "continuous" else binary_errors
        alarm_idx: Optional[int] = None

        for i, val in enumerate(stream):
            v = int(val) if self._input_kind == "binary" else float(val)
            det.update(v)
            if det.drift_detected:
                alarm_idx = i
                break

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
        """The result of the last ``feed`` call."""
        if self._result is None:
            raise RuntimeError("Call feed() before accessing result.")
        return self._result


# ─── Factory ────────────────────────────────────────────────────────


def build_all_detectors(
    alpha: float = 0.05,
    n_monitor_bins: int = 10,
    seed: int = 42,
) -> list:
    """Instantiate one of every detector type for a single trial.

    Parameters
    ----------
    alpha : float, default=0.05
        False alarm level for PITMonitor.
    n_monitor_bins : int, default=10
        n_bins for PITMonitor's histogram density estimator.
    seed : int, default=42
        RNG seed for PITMonitor's tie-randomized p-values.

    Returns
    -------
    list of detector instances
    """
    return [
        PITMonitorDetector(alpha=alpha, n_bins=n_monitor_bins, seed=seed),
        RiverDetector("ADWIN", lambda: ADWIN(), "continuous"),
        RiverDetector("KSWIN", lambda: KSWIN(), "continuous"),
        RiverDetector("PageHinkley", lambda: PageHinkley(), "continuous"),
        RiverDetector("DDM", lambda: DDM(), "binary"),
        RiverDetector("EDDM", lambda: EDDM(), "binary"),
        RiverDetector("HDDM_A", lambda: HDDM_A(), "binary"),
        RiverDetector("HDDM_W", lambda: HDDM_W(), "binary"),
    ]


ALL_DETECTOR_NAMES: list[str] = [
    "PITMonitor",
    "ADWIN",
    "KSWIN",
    "PageHinkley",
    "DDM",
    "EDDM",
    "HDDM_A",
    "HDDM_W",
]

"""Pitmon demo: detect when a weather forecast model loses calibration."""

import numpy as np
from scipy.stats import norm
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pitmon import PITMonitor

np.random.seed(42)
monitor = PITMonitor(false_alarm_rate=0.05)

# Phase 1: Model is well-calibrated (100 observations)
print("Phase 1: Well-calibrated model (N(0,1) predictions, N(0,1) data)")
for _ in range(100):
    outcome = np.random.normal(0, 1)
    monitor.update(norm(0, 1).cdf, outcome)

print(f"  {monitor.t} observations, evidence = {monitor.evidence:.4f}, alarm = {monitor.alarm_triggered}")

# Phase 2: Climate shifts — model doesn't adapt
print("\nPhase 2: Climate shifts to N(2,1), model still predicts N(0,1)")
for _ in range(200):
    outcome = np.random.normal(2, 1)  # shifted climate
    alarm = monitor.update(norm(0, 1).cdf, outcome)  # stale model

    if alarm:
        cp = monitor.localize_changepoint()
        print(f"  ALARM at t={monitor.alarm_time} (true shift at t=100)")
        print(f"  Estimated changepoint: t={cp}")
        print(f"  Diagnosis: {alarm.diagnosis}")
        break

# Plot diagnostics
fig = monitor.plot()
fig.savefig("demo_diagnostics.png", dpi=150, bbox_inches="tight")
print(f"\nSaved diagnostic plot to demo_diagnostics.png")

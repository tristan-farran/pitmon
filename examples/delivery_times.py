"""
═══════════════════════════════════════════════════════════════════════════════
  PITMonitor Demo: Catching a Neural Network Going Stale
═══════════════════════════════════════════════════════════════════════════════

Scenario
--------
A neural network is deployed to predict delivery times for a logistics company.
It outputs full predictive distributions (mean + variance) for each order.

Everything works fine for 300 days — then a new highway opens, cutting transit
times in the region. The NN knows nothing about this. Its predictions are now
systematically too high, and its uncertainty bands are wrong.

PITMonitor watches the stream of Probability Integral Transforms. Under good
calibration, PITs are uniform. After the regime shift they pile up near 1.0
(actual deliveries are faster than predicted), and the monitor fires an alarm
— all with anytime-valid Type-I error control.

No retraining data, no labels on "what changed" — just a stream of (forecast
distribution, realized outcome) pairs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pitmon import PITMonitor

# ─── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ─── Simulation parameters ───────────────────────────────────────────────────
N_STABLE    = 300   # days of stable calibration
N_SHIFTED   = 200   # days after regime shift
N_TOTAL     = N_STABLE + N_SHIFTED
TRUE_CHANGE = N_STABLE + 1

# "Neural network" predictive distribution parameters (in hours)
NN_PRED_MEAN  = 48.0   # network always predicts ~48h delivery
NN_PRED_STD   = 6.0    # network's learned aleatoric uncertainty

# True data-generating process
TRUE_MEAN_BEFORE = 48.0  # reality matches the model
TRUE_STD_BEFORE  = 6.0

TRUE_MEAN_AFTER  = 38.0  # new highway cuts 10 hours off delivery
TRUE_STD_AFTER   = 5.0   # also slightly less variable

print(__doc__)

# ─── Generate "reality" ──────────────────────────────────────────────────────
y_before = np.random.normal(TRUE_MEAN_BEFORE, TRUE_STD_BEFORE, N_STABLE)
y_after  = np.random.normal(TRUE_MEAN_AFTER,  TRUE_STD_AFTER,  N_SHIFTED)
y_all    = np.concatenate([y_before, y_after])

# The neural network's predictive CDF for every observation
# (frozen — it was trained once and never updated)
nn_cdf = stats.norm(loc=NN_PRED_MEAN, scale=NN_PRED_STD).cdf

# ─── Run the monitor ─────────────────────────────────────────────────────────
monitor = PITMonitor(alpha=0.05)

evidence_trace = []
pit_trace = []

for i, y in enumerate(y_all, start=1):
    pit = nn_cdf(y)                # PIT = F_nn(y)
    pit = np.clip(pit, 0, 1)      # numerical safety
    alarm = monitor.update(pit)

    evidence_trace.append(alarm.evidence)
    pit_trace.append(pit)

    # Print milestones
    if i == N_STABLE:
        print(f"  [t={i:>4d}]  End of stable period.  Evidence = {alarm.evidence:.4f}")
    if alarm.triggered and i == alarm.time:
        print(f"  [t={i:>4d}]  *** ALARM TRIGGERED ***  Evidence = {alarm.evidence:.1f}"
              f"  (threshold = {alarm.threshold:.0f})")

# ─── Results ──────────────────────────────────────────────────────────────────
summary = monitor.summary()
est_cp  = summary["changepoint"]

print(f"\n{'─'*60}")
print(f"  Summary")
print(f"{'─'*60}")
print(f"  Observations processed :  {summary['t']}")
print(f"  Alarm triggered        :  {'Yes' if summary['alarm_triggered'] else 'No'}")
print(f"  Alarm time             :  t = {summary['alarm_time']}")
print(f"  True changepoint       :  t = {TRUE_CHANGE}")
print(f"  Estimated changepoint  :  t ≈ {est_cp}")
print(f"  Detection delay        :  {summary['alarm_time'] - TRUE_CHANGE} observations")
print(f"  Final evidence         :  {summary['evidence']:.1f}")
print(f"  KS calibration score   :  {summary['calibration_score']:.4f}")
print(f"{'─'*60}\n")

# ─── Visualization ────────────────────────────────────────────────────────────
times = np.arange(1, N_TOTAL + 1)
evidence = np.array(evidence_trace)
pits = np.array(pit_trace)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle(
    "PITMonitor: Detecting Neural Network Miscalibration After Regime Shift",
    fontsize=14, fontweight="bold", y=0.98,
)

# ── Panel 1: The raw data (what the NN sees) ─────────────────────────────────
ax = axes[0, 0]
ax.scatter(times, y_all, s=4, alpha=0.4, c="steelblue", label="Actual delivery (hours)")
ax.axhline(NN_PRED_MEAN, color="darkorange", lw=2, ls="--", label="NN predicted mean")
ax.fill_between(
    times,
    NN_PRED_MEAN - 2 * NN_PRED_STD,
    NN_PRED_MEAN + 2 * NN_PRED_STD,
    color="darkorange", alpha=0.12, label="NN ±2σ band",
)
ax.axvline(TRUE_CHANGE, color="red", ls=":", lw=1.5, alpha=0.7, label=f"True shift (t={TRUE_CHANGE})")
ax.set(xlabel="Day", ylabel="Delivery time (hours)", title="Observations vs. Stale NN Forecast")
ax.legend(fontsize=8, loc="lower left")
ax.grid(True, alpha=0.2)

# ── Panel 2: PIT values over time ────────────────────────────────────────────
ax = axes[0, 1]
ax.scatter(times, pits, s=4, alpha=0.4, c=np.where(times < TRUE_CHANGE, "steelblue", "crimson"))
ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)
ax.axvline(TRUE_CHANGE, color="red", ls=":", lw=1.5, alpha=0.7)
if monitor.alarm_time:
    ax.axvline(monitor.alarm_time, color="orange", ls=":", lw=1.5, alpha=0.7,
               label=f"Alarm (t={monitor.alarm_time})")
ax.set(xlabel="Day", ylabel="PIT value", title="PIT Stream (uniform → skewed)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 3: Evidence process (log scale) ────────────────────────────────────
ax = axes[1, 0]
ax.semilogy(times, np.maximum(evidence, 1e-10), color="steelblue", lw=1.5)
ax.axhline(monitor.threshold, color="crimson", ls="--", lw=2,
           label=f"Threshold (1/α = {monitor.threshold:.0f})")
ax.axvline(TRUE_CHANGE, color="red", ls=":", lw=1.5, alpha=0.7, label=f"True shift (t={TRUE_CHANGE})")
if monitor.alarm_time:
    ax.axvline(monitor.alarm_time, color="orange", ls=":", lw=2,
               label=f"Alarm (t={monitor.alarm_time})")
if est_cp:
    ax.axvline(est_cp, color="green", ls="--", lw=1.5, alpha=0.7,
               label=f"Est. changepoint (t≈{est_cp})")
ax.set(xlabel="Day", ylabel="Evidence (log scale)", title="E-Process: Evidence Against Stable Calibration")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 4: PIT histograms before / after ───────────────────────────────────
ax = axes[1, 1]
bins = np.linspace(0, 1, 21)
ax.hist(pits[:N_STABLE], bins=bins, density=True, alpha=0.5,
        color="steelblue", edgecolor="white", label="Before shift (calibrated)")
ax.hist(pits[N_STABLE:], bins=bins, density=True, alpha=0.5,
        color="crimson", edgecolor="white", label="After shift (miscalibrated)")
ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Ideal (Uniform)")
ax.set(xlabel="PIT value", ylabel="Density", title="PIT Distributions: Before vs. After")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("pitmon_demo.png", dpi=180, bbox_inches="tight")
print("  Plot saved → pitmon_demo.png")
plt.close()

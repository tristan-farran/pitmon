"""
═══════════════════════════════════════════════════════════════════════════════
  PITMonitor Demo: Catching a Neural Network Going Stale
═══════════════════════════════════════════════════════════════════════════════

Scenario
--------
A logistics company trains a neural network (sklearn MLPRegressor) to predict
package delivery times from features like distance, package weight, and route
complexity. The network is trained on 2,000 historical shipments where
conditions were stable.

After deployment, PITMonitor watches the NN's probabilistic forecasts against
reality. For the first 300 shipments, everything is fine — the model is well
calibrated. Then a new highway opens, systematically cutting delivery times in
the region. The NN knows nothing about this infrastructure change.

PITMonitor detects the resulting miscalibration and fires an alarm, telling
the team it's time to retrain.

Key detail: the NN is a real trained model, not a stub. We build its
predictive distribution using the standard approach of Gaussian residual
calibration — fitting the variance of its errors on a held-out calibration
set, so PIT = Φ((y − μ_nn(x)) / σ_cal(x)).
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from pitmon import PITMonitor

# ─── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
#  Step 1: Define the data-generating process
# ══════════════════════════════════════════════════════════════════════════════


def generate_features(n: int) -> np.ndarray:
    """
    Generate shipment features:
      x0: distance (km, log-normal)
      x1: package weight (kg)
      x2: route complexity score (0-1)
      x3: time-of-day factor (cyclic)
    """
    distance = np.random.lognormal(mean=4.5, sigma=0.6, size=n)  # ~90km median
    weight = np.random.exponential(scale=5, size=n) + 0.5
    complexity = np.random.beta(2, 5, size=n)
    time_of_day = np.random.uniform(0, 2 * np.pi, size=n)
    return np.column_stack([distance, weight, complexity, time_of_day])


def true_delivery_time(X: np.ndarray, regime: str = "before") -> np.ndarray:
    """
    Ground truth: delivery time in hours. Nonlinear function of features
    with heteroscedastic noise.
    """
    distance, weight, complexity, tod = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

    # Base time: nonlinear in distance and weight
    base = 8 + 0.15 * distance + 0.08 * distance * complexity + 0.3 * weight

    # Time-of-day effect (rush hour penalty)
    rush_hour = 3.0 * np.exp(-((tod - 2.5) ** 2) / 0.5)

    # Heteroscedastic noise (more variance at longer distances)
    noise_std = 1.5 + 0.02 * distance
    noise = np.random.normal(0, noise_std)

    if regime == "after":
        # New highway: cuts ~30% off base time, biggest effect on longer routes
        highway_benefit = 0.30 * base * (1 / (1 + np.exp(-0.03 * (distance - 50))))
        # Also reduces variance (more predictable routes)
        return base + rush_hour - highway_benefit + noise * 0.7
    else:
        return base + rush_hour + noise


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: Train the neural network
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Step 1: Training the neural network on historical data")
print("=" * 65)

# Training data (all from "before" regime)
N_TRAIN = 2000
N_CAL = 500  # held-out calibration set for variance estimation

X_train = generate_features(N_TRAIN)
y_train = true_delivery_time(X_train, regime="before")

X_cal = generate_features(N_CAL)
y_cal = true_delivery_time(X_cal, regime="before")

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_cal_s = scaler.transform(X_cal)

# Train the MLP — a genuine multi-layer neural network
nn = MLPRegressor(
    hidden_layer_sizes=(64, 32, 16),
    activation="relu",
    solver="adam",
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
    learning_rate_init=0.001,
)
nn.fit(X_train_s, y_train)

train_score = nn.score(X_train_s, y_train)
cal_preds = nn.predict(X_cal_s)
cal_residuals = y_cal - cal_preds
cal_rmse = np.sqrt(np.mean(cal_residuals**2))

print(f"  Architecture     : {nn.hidden_layer_sizes}")
print(f"  Training R²      : {train_score:.4f}")
print(f"  Calibration RMSE : {cal_rmse:.2f} hours")
print(f"  Calibration bias : {np.mean(cal_residuals):+.2f} hours")

# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Build the predictive distribution via residual calibration
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard approach for turning a point-prediction NN into a probabilistic
# forecaster: assume y | x ~ Normal(μ_nn(x), σ²) where σ is estimated from
# calibration residuals. We use local variance estimation binned by predicted
# value to capture heteroscedasticity.

print(f"\n{'=' * 65}")
print("  Step 2: Building predictive distribution (residual calibration)")
print("=" * 65)

# Bin calibration predictions and estimate per-bin variance
N_VAR_BINS = 5
pred_quantiles = np.percentile(cal_preds, np.linspace(0, 100, N_VAR_BINS + 1))
bin_stds = []
for i in range(N_VAR_BINS):
    mask = (cal_preds >= pred_quantiles[i]) & (cal_preds < pred_quantiles[i + 1])
    if mask.sum() < 10:
        mask = cal_preds >= pred_quantiles[i]  # fallback
    bin_stds.append(np.std(cal_residuals[mask]))

bin_stds = np.array(bin_stds)
print(f"  Variance bins    : {N_VAR_BINS}")
print(f"  Bin std range    : [{bin_stds.min():.2f}, {bin_stds.max():.2f}] hours")


def predictive_cdf(x: np.ndarray, y: float) -> float:
    """
    Compute PIT = P(Y ≤ y | x) using the NN's predictive distribution.
    """
    x_s = scaler.transform(x.reshape(1, -1))
    mu = nn.predict(x_s)[0]

    # Find which variance bin this prediction falls into
    bin_idx = np.searchsorted(pred_quantiles[1:-1], mu)
    bin_idx = np.clip(bin_idx, 0, N_VAR_BINS - 1)
    sigma = bin_stds[bin_idx]

    return float(stats.norm.cdf(y, loc=mu, scale=sigma))


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3: Verify calibration on held-out "before" data
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("  Step 3: Verifying calibration on held-out pre-shift data")
print("=" * 65)

X_verify = generate_features(200)
y_verify = true_delivery_time(X_verify, regime="before")
pits_verify = np.array([predictive_cdf(X_verify[i], y_verify[i]) for i in range(200)])

ks_stat, ks_pval = stats.kstest(pits_verify, "uniform")
print(f"  KS statistic     : {ks_stat:.4f}")
print(f"  KS p-value       : {ks_pval:.4f}")
print(f"  Calibration      : {'PASS' if ks_pval > 0.05 else 'FAIL'} (p > 0.05)")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4: Deploy with PITMonitor — stable, then regime shift
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 65}")
print("  Step 4: Deploying with PITMonitor (300 stable → 200 shifted)")
print("=" * 65)

N_STABLE = 300
N_SHIFTED = 200
N_TOTAL = N_STABLE + N_SHIFTED
TRUE_CHANGE = N_STABLE + 1

# Generate deployment stream
X_stable = generate_features(N_STABLE)
y_stable = true_delivery_time(X_stable, regime="before")

X_shifted = generate_features(N_SHIFTED)
y_shifted = true_delivery_time(X_shifted, regime="after")

X_all = np.vstack([X_stable, X_shifted])
y_all = np.concatenate([y_stable, y_shifted])

# Run the monitor
monitor = PITMonitor(alpha=0.05)

evidence_trace = []
pit_trace = []
pred_trace = []

for i in range(N_TOTAL):
    x_i, y_i = X_all[i], y_all[i]

    # NN prediction (for plotting)
    x_s = scaler.transform(x_i.reshape(1, -1))
    mu_i = nn.predict(x_s)[0]
    pred_trace.append(mu_i)

    # Compute PIT from the NN's full predictive distribution
    pit = predictive_cdf(x_i, y_i)
    pit = np.clip(pit, 0, 1)
    alarm = monitor.update(pit)

    evidence_trace.append(alarm.evidence)
    pit_trace.append(pit)

    t = i + 1
    if t == N_STABLE:
        print(f"  [t={t:>4d}]  End of stable period.  Evidence = {alarm.evidence:.4f}")
    if alarm.triggered and t == alarm.time:
        print(
            f"  [t={t:>4d}]  *** ALARM TRIGGERED ***  "
            f"Evidence = {alarm.evidence:.1f}  (threshold = {alarm.threshold:.0f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5: Results
# ══════════════════════════════════════════════════════════════════════════════

summary = monitor.summary()
est_cp = summary["changepoint"]

print(f"\n{'─' * 65}")
print(f"  Results")
print(f"{'─' * 65}")
print(
    f"  Neural network       :  MLPRegressor{nn.hidden_layer_sizes}, R²={train_score:.3f}"
)
print(f"  Observations         :  {summary['t']}")
print(f"  Alarm triggered      :  {'Yes' if summary['alarm_triggered'] else 'No'}")
if summary["alarm_triggered"]:
    print(f"  Alarm time           :  t = {summary['alarm_time']}")
    print(f"  True changepoint     :  t = {TRUE_CHANGE}")
    print(f"  Estimated changepoint:  t ≈ {est_cp}")
    print(
        f"  Detection delay      :  {summary['alarm_time'] - TRUE_CHANGE} observations"
    )
print(f"  Final evidence       :  {summary['evidence']:.1f}")
print(f"  KS calibration score :  {summary['calibration_score']:.4f}")
print(f"{'─' * 65}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 6: Visualization
# ══════════════════════════════════════════════════════════════════════════════

times = np.arange(1, N_TOTAL + 1)
evidence = np.array(evidence_trace)
pits = np.array(pit_trace)
preds = np.array(pred_trace)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "PITMonitor: Detecting Miscalibration in a Deployed Neural Network",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# ── Panel 1: NN predictions vs reality ────────────────────────────────────────
ax = axes[0, 0]
ax.scatter(
    times, y_all, s=6, alpha=0.35, c="steelblue", label="Actual delivery", zorder=2
)
ax.scatter(
    times, preds, s=6, alpha=0.35, c="darkorange", label="NN prediction", zorder=2
)
ax.axvline(
    TRUE_CHANGE,
    color="red",
    ls=":",
    lw=1.5,
    alpha=0.8,
    label=f"Regime shift (t={TRUE_CHANGE})",
)
if monitor.alarm_time:
    ax.axvline(
        monitor.alarm_time,
        color="orange",
        ls="--",
        lw=1.5,
        label=f"Alarm (t={monitor.alarm_time})",
    )

# Add rolling residual to show divergence
window = 20
residuals = y_all - preds
rolling_resid = np.convolve(residuals, np.ones(window) / window, mode="valid")
ax.plot(
    np.arange(window, N_TOTAL + 1),
    rolling_resid + np.mean(y_all),
    color="crimson",
    lw=1.5,
    alpha=0.6,
    label=f"Rolling residual (w={window})",
)

ax.set(
    xlabel="Shipment",
    ylabel="Delivery time (hours)",
    title="NN Predictions vs. Reality",
)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax.grid(True, alpha=0.2)

# ── Panel 2: PIT stream ──────────────────────────────────────────────────────
ax = axes[0, 1]
colors = np.where(times <= N_STABLE, "steelblue", "crimson")
ax.scatter(times, pits, s=5, alpha=0.4, c=colors)
ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.4)
ax.axvline(TRUE_CHANGE, color="red", ls=":", lw=1.5, alpha=0.8)
if monitor.alarm_time:
    ax.axvline(
        monitor.alarm_time,
        color="orange",
        ls="--",
        lw=1.5,
        label=f"Alarm (t={monitor.alarm_time})",
    )

# Rolling mean of PITs to highlight the drift
rolling_pit = np.convolve(pits, np.ones(30) / 30, mode="valid")
ax.plot(
    np.arange(30, N_TOTAL + 1),
    rolling_pit,
    color="black",
    lw=1.5,
    alpha=0.6,
    label="Rolling mean (w=30)",
)

ax.set(
    xlabel="Shipment",
    ylabel="PIT value",
    title="PIT Stream — Uniform Before, Skewed After",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 3: Evidence process ─────────────────────────────────────────────────
ax = axes[1, 0]
ax.semilogy(times, np.maximum(evidence, 1e-10), color="steelblue", lw=1.5)
ax.axhline(
    monitor.threshold,
    color="crimson",
    ls="--",
    lw=2,
    label=f"Threshold (1/α = {monitor.threshold:.0f})",
)
ax.axvline(
    TRUE_CHANGE,
    color="red",
    ls=":",
    lw=1.5,
    alpha=0.8,
    label=f"Regime shift (t={TRUE_CHANGE})",
)
if monitor.alarm_time:
    ax.axvline(
        monitor.alarm_time,
        color="orange",
        ls="--",
        lw=2,
        label=f"Alarm (t={monitor.alarm_time})",
    )
if est_cp:
    ax.axvline(
        est_cp,
        color="green",
        ls="--",
        lw=1.5,
        alpha=0.7,
        label=f"Est. changepoint (t≈{est_cp})",
    )
ax.set(
    xlabel="Shipment",
    ylabel="Evidence (log scale)",
    title="E-Process: Evidence Against Stable Calibration",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

# ── Panel 4: PIT histograms ──────────────────────────────────────────────────
ax = axes[1, 1]
hist_bins = np.linspace(0, 1, 21)
ax.hist(
    pits[:N_STABLE],
    bins=hist_bins,
    density=True,
    alpha=0.5,
    color="steelblue",
    edgecolor="white",
    label="Pre-shift (calibrated)",
)
ax.hist(
    pits[N_STABLE:],
    bins=hist_bins,
    density=True,
    alpha=0.5,
    color="crimson",
    edgecolor="white",
    label="Post-shift (miscalibrated)",
)
ax.axhline(1.0, color="black", ls="--", lw=1.5, label="Ideal (Uniform)")
ax.set(
    xlabel="PIT value",
    ylabel="Density",
    title="PIT Distributions: Before vs. After the Highway Opens",
)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("pitmon_nn_demo.png", dpi=180, bbox_inches="tight")
print("  Plot saved → pitmon_nn_demo.png")
plt.close()

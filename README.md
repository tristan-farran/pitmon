# Pitmon

An anytime-valid monitor for Probability Integral Transform (PIT) values. It detects changes in calibration without needing a baseline period and controls the false alarm probability over the entire monitoring horizon. A stable but miscalibrated model will not trigger alarms, only *changes* in calibration are detected.

Installation
------------
Base library:
```bash
pip install -e .
```

Quick start
-----------
Monitor a stream of predictions and outcomes:

```python
from pitmon import PITMonitor

monitor = PITMonitor(alpha=0.05)

for prediction, outcome in data_stream:
    alarm = monitor.update(prediction.cdf(outcome))
    if alarm:
        print(f"Alarm at t={monitor.t}")
        print(f"Changepoint estimate: {monitor.changepoint()}")
        break
```

How it works (short)
--------------------
Each PIT is ranked among all previous PITs. Under exchangeability, these ranks
are uniform. Pitmon converts each rank into an e-value and tracks a mixture
e-process that accumulates evidence against exchangeability:
- Under exchangeability: p_t ~ Uniform(0,1), E[e_t] = 1, M_t is a supermartingale
- Under change: p_t concentrates, e_t > 1 on average, M_t grows exponentially
- Ville's inequality constrains the false alarm rate: P(sup M_t ≥ 1/α | H₀) ≤ α

Experiment
----------
The experiment compares PITMonitor against all seven drift detectors in
the [River](https://riverml.xyz) library on the standard FriedmanDrift
regression benchmark. The goal is to evaluate PITMonitor's ability to detect
calibration drift with anytime-valid false-alarm guarantees.

Three FriedmanDrift variants are tested:
- GRA (Global Recurring Abrupt) — sudden change over all features
- GSG (Global Slow Gradual) — smooth transition over 500 samples
- LEA (Local Expanding Abrupt) — partial, expanding drift regions

For each scenario × trial:
1. Data — Generate a FriedmanDrift stream (10 features, regression target)
2. Train — Fit a `ProbabilisticMLP` on the pre-drift segment
3. Monitor — Feed the monitoring segment to all detectors:
   - PITMonitor receives PITs: `Φ((y − ŷ) / σ̂)`
   - Continuous detectors receive squared residuals: `(y − ŷ)²`
   - Binary detectors receive thresholded errors: `1{|y − ŷ| > median}`
4. Record — Whether alarm fired, when, and if it was a true or false detection

Metrics:
- TPR (True Positive Rate) — fraction of trials where drift was correctly detected
- FPR (False Positive Rate) — fraction of trials with a false alarm (before drift)
- Detection Delay — samples between drift onset and alarm
- All rates reported with Wilson score 95% confidence intervals
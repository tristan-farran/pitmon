# PITMonitor

An anytime-valid monitor for Probability Integral Transform (PIT) values. Detects changes in model calibration without needing a baseline period, controlling the false alarm probability over the entire monitoring horizon. A stable but miscalibrated model will not trigger alarms — only *changes* in calibration are detected.

## Installation

```bash
pip install -e .
```

## Quick start

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

## How it works

Each PIT is ranked among all previous PITs.  Under exchangeability the ranks
are uniform.  PITMonitor converts each rank into an e-value and tracks a
mixture e-process that accumulates evidence against exchangeability:

- **Under exchangeability:** p_t ~ Uniform(0,1), E[e_t] = 1, M_t is a supermartingale
- **Under change:** p_t concentrates, e_t > 1 on average, M_t grows exponentially
- **Ville's inequality** constrains the false alarm rate: P(sup M_t ≥ 1/α | H₀) ≤ α

---

## Experiment

Compares PITMonitor against the seven drift detectors available in [River](https://riverml.xyz) on the standard **FriedmanDrift** regression benchmark.

The `ProbabilisticMLP` model outputs a Gaussian predictive distribution (mean + log-variance) trained with Gaussian NLL loss.  Inputs and targets are standardized before training. The model is trained **once** on one realisation of the pre-drift distribution and shared across all Monte-Carlo trials - correctly simulating a fixed deployed model whose calibration is being monitored. 

Per-trial flow:
1. **Data** – Generate a fresh monitoring stream (new seed; model is fixed)
2. **Signals** – Compute from the fixed model:
   - PITs:  `Φ((y − ŷ) / σ̂)`  ← fed to PITMonitor
   - Squared residuals: `(y − ŷ)²`  ← fed to ADWIN / KSWIN / PageHinkley
   - Binary errors: `1{|y − ŷ| > median}`  ← fed to DDM / EDDM / HDDM_A / HDDM_W
3. **Detect** – All detectors process the monitoring window
4. **Record** – Alarm time, true/false positive status, detection delay

Drift scenarios:
| Key         | Type                    | Description                                  |
| ----------- | ----------------------- | -------------------------------------------- |
| `gra_tw0`   | Global Recurring Abrupt | All relevant features change simultaneously  |
| `gsg_tw500` | Global Slow Gradual     | Smooth transition over 500 samples           |
| `lea_tw0`   | Local Expanding Abrupt  | Drift starts on a feature subset and expands |
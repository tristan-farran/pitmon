# PITMonitor

An anytime-valid monitor for Probability Integral Transform (PIT) values.
Detects changes in model calibration without needing a baseline period,
controlling the false alarm probability over the entire monitoring horizon.
A stable but miscalibrated model will not trigger alarms — only *changes*
in calibration are detected.

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

The number of histogram bins (`n_bins`) controls the density estimator's
resolution.  Fewer bins are coarser but more stable early in the stream;
more bins offer finer discrimination once sufficient data has arrived.  The
default of 10 bins is an MDL-reasonable choice; see the sensitivity experiment.

---

## Experiment

Compares PITMonitor against the seven drift detectors available in
[River](https://riverml.xyz) on the standard **FriedmanDrift** regression
benchmark.

### Setup

```
experiment/
├── config.py        – All hyperparameters
├── data.py          – FriedmanDrift stream generation
├── model.py         – ProbabilisticMLP + normalization + save/load
├── train_model.py   – Standalone training script (run once)
├── experiment.py    – MC trial loop, aggregation, n_bins sweep
├── detectors.py     – Unified wrapper for PITMonitor and River detectors
├── plots.py         – All figures
└── run.py           – Entry point
```

### Workflow

**1. Train the model once** (saves `out/model.pkl`):

```bash
cd experiment
python run.py --train
```

**2. Run the experiment and generate plots**:

```bash
python run.py --compute --plot
```

**3. Run the n_bins sensitivity sweep** (optional):

```bash
python run.py --compute --plot --bins-sweep
```

**4. Quick smoke-test** with 50 trials:

```bash
python run.py --train --compute --plot --trials 50 --workers 4
```

> If none of `--train`, `--compute`, `--plot` are specified, all three run.

### Model

`ProbabilisticMLP` outputs a Gaussian predictive distribution (mean + log-variance)
trained with Gaussian NLL loss.  Inputs and targets are standardized before
training; the scaler statistics are bundled with the weights so inference is
consistent.  The model is trained **once** on one realisation of the pre-drift
distribution and shared across all Monte-Carlo trials — correctly simulating a
fixed deployed model whose calibration is being monitored.

### Drift scenarios

| Key         | Type                    | Description                                  |
| ----------- | ----------------------- | -------------------------------------------- |
| `gra_tw0`   | Global Recurring Abrupt | All relevant features change simultaneously  |
| `gsg_tw500` | Global Slow Gradual     | Smooth transition over 500 samples           |
| `lea_tw0`   | Local Expanding Abrupt  | Drift starts on a feature subset and expands |

### Per-trial flow

1. **Data** – Generate a fresh monitoring stream (new seed; model is fixed)
2. **Signals** – Compute from the fixed model:
   - PITs:  `Φ((y − ŷ) / σ̂)`  ← fed to PITMonitor
   - Squared residuals: `(y − ŷ)²`  ← fed to ADWIN / KSWIN / PageHinkley
   - Binary errors: `1{|y − ŷ| > median}`  ← fed to DDM / EDDM / HDDM_A / HDDM_W
3. **Detect** – All detectors process the monitoring window
4. **Record** – Alarm time, true/false positive status, detection delay

### Metrics

| Metric          | Description                                                                |
| --------------- | -------------------------------------------------------------------------- |
| TPR             | True Positive Rate — fraction of trials with a correct detection           |
| FPR             | False Positive Rate — fraction of trials with a false alarm (before drift) |
| Detection Delay | Samples between drift onset and alarm (true positives only)                |

All rates are reported with **Wilson score 95% confidence intervals**.

### Output files

| File                              | Description                               |
| --------------------------------- | ----------------------------------------- |
| `out/model.pkl`                   | Trained `ModelBundle` (weights + scalers) |
| `out/results.json`                | Full MC results                           |
| `out/bins_sweep.json`             | n_bins sweep results (if run)             |
| `out/fig_detection_rates.png`     | TPR / FPR horizontal bar charts           |
| `out/fig_delay_distributions.png` | Detection delay box plots                 |
| `out/fig_summary_table.png`       | Colour-coded summary table                |
| `out/fig_single_run_<scen>.png`   | 4-panel single-run diagnostic             |
| `out/fig_nbins_sweep.png`         | n_bins sensitivity (if run)               |

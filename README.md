# PITMonitor

An anytime-valid monitor for Probability Integral Transform (PIT) values. Detects and locates changes in model calibration without needing a baseline period, controlling the false alarm probability over the entire monitoring horizon. A stable but miscalibrated model will not trigger alarms — only *changes* in calibration are detected.

## Installation

```bash
pip install -e .
```

**Dependencies:** `numpy`, `sortedcontainers`, `matplotlib`

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

Or use the convenience wrapper if your model exposes a CDF:

```python
from scipy.stats import norm

monitor = PITMonitor(alpha=0.05)
alarm = monitor.update_with_cdf(norm(loc=mu, scale=sigma).cdf, y_observed)
```

## API

### `PITMonitor(alpha=0.05, n_bins=100, weight_schedule=None, rng=None)`

| Parameter         | Description                                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `alpha`           | Anytime-valid false alarm rate: P(ever alarm \| H₀) ≤ α                                                                          |
| `n_bins`          | Histogram bins for density estimation (5–500)                                                                                    |
| `weight_schedule` | Custom mixture weight schedule over changepoint indices; must be deterministic, nonneg, and sum to 1. Default: w(k) = 1/(k(k+1)) |
| `rng`             | Seed or `numpy.random.Generator` for tie-breaking randomization                                                                  |

### Core methods

| Method                                  | Description                                                                                                 |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `update(pit)`                           | Process one PIT value; returns `Alarm` (usable as bool)                                                     |
| `update_with_cdf(cdf, y)`               | Compute PIT from CDF and observe `y`, then update                                                           |
| `update_many(pits, stop_on_alarm=True)` | Process a sequence of PITs                                                                                  |
| `changepoint()`                         | Bayes-factor estimate of changepoint index; `None` if no alarm yet                                          |
| `summary()`                             | Dict with `t`, `alarm_triggered`, `alarm_time`, `evidence`, `threshold`, `changepoint`, `calibration_score` |
| `trial_summary(n_stable)`               | Convenience diagnostics for a stable-then-shift stream                                                      |
| `calibration_score()`                   | 1 − KS statistic measuring deviation from uniformity (1 = perfect)                                          |
| `reset()`                               | Reset to initial state (preserves `alpha`, `n_bins`)                                                        |
| `plot(figsize=(12,4))`                  | Diagnostic plot of e-process and p-value histogram; returns `PlotResult`                                    |
| `save(filepath)`                        | Save state to `.pkl` (full fidelity) or `.json` (human-readable)                                            |
| `PITMonitor.load(filepath)`             | Class method; restore a saved monitor                                                                       |

### `Alarm` object

Returned by every `update` call. Evaluates as `bool`.

```python
alarm.triggered   # bool
alarm.time        # int — current time step
alarm.evidence    # float — current e-process value M_t
alarm.threshold   # float — 1/α
```

## How it works

Each PIT is ranked among all previous PITs. Under exchangeability the ranks are uniform. PITMonitor converts each rank into an e-value and tracks a mixture e-process that accumulates evidence against exchangeability:

- **Under exchangeability:** p_t ~ Uniform(0,1), E[e_t] = 1, M_t is a supermartingale
- **Under change:** p_t concentrates, e_t > 1 on average, M_t grows exponentially
- **Ville's inequality** constrains the false alarm rate: P(sup M_t ≥ 1/α | H₀) ≤ α

An alarm fires when M_t ≥ 1/α. After an alarm, the evidence is frozen and `changepoint()` estimates the onset via a log Bayes factor scan over candidate split points.

## Persistence

```python
monitor.save("state.json")          # JSON (human-readable)
monitor.save("state.pkl")           # pickle (full precision)
monitor2 = PITMonitor.load("state.json")
```

## Running tests

```bash
pytest tests/
```

---

## Experiment

Compares PITMonitor against the seven drift detectors available in [River](https://riverml.xyz) on the standard **FriedmanDrift** regression benchmark.

The `ProbabilisticMLP` model outputs a Gaussian predictive distribution (mean & log-variance) trained with Gaussian NLL loss. Inputs and targets are standardized before training. The model is trained **once** on one realisation of the pre-drift distribution and shared across all Monte-Carlo trials — correctly simulating a fixed deployed model whose calibration is being monitored.

Per-trial flow:
1. **Data** — Generate a fresh monitoring stream (new seed; model is fixed)
2. **Signals** — Compute from the fixed model:
   - PITs: `Φ((y − ŷ) / σ̂)` ← fed to PITMonitor
   - Squared residuals: `(y − ŷ)²` ← fed to ADWIN / KSWIN / PageHinkley
   - Binary errors: `1{|y − ŷ| > median}` ← fed to DDM / EDDM / HDDM_A / HDDM_W
3. **Detect** — All detectors process the monitoring window
4. **Record** — Alarm time, true/false positive status, detection delay

### Drift scenarios

| Key         | Type                    | Description                                                                |
| ----------- | ----------------------- | -------------------------------------------------------------------------- |
| `gra_tw0`   | Global Recurring Abrupt | All relevant features change simultaneously                                |
| `gsg_tw500` | Global Slow Gradual     | Smooth transition over 500 samples                                         |
| `lea_tw0`   | Local Expanding Abrupt  | Drift starts on a feature subset and expands across 3 evenly spaced phases |

### Reproducing results

```bash
cd experiment/core

# 1. Train the model once
python run_experiment.py --train

# 2. Run full experiment and generate plots
python run_experiment.py --compute --plot

# 3. Rerun a single scenario and merge into existing results
python run_experiment.py --compute --scenario lea --plot

# 4. Quick smoke-test (50 trials)
python run_experiment.py --train --compute --plot --trials 50 --workers 4
```

All artefacts (model bundle, results, figures) are written to `experiment/core/out/` by default. Override with `--output DIR`.

### Additional experiments

Focused, lightweight checks that complement the main benchmark by verifying theoretical properties and robustness under assumption violations:

- **Proposition verification** — asymptotic convergence of E[e_t | A], finite-time mean formula, and warmup bound under adverse prior alignment
- **Multi-step drift localization** — changepoint error reported to both onset and maximal-intensity phase

```bash
cd experiment/additional

# Run everything (compute + plot)
python run_additional.py

# Compute only, with a fast smoke-test profile
python run_additional.py --compute --profile quick

# Regenerate plots from existing results
python run_additional.py --plot

# Publication-grade precision
python run_additional.py --compute --plot --profile publication
```

All artefacts (results, figures, LaTeX macros) are written to `experiment/additional/out/` by default. Override with `--output DIR`.

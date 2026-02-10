# Getting Started with PIT Monitor

## Quick Installation

```bash
# Navigate to the pit_monitor directory
cd pit_monitor

# Install dependencies
pip install numpy scipy matplotlib

# Optional: Install in development mode
pip install -e .
```

## 5-Minute Tutorial

### Step 1: Import and Initialize

```python
from monitor import PITMonitor
from scipy.stats import norm
import numpy as np

# Create monitor (only parameter: false alarm tolerance)
monitor = PITMonitor(false_alarm_rate=0.05)
```

### Step 2: Process Observations

```python
# Your probabilistic model
predicted_distribution = norm(loc=0, scale=1)

# Each time you observe an outcome
for observation in observations:
    alarm = monitor.update(predicted_distribution.cdf, observation)
    
    if alarm:
        print(f"Model broke at time {monitor.t}")
        break
```

### Step 3: Interpret Results

```python
if alarm:
    # Get diagnostic information
    print(f"Diagnosis: {alarm.diagnosis}")
    
    # Find when it started
    changepoint = monitor.localize_changepoint()
    print(f"Problem started around time {changepoint}")
    
    # Visualize
    monitor.plot_diagnostics()
```

## Complete Working Example

```python
import numpy as np
from scipy.stats import norm
from monitor import PITMonitor

# Simulate scenario: model becomes wrong
np.random.seed(42)
monitor = PITMonitor(false_alarm_rate=0.05)

# Phase 1: Model is correct (100 observations)
correct_model = norm(0, 1)
correct_data = norm(0, 1).rvs(size=100)

for obs in correct_data:
    alarm = monitor.update(correct_model.cdf, obs)
    if alarm:
        print("Unexpected alarm!")

print(f"Phase 1: No alarm in {monitor.t} observations ✓")

# Phase 2: Data changes, model doesn't (regime shift)
wrong_data = norm(3, 1).rvs(size=100)  # Mean shifted to 3

for obs in wrong_data:
    alarm = monitor.update(correct_model.cdf, obs)  # Still using old model
    
    if alarm:
        print(f"\nAlarm at observation {monitor.t}!")
        print(f"Diagnosis: {alarm.diagnosis}")
        
        cp = monitor.localize_changepoint()
        print(f"Estimated changepoint: {cp} (true: 100)")
        
        # Visualize what happened
        monitor.plot_diagnostics()
        break
```

## What You Need

### Required

- **Probabilistic model**: Any model that provides a CDF function
  - scipy distributions: `norm(0,1).cdf`
  - Custom: `lambda x: your_cdf_function(x)`

- **Observations**: Actual outcomes to compare against predictions

### Not Required

- ❌ Training data
- ❌ Loss function
- ❌ Alternative model
- ❌ Window size or binning parameters
- ❌ Any domain knowledge

## Common Use Cases

### Weather Forecasting
```python
# Daily temperature forecasts
for day, (forecast_dist, observed_temp) in enumerate(daily_data):
    alarm = monitor.update(forecast_dist.cdf, observed_temp)
```

### Financial Risk Models
```python
# VaR model validation
for day, (var_model, return_) in enumerate(returns):
    alarm = monitor.update(var_model.cdf, return_)
```

### Medical Predictions
```python
# Risk score validation
for patient, (risk_dist, outcome) in enumerate(patient_data):
    alarm = monitor.update(risk_dist.cdf, outcome)
```

## Understanding Outputs

### AlarmInfo Object

When `update()` is called, it returns an `AlarmInfo` object:

```python
alarm = monitor.update(pred.cdf, obs)

# Check if alarm triggered
if alarm:  # or: if alarm.triggered:
    # Access information
    alarm.alarm_time        # When it happened
    alarm.ks_distance       # How bad the deviation is
    alarm.threshold         # What the threshold was
    alarm.diagnosis         # Human-readable explanation
```

### Diagnosis Interpretation

The diagnosis tells you *how* the model is wrong:

- **"lower tail - overconfident"**: Model underestimates extreme low values
- **"upper tail - underconfident"**: Model overestimates extreme high values  
- **"central region - overconfident"**: Model is too certain (too narrow)
- **"central region - underconfident"**: Model is too uncertain (too wide)

## Next Steps

1. **Run the tests**: `cd tests && python run_tests.py`
2. **Try the examples**: `cd examples && python example_weather.py`
3. **Read the README**: Full documentation in `README.md`
4. **Customize**: Adjust `false_alarm_rate` based on your tolerance

## Troubleshooting

### "PIT value outside [0,1]"
- Your `predicted_cdf` function is not a valid CDF
- Check that it returns values in [0, 1]
- Check that it's monotonically increasing

### No alarm when you expect one
- Increase sample size (some deviations need more data to detect)
- Check if deviation is actually systematic vs random noise
- Try visualizing: `monitor.plot_diagnostics()`

### Too many false alarms
- Increase `false_alarm_rate` (e.g., 0.05 → 0.10)
- Or: Your model might actually have problems!

## Philosophy

Remember the core idea:

> **A model is valid exactly as long as its PIT sequence remains indistinguishable from Uniform(0,1)**

Everything else is just:
- Making this practical (sequential thresholds)
- Making it interpretable (diagnostics)  
- Making it rigorous (statistical guarantees)

The method is simple. The math ensures it works correctly. That's it.

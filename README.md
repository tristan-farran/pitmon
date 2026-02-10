# PIT Monitor

**Model-agnostic sequential validation via Probability Integral Transform**

A simple, principled tool for monitoring whether probabilistic models remain valid over time.

## The Core Idea

A probabilistic model is valid as long as its predictions look like they come from the right distribution. The **Probability Integral Transform (PIT)** provides a universal test:

> If your model is correct, then transforming observations through the model's CDF produces values that are uniformly distributed on [0,1].

This monitor performs **sequential testing** of PIT uniformity using the Kolmogorov-Smirnov distance with anytime-valid thresholds.

## Why This Matters

Traditional model monitoring uses performance metrics (loss, accuracy, profit), but these are:
- **Task-specific**: What's "good" varies by application
- **Delayed**: Problems only show up after many bad decisions
- **Confounded**: Performance can degrade for reasons other than model failure

PIT monitoring is:
- **Model-agnostic**: Works for any probabilistic model
- **Outcome-independent**: Doesn't care about profit/loss, just distributional correctness
- **Early warning**: Detects systematic deviations before they accumulate
- **Diagnostic**: Tells you *how* the model is wrong

## Installation

```bash
# Clone or download this repository
cd pit_monitor

# Install dependencies
pip install numpy scipy matplotlib pytest
```

## Quick Start

```python
from monitor import PITMonitor
from scipy.stats import norm

# Initialize monitor (only parameter: false alarm tolerance)
monitor = PITMonitor(false_alarm_rate=0.05)

# For each prediction and observation
for prediction, observation in data_stream:
    # prediction should be a scipy distribution or have a .cdf method
    alarm = monitor.update(prediction.cdf, observation)
    
    if alarm:
        print(f"Model broke at time {monitor.t}")
        print(f"Diagnosis: {alarm.diagnosis}")
        
        # Localize when it started breaking
        changepoint = monitor.localize_changepoint()
        print(f"Problem started around time {changepoint}")
        break
```

## Complete Example

```python
import numpy as np
from scipy.stats import norm
from monitor import PITMonitor

# Simulate a scenario: weather forecast that becomes biased
np.random.seed(42)
monitor = PITMonitor(false_alarm_rate=0.05)

# Days 1-100: forecast is accurate
for day in range(100):
    true_temp = 70 + np.random.normal(0, 10)
    forecast = norm(loc=70, scale=10)  # Correct model
    
    alarm = monitor.update(forecast.cdf, true_temp)
    if alarm:
        print(f"Unexpected alarm at day {day}")

print(f"Days 1-100: No alarm (model is good)")

# Days 101-200: forecast becomes systematically wrong (always predicts too high)
for day in range(100, 200):
    true_temp = 65 + np.random.normal(0, 10)  # True mean dropped to 65
    forecast = norm(loc=70, scale=10)  # Still predicting 70
    
    alarm = monitor.update(forecast.cdf, true_temp)
    if alarm:
        print(f"\nAlarm triggered at day {day}")
        print(f"Diagnosis: {alarm.diagnosis}")
        
        cp = monitor.localize_changepoint()
        print(f"Estimated changepoint: day {cp}")
        print(f"(True changepoint was day 100)")
        break

# Visualize
monitor.plot_diagnostics()
```

## Key Features

### 1. **No Arbitrary Parameters**

The only parameter is `false_alarm_rate`: your tolerance for false positives. This is not arbitrary—it's the fundamental question you must answer in any statistical test.

```python
# Conservative: rarely alarm unless very confident
monitor = PITMonitor(false_alarm_rate=0.01)

# Standard: typical scientific threshold
monitor = PITMonitor(false_alarm_rate=0.05)

# Liberal: catch problems early, tolerate occasional false alarms
monitor = PITMonitor(false_alarm_rate=0.10)
```

### 2. **Automatic Diagnosis**

When an alarm triggers, you get interpretable diagnostics:

```python
alarm = monitor.update(prediction.cdf, observation)
if alarm:
    print(alarm.diagnosis)
    # Example outputs:
    # "lower tail - overconfident (observed values more extreme than predicted)"
    # "upper tail - underconfident (observed values less extreme than predicted)"
    # "central region - underconfident (observed values more extreme than predicted)"
```

### 3. **Changepoint Localization**

After an alarm, estimate when the model started failing:

```python
if alarm:
    changepoint = monitor.localize_changepoint()
    print(f"Model was valid until approximately time {changepoint}")
```

### 4. **Visual Diagnostics**

```python
monitor.plot_diagnostics()
# Creates 4-panel figure:
# 1. PIT histogram (should be flat/uniform)
# 2. Empirical CDF vs uniform (should match diagonal)
# 3. KS distance over time (should stay below threshold)
# 4. PIT sequence (should be randomly scattered)
```

## How It Works

### The Math (Simple Version)

1. **Transform observations**: For outcome Y and model CDF F, compute U = F(Y)
2. **Test uniformity**: If model is correct, U ~ Uniform(0,1)
3. **Monitor deviation**: Track KS distance D_t = sup|F_empirical(u) - u|
4. **Alarm when**: D_t > threshold_t, where threshold_t = √(log(1/α)/t) approximately

That's it. The implementation handles:
- Time-varying thresholds that account for sequential testing
- Optional stopping guarantees (no p-hacking)
- Efficient computation

### Theoretical Guarantees

- **False alarm control**: P(false alarm) ≤ α over entire monitoring period
- **Anytime validity**: Can stop monitoring at any time without invalidating the test
- **No data peeking**: Optional stopping is built in (unlike classical hypothesis tests)

### Methods

Two threshold methods available:

1. **`alpha_spending`** (default, simple, fully derivable):
   - Uses DKW inequality + union bound
   - Threshold scales as √(log t / t)
   - Straightforward proof from first principles

2. **`stitching`** (tighter, but more complex):
   - Epoch-based threshold refinement
   - Threshold scales as √(log log t / t)
   - Slightly better power, same guarantees

For most applications, the difference is negligible. Use `alpha_spending` unless you're operating at extreme scales.

## Examples

See `examples/` directory:

### Weather Forecasting (`example_weather.py`)
- Monitor probabilistic temperature forecasts
- Detect when forecast model becomes miscalibrated
- Works with seasonal patterns and varying uncertainty

### Financial Risk Models (`example_financial.py`)
- Validate Value-at-Risk (VaR) models
- Detect volatility regime changes
- Compare to traditional VaR backtesting

### Running Examples

```bash
cd examples
python example_weather.py
python example_financial.py
```

## Testing

```bash
cd tests
pytest test_monitor.py -v
```

Tests cover:
- Basic PIT computation and monitoring
- Alarm triggering on misspecified models
- False alarm rate control under null
- Changepoint localization
- Edge cases and error handling

## API Reference

### `PITMonitor`

Main class for sequential model validation.

**Parameters:**
- `false_alarm_rate` (float, default=0.05): Maximum probability of false alarm
- `method` (str, default='alpha_spending'): Threshold method ('alpha_spending' or 'stitching')
- `changepoint_budget` (float, default=0.5): Fraction of α reserved for changepoint localization

**Key Methods:**
- `update(predicted_cdf, outcome)`: Process one observation, returns `AlarmInfo`
- `localize_changepoint()`: Estimate when model started failing (call after alarm)
- `plot_diagnostics()`: Create diagnostic visualizations
- `get_state()`: Export current state for inspection

**Attributes:**
- `t`: Current time (number of observations)
- `pits`: List of PIT values
- `alarm_triggered`: Boolean indicating if alarm has fired
- `alarm_time`: Time when alarm triggered (or None)

### `AlarmInfo`

Dataclass returned by `update()`.

**Attributes:**
- `triggered` (bool): Whether alarm fired
- `alarm_time` (int): When alarm occurred
- `ks_distance` (float): Current KS distance
- `threshold` (float): Current threshold
- `diagnosis` (str): Interpretation of the deviation
- `changepoint_estimate` (int): Estimated changepoint (if localized)

Evaluates to `True` if alarm triggered, allowing:
```python
if monitor.update(pred.cdf, obs):
    # Alarm triggered
```

## When to Use PIT Monitoring

**Good fit:**
- Probabilistic forecasts (weather, demand, risk)
- Online learning / adaptive models
- Production ML models that might drift
- Any setting where model correctness matters more than immediate performance

**Not ideal for:**
- Point predictions without uncertainty (use residual monitoring)
- Black-box models where you can't extract a predictive distribution
- Settings where only predictive accuracy matters (though you could monitor prediction intervals)

## Comparison to Alternatives

| Method | Scope | Stopping | Interpretability |
|--------|-------|----------|------------------|
| **PIT Monitor** | Model validity | Anytime-valid | Diagnostic (how model is wrong) |
| Traditional backtesting | Model performance | Fixed sample | Performance metrics only |
| Residual monitoring | Point predictions | Ad-hoc | Limited |
| Conformal prediction | Coverage guarantees | Fixed window | Coverage focus |
| Drift detection (KS test) | Distribution shift | Fixed sample | Non-sequential |

## Mathematical Background

The method combines three classical results:

1. **Rosenblatt (1952)**: PIT uniformity characterizes correct specification
2. **Dvoretzky-Kiefer-Wolfowitz (1956)**: Confidence bands for empirical CDFs
3. **Ville (1939) / Ramdas et al. (2020)**: Anytime-valid testing via supermartingales

The innovation is the clean synthesis: PIT uniformity testing + sequential monitoring with optional stopping.

## Citations

If you use this in research, please cite:

```
Rosenblatt, M. (1952). Remarks on a multivariate transformation. 
The Annals of Mathematical Statistics, 23(3), 470-472.

Dvoretzky, A., Kiefer, J., & Wolfowitz, J. (1956). 
Asymptotic minimax character of the sample distribution function and 
of the classical multinomial estimator. The Annals of Mathematical Statistics, 642-669.

Ramdas, A., Grünwald, P., Vovk, V., & Shafer, G. (2020). 
Game-theoretic statistics and safe anytime-valid inference. 
arXiv preprint arXiv:2210.01948.
```

## License

MIT License - feel free to use in research or production.

## Contributing

This is a research tool / proof of concept. Contributions welcome:
- Additional examples on real datasets
- Performance optimizations
- Additional diagnostic capabilities
- Extensions (multivariate monitoring, etc.)

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Remember**: The method is simple (KS on PITs with a shrinking threshold). The math ensures it works correctly under sequential testing. Everything else is convenience and interpretation.

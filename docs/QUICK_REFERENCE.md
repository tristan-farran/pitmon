# PIT Monitor - Quick Reference Card

## One-Line Summary
**Monitor whether your probabilistic model remains valid by testing if its probability integral transform (PIT) stays uniformly distributed.**

---

## Minimal Working Example

```python
from monitor import PITMonitor
from scipy.stats import norm

monitor = PITMonitor(false_alarm_rate=0.05)

for predicted_dist, observation in data_stream:
    if monitor.update(predicted_dist.cdf, observation):
        print(f"Model broke at t={monitor.t}")
        print(f"Started failing around t={monitor.localize_changepoint()}")
        break
```

---

## Key Concepts

### The PIT
- **What**: Transform observation through model's CDF: `U = F(Y)`
- **Property**: If model is correct, `U ~ Uniform(0,1)`
- **Test**: Check if PITs look uniform using KS distance

### The Threshold
- Shrinks over time: roughly `√(log(1/α) / t)`
- Accounts for sequential testing (no p-hacking)
- Only parameter: `false_alarm_rate` (your risk tolerance)

### The Alarm
- Triggers when: `KS_distance > threshold`
- Means: Model's predictions systematically wrong
- Action: Investigate, recalibrate, or replace model

---

## API Cheat Sheet

### Initialization
```python
monitor = PITMonitor(
    false_alarm_rate=0.05,      # Only required parameter
    method='alpha_spending',     # or 'stitching' (tighter)
    changepoint_budget=0.5       # For localization
)
```

### Update
```python
alarm = monitor.update(
    predicted_cdf,  # Callable: outcome → [0,1]
    outcome         # float: observed value
)
# Returns: AlarmInfo (evaluates to True if alarm)
```

### After Alarm
```python
if alarm:
    alarm.diagnosis              # Human-readable problem type
    alarm.ks_distance           # How far from uniform
    monitor.localize_changepoint()  # When it started
    monitor.plot_diagnostics()   # Visualize
```

### Inspection
```python
state = monitor.get_state()
# Returns: {'t', 'pits', 'ks_distance', 'threshold', 
#           'alarm_triggered', 'alarm_time', 'alpha', 'method'}
```

---

## Diagnosis Decoder

| Diagnosis Pattern | Meaning | Action |
|------------------|---------|--------|
| `lower tail - overconfident` | Underestimating extreme lows | Widen lower tail |
| `upper tail - overconfident` | Underestimating extreme highs | Widen upper tail |
| `central - underconfident` | Too uncertain (too wide) | Narrow distribution |
| `central - overconfident` | Too certain (too narrow) | Widen distribution |
| `X - underconfident - (less extreme)` | Observed less extreme than predicted | Reduce uncertainty |
| `X - overconfident - (more extreme)` | Observed more extreme than predicted | Increase uncertainty |

---

## Decision Tree

```
Do you have probabilistic predictions?
├─ NO → Use different monitoring method
└─ YES ↓

   Can you extract a CDF function?
   ├─ NO → Convert to distribution first
   └─ YES ↓

      Is model validity important?
      ├─ NO → Use performance metrics instead
      └─ YES ↓

         Want early warning of problems?
         ├─ NO → Use traditional backtesting
         └─ YES → USE PIT MONITOR ✓
```

---

## Common Patterns

### Pattern 1: Continuous Monitoring
```python
monitor = PITMonitor()
for pred, obs in production_stream():
    if monitor.update(pred.cdf, obs):
        alert_operations_team()
        trigger_model_retraining()
```

### Pattern 2: Batch Validation
```python
monitor = PITMonitor()
for pred, obs in validation_set:
    monitor.update(pred.cdf, obs)

if monitor.alarm_triggered:
    print(f"Model failed at observation {monitor.alarm_time}")
else:
    print("Model passed validation")
```

### Pattern 3: Comparative Testing
```python
monitors = {
    'model_v1': PITMonitor(),
    'model_v2': PITMonitor()
}

for pred_v1, pred_v2, obs in test_data:
    monitors['model_v1'].update(pred_v1.cdf, obs)
    monitors['model_v2'].update(pred_v2.cdf, obs)

# Which model broke first?
```

---

## What to Watch

### Green Flags ✓
- PITs scattered uniformly
- KS distance stays well below threshold
- PIT histogram roughly flat
- Empirical CDF follows diagonal

### Yellow Flags ⚠
- KS distance approaching threshold
- PITs clustering near 0 or 1
- Systematic drift in PIT sequence

### Red Flags 🚨
- Alarm triggered
- Diagnosis shows systematic bias
- KS distance >> threshold
- PIT histogram highly non-uniform

---

## Gotchas

❌ **Don't**: Use for point predictions without uncertainty
✓ **Do**: Ensure your model outputs a distribution

❌ **Don't**: Expect instant detection of small deviations  
✓ **Do**: Wait for systematic patterns to emerge

❌ **Don't**: Tune false_alarm_rate based on results
✓ **Do**: Choose it beforehand based on tolerance

❌ **Don't**: Assume alarm means immediate action
✓ **Do**: Investigate, diagnose, then decide

---

## Math Essentials

### PIT Theorem
```
Y ~ F  ⟹  U = F(Y) ~ Uniform(0,1)
```

### Test Statistic
```
D_t = sup|F̂_t(u) - u| = max|k/t - U_(k)|
```

### Threshold (α-spending)
```
ε_t = √(log(2/α_t) / 2t)  where α_t = α/(π²t²)
```

### Alarm Rule
```
Alarm when: D_t > ε_t
```

---

## Files in Package

```
pit_monitor/
├── monitor.py              # Core implementation
├── __init__.py            # Package exports
├── setup.py               # Installation
├── README.md              # Full documentation
├── GETTING_STARTED.md     # Tutorial
├── QUICK_REFERENCE.md     # This file
├── examples/
│   ├── example_weather.py      # Weather forecasting
│   ├── example_financial.py    # Financial risk
│   └── demo_comprehensive.py   # Full demo
└── tests/
    ├── test_monitor.py         # Full test suite
    └── run_tests.py            # Simple runner
```

---

## Dependencies

- **Required**: numpy, scipy
- **Optional**: matplotlib (for plotting)
- **Development**: pytest (for testing)

---

## When NOT to Use

- Point predictions without uncertainty → Use residual monitoring
- Immediate performance matters more → Use task-specific metrics  
- Can't extract predictive distribution → Convert model first
- Tiny sample sizes (< 20-30) → Wait for more data

---

## The One Thing to Remember

> **If PITs look uniform, model is valid. If not, it's broken.**

Everything else is just making this check:
- Sequential (over time)
- Rigorous (statistical guarantees)
- Actionable (diagnostics and localization)

---

**For more details**: See README.md
**To get started**: See GETTING_STARTED.md
**To understand deeply**: See the original document you uploaded

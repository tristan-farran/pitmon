# Pitmon

An anytime-valid monitor for Probability Integral Transform (PIT) values. It detects changes in calibration without needing a baseline period and controls the false alarm probability over the entire monitoring horizon. A stable but miscalibrated model will not trigger alarms; only *changes* in calibration are detected.

Installation
------------
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
-------------------
Each PIT is ranked among all previous PITs. Under exchangeability, these ranks
are uniform. Pitmon converts each rank into an e-value and tracks a mixture
e-process that accumulates evidence against exchangeability. An alarm is raised
once the e-process crosses $1/\alpha$.

For each new PIT u_t:
  1. RANK: Insert u_t into sorted list, get rank R_t ~ Uniform{1,...,t}
  2. P-VALUE: p_t = (R_t + U) / t where U ~ Uniform(0,1)
  3. BET: e_t = estimated density at p_t (plug-in from histogram)
  4. UPDATE: M_t = e_t · (M_{t-1} + 1/(t(t+1))) (mixture e-process)
  5. ALARM: if M_t ≥ 1/α

Why it works:
- Under exchangeability: p_t ~ Uniform(0,1), E[e_t] = 1, M_t is supermartingale
- Under change: p_t concentrates, e_t > 1 on average, M_t grows exponentially
- Ville's inequality: P(sup M_t ≥ 1/α) ≤ α

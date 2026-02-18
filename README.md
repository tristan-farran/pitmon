Pitmon
======

Sequential calibration monitor for probabilistic forecasts.

Pitmon implements an anytime-valid monitor for Probability Integral Transform (PIT) values. It detects changes in calibration without needing a baseline period and controls the false alarm probability over the entire monitoring horizon. A stable but miscalibrated model will not trigger alarms; only *changes* in calibration are detected.

Quick start
-----------
Monitor a stream of predictions and outcomes:

```python
from pitmonitor import PITMonitor

monitor = PITMonitor(false_alarm_rate=0.05)

for prediction, outcome in data_stream:
		alarm = monitor.update(prediction.cdf, outcome)
		if alarm:
				print(f"Alarm at t={monitor.t}")
				print(f"Changepoint estimate: {monitor.localize_changepoint()}")
				break
```

How it works (short)
-------------------
Each PIT is ranked among all previous PITs. Under exchangeability, these ranks
are uniform. Pitmon converts each rank into an e-value and tracks a mixture
e-process that accumulates evidence against exchangeability. An alarm is raised
once the e-process crosses $1/\alpha$.

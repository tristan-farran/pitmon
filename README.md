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

Examples
--------
- `examples/cifar_demo`: CIFAR-10 → CIFAR-10-C monitoring demo.
- `examples/delivery_demo`: synthetic delivery-time monitoring demo.

Run the CIFAR demo:
```bash
python examples/cifar_demo/run.py
```
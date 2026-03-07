# Additional Experiments

This folder contains focused, lightweight experiments that complement the main FriedmanDrift benchmark.

Implemented checks:

- Proposition verification:
  - asymptotic convergence of `E[e_t | A]` to `B * sum_b theta_b^2`
  - finite-time mean formula (Proposition "Finite-Sample Mean Gain")
  - warmup crossover and worst-case bound `n* <= m / (B * delta)`
- Stationary but dependent PIT stream (Gaussian-copula AR(1) with uniform margins)
- Mildly non-stationary PIT stream (slowly drifting symmetric Beta shape)
- Seasonal PIT stream (periodic symmetric Beta shape)
- Multi-step drift localization: changepoint error reported to both onset and maximal-intensity phase

## Run

From `Pitmon/experiment/additional`:

```bash
python run_additional.py
```

Profiles:

```bash
python run_additional.py --profile quick
python run_additional.py --profile standard
python run_additional.py --profile publication
```

Default profile is `standard` (research-grade precision with moderate runtime).
The runner auto-generates LaTeX macros at `out/additional_macros.tex` for appendix tables.

Outputs JSON results to:

- default: `out/results.json`
- configurable with `--output`

## Plot

Generate figures from the experiment JSON:

```bash
python plot_additional.py
```

Custom paths:

```bash
python plot_additional.py --input out/results.json --output-dir out/figures
```

## Notes on scientific validity

- The proposition checks estimate conditional expectations by Monte Carlo and report formula-vs-empirical errors.
- Stream-regime checks estimate stream-level alarm rate over repeated trials and include Wilson 95% confidence intervals.
- Diagnostics (mean, variance, lag-1 autocorrelation, KS statistic vs Uniform) are included to characterize each regime.

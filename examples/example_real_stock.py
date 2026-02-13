"""
Real-World Example: Stock Market Volatility Model Monitoring

This example demonstrates using PITMon to monitor option pricing and VaR models
in quantitative finance. It shows realistic patterns based on actual market behavior.

Use case: Detect when volatility models (GARCH, implied vol, etc.) become miscalibrated
due to regime changes, market stress, or model drift.

Data: Simulated based on S&P 500 historical patterns
In production: Use real market data from Bloomberg, Yahoo Finance, or your data vendor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from datetime import datetime, timedelta
from pitmon import PITMonitor


class VolatilityModelSimulator:
    """
    Simulates realistic stock returns and volatility forecasts.

    Based on real market patterns:
    - Normal times: ~15% annualized volatility
    - Crisis times: ~40%+ annualized volatility
    - Fat tails, volatility clustering
    """

    def __init__(self, ticker="SPY", seed=123):
        self.ticker = ticker
        self.rng = np.random.RandomState(seed)
        self.start_date = datetime(2020, 1, 1)

        # Model parameters (annualized)
        self.normal_vol = 0.15  # 15% vol
        self.crisis_vol = 0.45  # 45% vol
        self.vol_of_vol = 0.3

        # Current state
        self.current_vol = self.normal_vol
        self.last_return = 0

    def generate_trading_day(self, day_index):
        """
        Generate returns and volatility forecast for one trading day.

        Returns
        -------
        forecast : dict
            Volatility model forecast (VaR, implied vol, etc.)
        realized : dict
            Actual market return
        """
        date = self.start_date + timedelta(days=day_index)

        # Simulate regime changes (normal -> crisis -> recovery)
        if day_index < 120:
            # Normal market regime
            target_vol = self.normal_vol
        elif day_index < 140:
            # Crisis begins (e.g., COVID crash March 2020)
            target_vol = self.crisis_vol
        elif day_index < 180:
            # Crisis continues
            target_vol = self.crisis_vol * 0.9
        else:
            # Recovery to lower but elevated volatility
            target_vol = self.normal_vol * 1.5

        # Volatility mean reversion
        self.current_vol = 0.9 * self.current_vol + 0.1 * target_vol
        self.current_vol += self.rng.normal(0, self.vol_of_vol * self.current_vol * 0.1)
        self.current_vol = max(0.05, self.current_vol)  # Floor at 5%

        # Daily volatility (sqrt rule)
        daily_vol = self.current_vol / np.sqrt(252)

        # Generate return with volatility clustering and fat tails
        # Use Student's t with df=5 for fat tails
        df = 5
        scale = daily_vol * np.sqrt((df - 2) / df)  # Scale to match variance
        actual_return = self.rng.standard_t(df) * scale

        # Volatility forecast model
        if day_index < 120:
            # Model 1: GARCH estimated during normal times
            forecast_vol = self.normal_vol / np.sqrt(252)
            forecast_df = 5  # Correctly specified tails
            model_name = "GARCH(1,1) - Normal Regime"
        elif day_index < 160:
            # Transition: Model slow to adapt to crisis
            forecast_vol = min(self.normal_vol * 1.5, self.current_vol) / np.sqrt(252)
            forecast_df = 5
            model_name = "GARCH(1,1) - Adapting"
        else:
            # Model 2: Re-estimated on crisis data, but overfit
            forecast_vol = self.current_vol / np.sqrt(252)
            forecast_df = 8  # Wrong tail assumption (too thin)
            model_name = "GARCH(1,1) - Crisis Refit"

        forecast = {
            'date': date,
            'volatility': forecast_vol,
            'distribution': 't',
            'df': forecast_df,
            'model': model_name,
            'var_95': self._compute_var(forecast_vol, forecast_df, 0.95),
            'var_99': self._compute_var(forecast_vol, forecast_df, 0.99),
        }

        realized = {
            'date': date,
            'return': actual_return,
            'true_vol': daily_vol,
            'price_change_pct': actual_return * 100,
        }

        return forecast, realized

    @staticmethod
    def _compute_var(vol, df, confidence):
        """Compute Value at Risk for Student's t distribution."""
        # Two-sided VaR
        alpha = 1 - confidence
        t_critical = student_t.ppf(alpha / 2, df)
        return abs(t_critical * vol)


def run_real_stock_example():
    """Run realistic stock volatility monitoring example."""

    print("=" * 80)
    print("Real-World Example: Stock Volatility Model Monitoring (SPY)")
    print("=" * 80)
    print()
    print("Use Case: Monitor GARCH/VaR model calibration during market regimes")
    print("Goal: Detect when volatility forecasts become unreliable")
    print()

    # Initialize simulator
    market = VolatilityModelSimulator(ticker="SPY", seed=123)

    # Create PIT monitor for volatility forecasts
    monitor = PITMonitor(
        false_alarm_rate=0.05,
        baseline_size=60,  # ~3 months of trading days
    )

    # Storage for analysis
    forecasts = []
    returns = []
    dates = []
    models = []

    print("Monitoring volatility model calibration...")
    print("-" * 80)

    baseline_reported = False

    # Simulate 250 trading days (~1 year)
    for day in range(250):
        forecast, realized = market.generate_trading_day(day)

        forecasts.append(forecast)
        returns.append(realized)
        dates.append(forecast['date'])
        models.append(forecast['model'])

        # Create forecast distribution
        forecast_dist = student_t(
            df=forecast['df'],
            loc=0,  # Assume zero mean return
            scale=forecast['volatility']
        )

        # Update PIT monitor
        alarm = monitor.update(forecast_dist.cdf, realized['return'])

        # Report baseline
        if alarm.baseline_complete and not baseline_reported:
            print(f"\n✓ Baseline established (day {day + 1})")
            print(f"  Date: {forecast['date'].strftime('%Y-%m-%d')}")
            diag = monitor.get_baseline_diagnostics()
            print(f"  Quality: {diag['quality']}")
            print(f"  Baseline KS: {diag['ks_from_uniform']:.4f}")
            if diag['quality'] == 'poor':
                print(f"  Note: Model may be miscalibrated, but we'll detect CHANGES from this baseline")
            print(f"\nMonitoring for calibration changes...")
            print("-" * 80)
            baseline_reported = True

        # Check for alarm
        if alarm:
            print(f"\n{'='*80}")
            print(f"🚨 VOLATILITY MODEL ALARM")
            print(f"{'='*80}")
            print(f"Date: {forecast['date'].strftime('%Y-%m-%d')} (Day {day + 1})")
            print(f"Model: {forecast['model']}")
            print()
            print(f"PIT Statistics:")
            print(f"  KS Distance: {alarm.ks_distance:.4f}")
            print(f"  Threshold: {alarm.threshold:.4f}")
            print(f"  Significance: {(alarm.ks_distance / alarm.threshold - 1) * 100:.1f}% above threshold")
            print()
            print(f"Diagnosis: {alarm.diagnosis}")
            print()

            # Localize changepoint
            cp = monitor.localize_changepoint()
            cp_date = dates[cp - 1] if cp <= len(dates) else dates[-1]
            print(f"Changepoint Analysis:")
            print(f"  Estimated changepoint: Day {cp} ({cp_date.strftime('%Y-%m-%d')})")
            if cp <= len(models):
                print(f"  Model at changepoint: {models[cp-1]}")
            print()

            print(f"Risk Management Implications:")
            if "more extremes" in alarm.diagnosis:
                print(f"  ⚠️  Model is UNDERESTIMATING risk")
                print(f"  📊 VaR breaches likely higher than expected")
                print(f"  💰 Portfolio may be overexposed")
            else:
                print(f"  ℹ️  Model may be OVERESTIMATING risk")
                print(f"  📊 VaR may be too conservative")
                print(f"  💰 Potentially missing opportunities")
            print()

            print(f"Recommended Actions:")
            print(f"  1. Immediate: Review current positions and risk limits")
            print(f"  2. Investigate: Check for market regime changes")
            print(f"  3. Recalibrate: Re-estimate GARCH parameters on recent data")
            print(f"  4. Validate: Backtest recalibrated model")
            print(f"  5. Monitor: Establish new baseline after recalibration")
            print(f"{'='*80}")
            break

        # Progress updates
        if (day + 1) % 50 == 0 and not alarm:
            state = monitor.get_state()
            if monitor.baseline_locked and state['ks_distance'] and state['threshold']:
                margin = state['threshold'] - state['ks_distance']
                status = "✓ OK" if margin > 0 else "⚠ Warning"
                print(f"Day {day + 1} ({forecast['date'].strftime('%Y-%m-%d')}): {status} | Margin: {margin:.4f}")

    if not monitor.alarm_triggered:
        print(f"\n✓ No model calibration issues detected over {len(dates)} trading days")

    print()

    # Create visualizations
    print("Creating financial diagnostics...")
    create_financial_visualizations(
        monitor, forecasts, returns, dates, models
    )

    # Detailed summary
    print("\n" + "="*80)
    print("PITMon Summary Report")
    print("="*80)
    monitor.print_summary()

    # Additional financial metrics
    print("\n" + "="*80)
    print("Traditional VaR Backtesting (for comparison)")
    print("="*80)
    compute_var_backtest(forecasts, returns)


def compute_var_backtest(forecasts, returns):
    """Traditional VaR backtesting metrics."""
    returns_arr = np.array([r['return'] for r in returns])

    # 95% VaR violations
    var_95 = np.array([f['var_95'] for f in forecasts])
    violations_95 = np.abs(returns_arr) > var_95
    violation_rate_95 = np.mean(violations_95) * 100

    # 99% VaR violations
    var_99 = np.array([f['var_99'] for f in forecasts])
    violations_99 = np.abs(returns_arr) > var_99
    violation_rate_99 = np.mean(violations_99) * 100

    print(f"95% VaR:")
    print(f"  Expected violations: 5.0%")
    print(f"  Actual violations: {violation_rate_95:.2f}%")
    print(f"  Status: {'❌ FAIL' if abs(violation_rate_95 - 5.0) > 2.0 else '✓ PASS'}")
    print()
    print(f"99% VaR:")
    print(f"  Expected violations: 1.0%")
    print(f"  Actual violations: {violation_rate_99:.2f}%")
    print(f"  Status: {'❌ FAIL' if abs(violation_rate_99 - 1.0) > 1.0 else '✓ PASS'}")
    print()
    print(f"Note: Traditional VaR backtesting tells you IF the model failed,")
    print(f"      but PITMon tells you WHEN it started failing and HOW.")
    print("="*80)


def create_financial_visualizations(monitor, forecasts, returns, dates, models):
    """Create financial monitoring visualizations."""

    # Standard PITMon diagnostics
    fig1 = monitor.plot_diagnostics(figsize=(16, 10))
    fig1.suptitle(
        f'PITMon Diagnostics: SPY Volatility Model\n' +
        f'{dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}',
        fontsize=14, y=0.995
    )
    plt.tight_layout()
    plt.savefig("real_stock_pit_diagnostics.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: real_stock_pit_diagnostics.png")

    # Financial-specific plots
    fig2, axes = plt.subplots(4, 1, figsize=(16, 12))

    returns_arr = np.array([r['return'] * 100 for r in returns])  # Convert to %
    forecast_vols = np.array([f['volatility'] * 100 * np.sqrt(252) for f in forecasts])  # Annualized %
    true_vols = np.array([r['true_vol'] * 100 * np.sqrt(252) for r in returns])
    var_95 = np.array([f['var_95'] * 100 for f in forecasts])

    # Plot 1: Returns with VaR bands
    ax = axes[0]
    ax.plot(dates, returns_arr, 'k-', linewidth=0.6, alpha=0.7, label='Daily Returns')
    ax.fill_between(dates, -var_95, var_95, alpha=0.2, color='red', label='95% VaR Band')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Mark VaR violations
    violations = np.abs(returns_arr) > var_95
    violation_dates = [dates[i] for i in range(len(dates)) if violations[i]]
    violation_returns = [returns_arr[i] for i in range(len(returns_arr)) if violations[i]]
    ax.scatter(violation_dates, violation_returns, color='red', s=50,
               marker='x', linewidths=2, label=f'VaR Violations ({np.sum(violations)})', zorder=5)

    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='orange', linestyle=':',
                   linewidth=2.5, label=f'PIT Alarm (day {monitor.alarm_time})', zorder=3)

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Return (%)', fontsize=10)
    ax.set_title('Daily Returns and Value at Risk', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Volatility forecasts vs realized
    ax = axes[1]
    ax.plot(dates, forecast_vols, 'b-', linewidth=1.5, alpha=0.8, label='Forecast Volatility')
    ax.plot(dates, true_vols, 'g--', linewidth=1.5, alpha=0.6, label='True Volatility')

    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='orange', linestyle=':',
                   linewidth=2.5, label='PIT Alarm')

    # Mark regime changes
    model_changes = [i for i in range(1, len(models)) if models[i] != models[i-1]]
    for idx in model_changes:
        ax.axvline(dates[idx], color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Model Change' if idx == model_changes[0] else '')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Volatility (% ann.)', fontsize=10)
    ax.set_title('Volatility Forecasts vs Realized Volatility', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: PIT values (colored by regime)
    ax = axes[2]
    pits = monitor.pits
    pit_dates = dates[:len(pits)]
    baseline_size = monitor.baseline_size

    if len(pits) > baseline_size:
        ax.scatter(pit_dates[:baseline_size], pits[:baseline_size],
                   alpha=0.5, s=20, color='blue', label='Baseline', zorder=3)
        ax.scatter(pit_dates[baseline_size:], pits[baseline_size:],
                   alpha=0.5, s=20, color='orange', label='Monitoring', zorder=3)
    else:
        ax.scatter(pit_dates, pits, alpha=0.5, s=20, color='blue', label='Baseline', zorder=3)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.05, color='red', linestyle=':', linewidth=0.8, alpha=0.3, label='5% / 95%')
    ax.axhline(0.95, color='red', linestyle=':', linewidth=0.8, alpha=0.3)

    if baseline_size < len(dates):
        ax.axvline(dates[baseline_size], color='green', linestyle='--',
                   linewidth=2, alpha=0.6, label='Baseline Complete')

    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='orange', linestyle=':',
                   linewidth=2.5, label='Alarm')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('PIT Value', fontsize=10)
    ax.set_title('Probability Integral Transform Values', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Cumulative VaR violations vs expected
    ax = axes[3]
    violations = np.abs(returns_arr) > var_95
    cum_violations = np.cumsum(violations)
    expected_violations = np.arange(1, len(dates) + 1) * 0.05

    ax.plot(dates, cum_violations, 'r-', linewidth=2, label='Actual Violations', alpha=0.8)
    ax.plot(dates, expected_violations, 'b--', linewidth=2, label='Expected (5%)', alpha=0.8)
    ax.fill_between(dates, expected_violations, cum_violations,
                     where=(cum_violations > expected_violations),
                     alpha=0.2, color='red', label='Excess Violations')

    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='orange', linestyle=':',
                   linewidth=2.5, label='PIT Alarm (Early Warning!)')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Cumulative Violations', fontsize=10)
    ax.set_title('VaR Backtesting: Cumulative Violations', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("real_stock_analysis.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: real_stock_analysis.png")
    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PITMon: Real-World Stock Volatility Model Monitoring")
    print("="*80)
    print()
    print("Application: Monitor GARCH/VaR models in quantitative finance")
    print()
    print("Real-world deployment:")
    print("  • Integrate with trading systems (Bloomberg, Reuters, internal)")
    print("  • Monitor daily P&L distributions")
    print("  • Alert risk managers when models drift")
    print("  • Automate model recalibration triggers")
    print("  • Regulatory compliance (Basel, Dodd-Frank)")
    print()

    run_real_stock_example()

    print("\n" + "="*80)
    print("Key Advantages of PIT Monitoring for Finance:")
    print("="*80)
    print("✓ Early detection: Catches model issues before many VaR breaches accumulate")
    print("✓ Diagnostic: Explains HOW model is wrong (under/over-estimating risk)")
    print("✓ Regime-aware: Detects changes without assuming constant volatility")
    print("✓ Distribution-free: Works for any model (GARCH, EWMA, implied vol, etc.)")
    print("✓ Actionable: Provides changepoint estimates for investigation")
    print()
    print("Traditional VaR backtesting:")
    print("  • Only counts violations")
    print("  • Hard to tell WHEN model broke")
    print("  • Low power with small samples")
    print()
    print("PIT monitoring:")
    print("  • Uses full distributional information")
    print("  • Localizes changepoints")
    print("  • Higher statistical power")
    print("="*80 + "\n")

    plt.show()

"""
Example 2: Financial Risk Model Validation

Demonstrates using PIT monitor to validate Value-at-Risk (VaR) models.
Shows detection of regime change when market volatility shifts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from pitmon import PITMonitor


def simulate_returns_with_volatility_shift(n_days=500, changepoint=250):
    """
    Simulate stock returns with volatility regime change.
    
    Parameters
    ----------
    n_days : int
        Number of trading days
    changepoint : int
        Day when volatility regime changes
        
    Returns
    -------
    returns : np.ndarray
        Daily returns
    true_volatility : np.ndarray
        True volatility at each time
    """
    np.random.seed(42)
    
    returns = []
    volatility = []
    
    for day in range(n_days):
        if day < changepoint:
            # Low volatility regime
            sigma = 0.01  # 1% daily vol
        else:
            # High volatility regime (crisis!)
            sigma = 0.03  # 3% daily vol
        
        # Generate return with fat tails (Student's t)
        ret = sigma * student_t.rvs(df=5)
        
        returns.append(ret)
        volatility.append(sigma)
    
    return np.array(returns), np.array(volatility)


def run_financial_example():
    """Run financial risk model validation example."""
    
    print("=" * 70)
    print("PIT Monitor Example: Financial Risk Model Validation")
    print("=" * 70)
    print()
    
    # Generate returns with volatility shift
    returns, true_vol = simulate_returns_with_volatility_shift(
        n_days=500, changepoint=250
    )
    
    # Scenario: Using a VaR model calibrated to low-volatility regime
    print("Scenario: VaR model calibrated in low-vol regime, then crisis hits")
    print("-" * 70)
    print()
    
    # Model: Assume returns ~ N(0, sigma) with sigma estimated from first 100 days
    initial_window = returns[:100]
    estimated_sigma = np.std(initial_window)
    
    print(f"VaR model calibration:")
    print(f"  Estimated volatility: {estimated_sigma:.4f}")
    print(f"  True volatility (early): {true_vol[0]:.4f}")
    print(f"  True volatility (late): {true_vol[-1]:.4f}")
    print()
    
    # Now monitor the model going forward
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    # For each day, use the calibrated model to predict, then observe
    from scipy.stats import norm
    var_model = norm(loc=0, scale=estimated_sigma)
    
    print("Monitoring VaR model validity...")
    for day, ret in enumerate(returns, 1):
        alarm = monitor.update(var_model.cdf, ret)
        
        if alarm:
            print(f"\n⚠️  ALARM at day {day}")
            print(f"   KS distance: {alarm.ks_distance:.4f}")
            print(f"   Threshold: {alarm.threshold:.4f}")
            print(f"   Diagnosis: {alarm.diagnosis}")
            
            cp = monitor.localize_changepoint()
            print(f"   Estimated changepoint: day {cp}")
            print(f"   (True changepoint: day 250)")
            
            # What does this mean in practice?
            print(f"\n   Practical interpretation:")
            print(f"   - VaR estimates are no longer reliable")
            print(f"   - Risk is being systematically underestimated")
            print(f"   - Model should be recalibrated immediately")
            break
    
    if not monitor.alarm_triggered:
        print(f"✓ No alarm in {monitor.t} days (model still valid)")
    
    print()
    
    # Visualizations
    print("Creating diagnostic plots...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    days = np.arange(1, len(returns) + 1)
    
    # Plot 1: Returns and volatility regimes
    ax = axes[0]
    ax.plot(days, returns * 100, 'k-', alpha=0.5, linewidth=0.5)
    ax.axvline(250, color='red', linestyle='--', alpha=0.7, 
              linewidth=2, label='True volatility shift')
    if monitor.alarm_triggered:
        ax.axvline(monitor.alarm_time, color='orange', linestyle=':', 
                  linewidth=2, label=f'Alarm (day {monitor.alarm_time})')
    
    # Add confidence bands from VaR model
    ax.fill_between(days, -2*estimated_sigma*100, 2*estimated_sigma*100,
                    alpha=0.2, color='blue', label='Model 95% CI')
    
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Daily Return (%)')
    ax.set_title('Stock Returns and VaR Model Confidence Interval')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: PIT values
    ax = axes[1]
    pits = monitor.pits
    ax.scatter(days[:len(pits)], pits, alpha=0.4, s=15, c='blue')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.05, color='red', linestyle=':', alpha=0.5, label='5% VaR level')
    ax.axhline(0.95, color='red', linestyle=':', alpha=0.5)
    ax.axvline(250, color='red', linestyle='--', alpha=0.7, linewidth=2)
    if monitor.alarm_triggered:
        ax.axvline(monitor.alarm_time, color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('PIT Value')
    ax.set_title('PIT Values (excess clustering near 0 or 1 indicates problems)')
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Cumulative VaR violations
    ax = axes[2]
    # Count VaR violations (returns beyond 95% CI)
    var_95 = 1.96 * estimated_sigma
    violations = np.abs(returns) > var_95
    cumulative_violations = np.cumsum(violations)
    expected_violations = days * 0.05  # Should be ~5%
    
    ax.plot(days, cumulative_violations, 'b-', linewidth=2, label='Actual violations')
    ax.plot(days, expected_violations, 'r--', linewidth=2, label='Expected (5%)')
    ax.axvline(250, color='red', linestyle='--', alpha=0.7, linewidth=2)
    if monitor.alarm_triggered:
        ax.axvline(monitor.alarm_time, color='orange', linestyle=':', 
                  linewidth=2, label=f'PIT alarm (day {monitor.alarm_time})')
    ax.set_xlabel('Trading Day')
    ax.set_ylabel('Cumulative VaR Violations')
    ax.set_title('VaR Backtesting: Cumulative Violations vs Expected')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('financial_var_monitoring.png', dpi=150, bbox_inches='tight')
    print("Saved: financial_var_monitoring.png")
    
    # Full diagnostics
    fig = monitor.plot_diagnostics(figsize=(14, 10))
    plt.savefig('financial_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: financial_diagnostics.png")
    
    print()
    print("=" * 70)
    print("Key Insights:")
    print()
    print("Traditional VaR backtesting:")
    print(f"  - Counts violations (actual: {cumulative_violations[-1]:.0f}, expected: {expected_violations[-1]:.0f})")
    print(f"  - But when did the model break? Hard to tell from violation counts")
    print()
    print("PIT monitoring:")
    print(f"  - Detected problem at day {monitor.alarm_time}")
    print(f"  - Localized changepoint to ~day {monitor.localize_changepoint()}")
    print(f"  - Diagnosis: {monitor._alarm_info.diagnosis}")
    print()
    print("Advantage: PIT monitoring is:")
    print("  1. Model-agnostic (works for any distributional model)")
    print("  2. Early warning (detects patterns before many violations accumulate)")
    print("  3. Diagnostic (tells you HOW the model is wrong)")
    print("=" * 70)


if __name__ == '__main__':
    run_financial_example()
    plt.show()

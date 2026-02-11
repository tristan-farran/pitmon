"""
Example 1: Weather Forecast Validation

Demonstrates using PIT monitor to validate probabilistic temperature forecasts.
Shows how to detect when a forecast model becomes miscalibrated.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from pitmon import PITMonitor


def simulate_weather_forecasts(n_days=365, changepoint=None):
    """
    Simulate probabilistic temperature forecasts and observations.
    
    Parameters
    ----------
    n_days : int
        Number of days to simulate
    changepoint : int or None
        Day when forecast model becomes miscalibrated
        
    Returns
    -------
    forecasts : list of scipy.stats distributions
        Predicted distributions for each day
    observations : np.ndarray
        Observed temperatures
    """
    np.random.seed(42)
    
    forecasts = []
    observations = []
    
    for day in range(n_days):
        # True temperature has seasonal variation
        true_mean = 60 + 20 * np.sin(2 * np.pi * day / 365)
        true_std = 10
        
        # Forecast model
        if changepoint is None or day < changepoint:
            # Correct forecast (calibrated)
            forecast_mean = true_mean
            forecast_std = true_std
        else:
            # After changepoint: systematic bias (e.g., model not updated for climate change)
            forecast_mean = true_mean - 8  # Forecasts significantly too low
            forecast_std = true_std  # Keep same uncertainty
        
        # Generate observation from true distribution
        observation = np.random.normal(true_mean, true_std)
        
        # Store forecast as scipy distribution
        forecast_dist = norm(loc=forecast_mean, scale=forecast_std)
        
        forecasts.append(forecast_dist)
        observations.append(observation)
    
    return forecasts, np.array(observations)


def run_weather_example():
    """Run weather forecast validation example."""
    
    print("=" * 70)
    print("PIT Monitor Example: Weather Forecast Validation")
    print("=" * 70)
    print()
    
    # Scenario 1: Well-calibrated forecasts
    print("Scenario 1: Well-calibrated forecasts (no changepoint)")
    print("-" * 70)
    
    forecasts, observations = simulate_weather_forecasts(n_days=365, changepoint=None)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    for day, (forecast, obs) in enumerate(zip(forecasts, observations), 1):
        alarm = monitor.update(forecast.cdf, obs)
        if alarm:
            print(f"⚠️  ALARM at day {day}")
            print(f"   Diagnosis: {alarm.diagnosis}")
            break
    
    if not monitor.alarm_triggered:
        print(f"✓ No alarm triggered in {monitor.t} days")
        print(f"  Final KS distance: {monitor.get_state()['ks_distance']:.4f}")
        print(f"  Final threshold: {monitor.get_state()['threshold']:.4f}")
    
    print()
    
    # Scenario 2: Forecast model becomes miscalibrated
    print("Scenario 2: Forecast model becomes miscalibrated at day 180")
    print("-" * 70)
    
    forecasts, observations = simulate_weather_forecasts(n_days=365, changepoint=180)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    for day, (forecast, obs) in enumerate(zip(forecasts, observations), 1):
        alarm = monitor.update(forecast.cdf, obs)
        if alarm:
            print(f"⚠️  ALARM at day {day}")
            print(f"   KS distance: {alarm.ks_distance:.4f}")
            print(f"   Threshold: {alarm.threshold:.4f}")
            print(f"   Diagnosis: {alarm.diagnosis}")
            
            # Localize the changepoint
            cp = monitor.localize_changepoint()
            print(f"   Estimated changepoint: day {cp}")
            print(f"   (True changepoint: day 180)")
            break
    
    print()
    
    # Create diagnostic plots
    print("Creating diagnostic plots...")
    fig = monitor.plot_diagnostics(figsize=(14, 10))
    plt.savefig('weather_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved: weather_diagnostics.png")
    
    # Create custom plot showing the forecasts
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    days = np.arange(1, len(observations) + 1)
    
    # Plot observations and forecast means
    ax = axes[0]
    forecast_means = [f.mean() for f in forecasts]
    ax.plot(days, observations, 'o', alpha=0.3, label='Observed temp', markersize=3)
    ax.plot(days, forecast_means, '-', alpha=0.7, label='Forecast mean', linewidth=2)
    ax.axvline(180, color='red', linestyle='--', alpha=0.5, label='True changepoint')
    if monitor.alarm_triggered:
        ax.axvline(monitor.alarm_time, color='orange', linestyle=':', 
                  linewidth=2, label=f'Alarm (day {monitor.alarm_time})')
    ax.set_xlabel('Day')
    ax.set_ylabel('Temperature (°F)')
    ax.set_title('Temperature Forecasts and Observations')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot PIT values
    ax = axes[1]
    pits = monitor.pits
    ax.scatter(days[:len(pits)], pits, alpha=0.4, s=20)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(180, color='red', linestyle='--', alpha=0.5, label='True changepoint')
    if monitor.alarm_triggered:
        ax.axvline(monitor.alarm_time, color='orange', linestyle=':', 
                  linewidth=2, label=f'Alarm (day {monitor.alarm_time})')
    ax.set_xlabel('Day')
    ax.set_ylabel('PIT value')
    ax.set_title('PIT Values Over Time (should be uniformly scattered)')
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weather_forecast_timeline.png', dpi=150, bbox_inches='tight')
    print("Saved: weather_forecast_timeline.png")
    
    print()
    print("=" * 70)
    print("Key Takeaways:")
    print("- PIT monitor detected miscalibration ~{} days after changepoint".format(
        monitor.alarm_time - 180 if monitor.alarm_triggered else 'N/A'))
    print("- Changepoint localization estimated day {} (true: day 180)".format(
        monitor.localize_changepoint() if monitor.alarm_triggered else 'N/A'))
    print("- This works without knowing anything about temperature scales,")
    print("  seasonal patterns, or what 'good' forecast error looks like")
    print("=" * 70)


if __name__ == '__main__':
    run_weather_example()
    plt.show()

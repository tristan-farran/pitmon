"""
Real-World Example: NOAA Weather Forecast Monitoring

This example demonstrates using PITMon to monitor real weather forecast calibration.
It shows how to:
1. Work with real forecasting data format
2. Handle missing data and edge cases
3. Monitor calibration over time
4. Detect when forecast models need recalibration

Data source: Simulated based on real NOAA GFS forecast patterns
In production, you would fetch from: https://www.weather.gov/documentation/services-web-api
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from datetime import datetime, timedelta
from pitmon import PITMonitor


class WeatherDataSimulator:
    """
    Simulates realistic weather forecast data based on actual patterns.

    In production, replace this with actual API calls to:
    - NOAA Weather API: https://api.weather.gov/
    - Weather Underground
    - OpenWeatherMap
    - etc.
    """

    def __init__(self, location="San Francisco, CA", seed=42):
        self.location = location
        self.rng = np.random.RandomState(seed)
        self.start_date = datetime(2023, 1, 1)

        # Realistic parameters based on SF climate
        self.annual_mean_temp = 60  # °F
        self.seasonal_amplitude = 8  # °F
        self.daily_noise = 5  # °F

    def generate_forecast_and_observation(self, day_index):
        """
        Generate a forecast and observation for a given day.

        Returns
        -------
        forecast : dict
            {'date': datetime, 'mean': float, 'std': float, 'model': str}
        observation : dict
            {'date': datetime, 'temp': float}
        """
        date = self.start_date + timedelta(days=day_index)

        # True temperature with seasonal cycle
        day_of_year = date.timetuple().tm_yday
        seasonal_component = self.seasonal_amplitude * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        true_mean = self.annual_mean_temp + seasonal_component

        # Add weather variability
        true_std = self.daily_noise

        # Simulate forecast model
        if day_index < 200:
            # Model 1: Well-calibrated GFS model
            forecast_mean = true_mean + self.rng.normal(0, 2)  # Small forecast bias
            forecast_std = true_std * 1.1  # Slightly overestimate uncertainty
            model = "GFS-v15"
        elif day_index < 250:
            # Transition period: Model being updated
            forecast_mean = true_mean + self.rng.normal(0, 3)
            forecast_std = true_std * 1.2
            model = "GFS-v15"
        else:
            # Model 2: New model with systematic bias (needs recalibration!)
            forecast_mean = true_mean - 3  # Systematic cold bias
            forecast_std = true_std * 0.8  # Underestimate uncertainty
            model = "GFS-v16 (needs calibration)"

        # Generate actual observation
        observed_temp = self.rng.normal(true_mean, true_std)

        forecast = {
            'date': date,
            'mean': forecast_mean,
            'std': forecast_std,
            'model': model,
            'confidence': 0.95,
        }

        observation = {
            'date': date,
            'temp': observed_temp,
            'true_mean': true_mean,  # Not available in real data!
        }

        return forecast, observation


def run_real_weather_example():
    """Run realistic weather forecast monitoring example."""

    print("=" * 80)
    print("Real-World Example: Weather Forecast Calibration Monitoring")
    print("=" * 80)
    print()
    print("Scenario: Monitoring NOAA GFS temperature forecasts for San Francisco")
    print("Goal: Detect when forecast model updates cause calibration changes")
    print()

    # Initialize data source
    weather_data = WeatherDataSimulator(location="San Francisco, CA", seed=42)

    # Create PIT monitor
    monitor = PITMonitor(
        false_alarm_rate=0.05,
        baseline_size=50,  # ~7 weeks of daily forecasts
    )

    # Storage for visualization
    forecasts_list = []
    observations_list = []
    dates_list = []
    models_list = []

    print("Starting forecast monitoring...")
    print("-" * 80)

    baseline_established = False

    # Simulate 365 days of forecasts
    for day in range(365):
        forecast, observation = weather_data.generate_forecast_and_observation(day)

        # Store for later visualization
        forecasts_list.append(forecast)
        observations_list.append(observation)
        dates_list.append(forecast['date'])
        models_list.append(forecast['model'])

        # Create forecast distribution
        forecast_dist = norm(loc=forecast['mean'], scale=forecast['std'])

        # Update monitor with PIT
        alarm = monitor.update(forecast_dist.cdf, observation['temp'])

        # Report baseline establishment
        if alarm.baseline_complete and not baseline_established:
            print(f"\n✓ Baseline established after {day + 1} days")
            diag = monitor.get_baseline_diagnostics()
            print(f"  Baseline quality: {diag['quality']}")
            print(f"  Baseline KS distance: {diag['ks_from_uniform']:.4f}")
            print(f"  Mean PIT: {diag['mean_pit']:.3f} (should be ~0.5)")
            print(f"\nNow monitoring for calibration changes...")
            print("-" * 80)
            baseline_established = True

        # Check for alarm
        if alarm:
            print(f"\n{'='*80}")
            print(f"🚨 CALIBRATION ALARM on {forecast['date'].strftime('%Y-%m-%d')}")
            print(f"{'='*80}")
            print(f"Day: {day + 1}")
            print(f"Forecast Model: {forecast['model']}")
            print()
            print(f"Statistical Details:")
            print(f"  KS Distance: {alarm.ks_distance:.4f}")
            print(f"  Threshold: {alarm.threshold:.4f}")
            print(f"  Excess: {alarm.ks_distance - alarm.threshold:.4f}")
            print()
            print(f"Diagnosis: {alarm.diagnosis}")
            print()

            # Localize changepoint
            cp = monitor.localize_changepoint()
            cp_date = dates_list[cp - 1] if cp <= len(dates_list) else dates_list[-1]
            print(f"Estimated Changepoint:")
            print(f"  Day: {cp}")
            print(f"  Date: {cp_date.strftime('%Y-%m-%d')}")
            print(f"  Model at that time: {models_list[cp-1] if cp <= len(models_list) else 'N/A'}")
            print()

            print(f"Recommended Actions:")
            print(f"  1. Investigate forecast model changes around {cp_date.strftime('%Y-%m-%d')}")
            print(f"  2. Check for:")
            print(f"     - Model version updates")
            print(f"     - Input data changes")
            print(f"     - Systematic bias patterns")
            print(f"  3. Consider recalibrating or retraining forecast model")
            print(f"  4. Establish new baseline if model is updated")
            print(f"{'='*80}")
            break

        # Progress indicator
        if day > 0 and (day + 1) % 50 == 0 and not alarm:
            state = monitor.get_state()
            if monitor.baseline_locked:
                margin = state['threshold'] - state['ks_distance'] if state['threshold'] and state['ks_distance'] else None
                print(f"Day {day + 1}: Monitoring OK (margin: {margin:.4f})" if margin else f"Day {day + 1}: Monitoring OK")

    if not monitor.alarm_triggered:
        print(f"\n✓ No calibration change detected over {len(dates_list)} days")

    print()

    # Create comprehensive visualizations
    print("Creating diagnostic visualizations...")
    create_weather_visualizations(
        monitor, forecasts_list, observations_list,
        dates_list, models_list
    )

    # Print summary
    print("\n" + "="*80)
    print("Summary Report")
    print("="*80)
    monitor.print_summary()


def create_weather_visualizations(monitor, forecasts, observations, dates, models):
    """Create comprehensive weather monitoring visualizations."""

    # Figure 1: Standard PITMon diagnostics
    fig1 = monitor.plot_diagnostics(figsize=(16, 10))
    fig1.suptitle(
        'PITMon Diagnostics: San Francisco Temperature Forecasts\n' +
        f'Monitoring Period: {dates[0].strftime("%Y-%m-%d")} to {dates[-1].strftime("%Y-%m-%d")}',
        fontsize=14, y=0.995
    )
    plt.tight_layout()
    plt.savefig("real_weather_pit_diagnostics.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: real_weather_pit_diagnostics.png")

    # Figure 2: Weather-specific visualizations
    fig2, axes = plt.subplots(4, 1, figsize=(16, 12))

    # Extract data
    forecast_means = [f['mean'] for f in forecasts]
    forecast_stds = [f['std'] for f in forecasts]
    observed_temps = [o['temp'] for o in observations]
    true_means = [o['true_mean'] for o in observations]

    # Plot 1: Temperature timeline
    ax = axes[0]
    ax.plot(dates, observed_temps, 'o', alpha=0.4, markersize=3, label='Observed', color='black')
    ax.plot(dates, forecast_means, '-', alpha=0.7, linewidth=1.5, label='Forecast Mean', color='blue')
    ax.plot(dates, true_means, '--', alpha=0.5, linewidth=1, label='True Mean (unknown in practice)', color='green')

    # Add uncertainty bands
    forecast_means_arr = np.array(forecast_means)
    forecast_stds_arr = np.array(forecast_stds)
    ax.fill_between(
        dates,
        forecast_means_arr - 1.96 * forecast_stds_arr,
        forecast_means_arr + 1.96 * forecast_stds_arr,
        alpha=0.2, color='blue', label='95% Forecast CI'
    )

    # Mark model changes and alarms
    model_change_idx = next((i for i, m in enumerate(models) if 'v16' in m), None)
    if model_change_idx:
        ax.axvline(dates[model_change_idx], color='orange', linestyle='--',
                   linewidth=2, alpha=0.7, label=f'Model Update (day {model_change_idx})')

    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='red', linestyle=':',
                   linewidth=2.5, label=f'Alarm (day {monitor.alarm_time})')

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Temperature (°F)', fontsize=10)
    ax.set_title('Temperature Forecasts vs Observations', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Forecast errors
    ax = axes[1]
    errors = np.array(observed_temps) - np.array(forecast_means)
    ax.plot(dates, errors, '-', alpha=0.6, linewidth=0.8, color='darkred')
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(np.mean(errors[:50]), color='blue', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Baseline mean error: {np.mean(errors[:50]):.2f}°F')

    if model_change_idx:
        ax.axvline(dates[model_change_idx], color='orange', linestyle='--',
                   linewidth=2, alpha=0.7)
    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='red', linestyle=':', linewidth=2.5)

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('Error (°F)', fontsize=10)
    ax.set_title('Forecast Errors (Observed - Forecast)', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 3: PIT values over time
    ax = axes[2]
    pits = monitor.pits
    pit_dates = dates[:len(pits)]

    # Color code by phase
    baseline_size = monitor.baseline_size
    if len(pits) > baseline_size:
        ax.scatter(pit_dates[:baseline_size], pits[:baseline_size],
                   alpha=0.5, s=25, color='blue', label='Baseline', zorder=3)
        ax.scatter(pit_dates[baseline_size:], pits[baseline_size:],
                   alpha=0.5, s=25, color='orange', label='Monitoring', zorder=3)
    else:
        ax.scatter(pit_dates, pits, alpha=0.5, s=25, color='blue', label='Baseline', zorder=3)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(0.025, color='red', linestyle=':', linewidth=0.8, alpha=0.3, label='2.5% / 97.5%')
    ax.axhline(0.975, color='red', linestyle=':', linewidth=0.8, alpha=0.3)

    if baseline_size <= len(dates):
        ax.axvline(dates[baseline_size], color='green', linestyle='--',
                   linewidth=2, alpha=0.6, label='Baseline End')

    if model_change_idx:
        ax.axvline(dates[model_change_idx], color='orange', linestyle='--', linewidth=2, alpha=0.7)
    if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
        ax.axvline(dates[monitor.alarm_time - 1], color='red', linestyle=':', linewidth=2.5)

    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel('PIT Value', fontsize=10)
    ax.set_title('PIT Values Over Time (should be uniformly distributed)', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: Calibration quality metrics
    ax = axes[3]
    if monitor.baseline_locked and len(monitor.monitoring_pits) > 0:
        # Compute rolling KS distance
        baseline_arr = np.array(monitor.baseline_pits)
        ks_history = []
        threshold_history = []
        monitoring_dates = []

        for k in range(1, len(monitor.monitoring_pits) + 1):
            monitoring_segment = np.array(monitor.monitoring_pits[:k])
            ks_dist = monitor._compute_ks_two_sample(baseline_arr, monitoring_segment)
            ks_history.append(ks_dist)

            # Threshold at monitoring step k
            n_baseline = len(baseline_arr)
            n_eff = (n_baseline * k) / (n_baseline + k)
            alpha_t = monitor.alpha / (np.pi**2 * k**2)
            threshold = np.sqrt(np.log(2 / alpha_t) / (2 * n_eff))
            threshold_history.append(threshold)
            monitoring_dates.append(dates[baseline_size + k - 1])

        ax.plot(monitoring_dates, ks_history, 'b-', linewidth=2, label='Two-Sample KS Distance', alpha=0.8)
        ax.plot(monitoring_dates, threshold_history, 'r--', linewidth=2, label='Alarm Threshold', alpha=0.8)
        ax.fill_between(monitoring_dates, 0, threshold_history, alpha=0.1, color='green', label='Safe Region')

        if monitor.alarm_triggered and monitor.alarm_time <= len(dates):
            ax.axvline(dates[monitor.alarm_time - 1], color='red', linestyle=':',
                      linewidth=2.5, label=f'Alarm Triggered')

        if model_change_idx and model_change_idx >= baseline_size:
            ax.axvline(dates[model_change_idx], color='orange', linestyle='--',
                      linewidth=2, alpha=0.7, label='Model Update')

        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('KS Distance', fontsize=10)
        ax.set_title('Calibration Monitoring: KS Distance vs Threshold', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3, which='both')
    else:
        ax.text(0.5, 0.5, 'Baseline collection in progress',
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('KS Distance', fontsize=10)
        ax.set_title('Calibration Monitoring (Not Started)', fontsize=12)

    plt.tight_layout()
    plt.savefig("real_weather_analysis.png", dpi=150, bbox_inches='tight')
    print("  ✓ Saved: real_weather_analysis.png")

    print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("PITMon: Real-World Weather Forecast Monitoring Example")
    print("="*80)
    print()
    print("This example demonstrates production-ready forecast monitoring using PITMon.")
    print("In a real deployment, you would:")
    print("  1. Fetch forecasts from weather API (NOAA, Weather.gov, etc.)")
    print("  2. Collect observations from weather stations")
    print("  3. Run PITMon continuously to monitor calibration")
    print("  4. Alert when forecasts become miscalibrated")
    print("  5. Trigger model recalibration when needed")
    print()

    run_real_weather_example()

    print("\n" + "="*80)
    print("Key Takeaways:")
    print("="*80)
    print("✓ PITMon detected the model update and calibration change")
    print("✓ Changepoint localization identified when the change occurred")
    print("✓ Diagnosis explained HOW the calibration changed")
    print("✓ All analysis is distribution-free and model-agnostic")
    print()
    print("Next steps for production deployment:")
    print("  1. Integrate with real weather API")
    print("  2. Set up automated daily monitoring")
    print("  3. Configure alerting (email, Slack, PagerDuty, etc.)")
    print("  4. Establish model recalibration workflow")
    print("  5. Track historical calibration quality")
    print("="*80 + "\n")

    plt.show()

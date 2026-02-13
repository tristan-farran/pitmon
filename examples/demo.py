from pitmon import PITMonitor
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def demo_basic_usage():
    """Demonstrate basic usage."""
    print_section("1. BASIC USAGE")

    print("Creating a monitor with baseline_size=50:")
    monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)
    print(f"  • Initial state: t={monitor.t}, baseline_locked={monitor.baseline_locked}")
    print()

    print("Phase 1: Collecting baseline (50 observations)...")
    np.random.seed(42)
    predicted_dist = norm(0, 1)

    for i in range(50):
        observation = np.random.normal(0, 1)
        alarm = monitor.update(predicted_dist.cdf, observation)

        if i < 2:  # Show first few
            pit = monitor.baseline_pits[i]
            print(f"  Step {i+1}: observation={observation:+.3f}, PIT={pit:.3f}, baseline_complete={alarm.baseline_complete}")

    baseline_diag = monitor.get_baseline_diagnostics()
    print(f"  ...")
    print(f"  Baseline complete: {baseline_diag['complete']}")
    print(f"  Baseline quality: {baseline_diag['quality']} (KS={baseline_diag['ks_from_uniform']:.3f})")
    print()

    print("Phase 2: Monitoring for changes from baseline...")
    for i in range(20):
        observation = np.random.normal(0, 1)
        alarm = monitor.update(predicted_dist.cdf, observation)

    state = monitor.get_state()
    print(f"  After 20 monitoring steps:")
    print(f"    Two-sample KS = {state['ks_distance']:.4f}")
    print(f"    Threshold = {state['threshold']:.4f}")
    print(f"  → No calibration change detected")


def demo_alarm_triggering():
    """Demonstrate alarm triggering on misspecified model."""
    print_section("2. ALARM TRIGGERING")

    print("Scenario: Model predicts μ=2, but true data has μ=0")
    print()

    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)

    # Baseline: Model is wrong but stable
    predicted_dist = norm(2, 1)  # Predicting mean=2
    true_data = norm(0, 1)  # But data has mean=0

    observations = true_data.rvs(size=150, random_state=42)

    for i, obs in enumerate(observations, 1):
        alarm = monitor.update(predicted_dist.cdf, obs)

        if alarm:
            print(f"⚠️  ALARM TRIGGERED at step {i}")
            print(f"  • Two-sample KS distance: {alarm.ks_distance:.4f}")
            print(f"  • Threshold: {alarm.threshold:.4f}")
            print(f"  • Diagnosis: {alarm.diagnosis}")
            break

    if not monitor.alarm_triggered:
        print(f"✓ No alarm after {monitor.t} steps")
        print(f"  Note: Model is miscalibrated, but calibration is STABLE")
        print(f"  Baseline collected miscalibrated PITs, no change detected in monitoring")

    print()
    print("What this means:")
    print("  → Change detection monitors for DRIFT, not absolute miscalibration")
    print("  → A consistently biased model won't alarm (by design)")
    print("  → Only changes from baseline calibration trigger alarms")


def demo_regime_change():
    """Demonstrate regime change detection."""
    print_section("3. REGIME CHANGE DETECTION")

    print("Scenario: Baseline establishes 'normal', then regime changes")
    print()

    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)

    predicted_dist = norm(0, 1)

    # Baseline: Well-calibrated model
    print("Baseline (steps 1-50): Data ~ N(0,1), Model ~ N(0,1) ✓")
    data_baseline = norm(0, 1).rvs(size=50, random_state=42)
    for obs in data_baseline:
        alarm = monitor.update(predicted_dist.cdf, obs)
    print(f"  → Baseline established: {monitor.baseline_locked}")
    baseline_diag = monitor.get_baseline_diagnostics()
    print(f"  → Baseline quality: {baseline_diag['quality']} (KS={baseline_diag['ks_from_uniform']:.3f})")
    print()

    # Monitoring phase 1: Same distribution (no change)
    print("Monitoring Phase 1 (steps 51-100): Data ~ N(0,1), Model ~ N(0,1) ✓")
    data1 = norm(0, 1).rvs(size=50, random_state=43)
    for obs in data1:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            print(f"  Unexpected alarm at step {monitor.t}")
            break
    print(f"  → No alarm (calibration unchanged from baseline)")
    print()

    # Monitoring phase 2: Distribution changes
    print("Monitoring Phase 2 (steps 101+): Data ~ N(3,1), Model still ~ N(0,1) ✗")
    data2 = norm(3, 1).rvs(size=100, random_state=44)
    for obs in data2:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            print(f"  ⚠️  ALARM at step {monitor.t}")

            # Localize changepoint
            cp = monitor.localize_changepoint()
            print(f"  • Estimated changepoint: step {cp}")
            print(f"  • True changepoint: step 100 (monitoring step 50)")
            print(f"  • Detection lag: {monitor.t - 100} steps")
            print(f"  • Diagnosis: {alarm.diagnosis}")
            break


def demo_diagnostics():
    """Demonstrate diagnostic capabilities."""
    print_section("4. DIAGNOSTIC CAPABILITIES")

    np.random.seed(42)

    # Create three scenarios with different types of miscalibration
    scenarios = [
        {
            "name": "Overconfident (too narrow)",
            "true": norm(0, 1),
            "model": norm(0, 0.5),  # Underestimates variance
        },
        {
            "name": "Underconfident (too wide)",
            "true": norm(0, 1),
            "model": norm(0, 2),  # Overestimates variance
        },
        {
            "name": "Biased mean",
            "true": norm(0, 1),
            "model": norm(2, 1),  # Wrong location
        },
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 70)

        monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)
        data = scenario["true"].rvs(size=200, random_state=42)

        for obs in data:
            alarm = monitor.update(scenario["model"].cdf, obs)
            if alarm:
                print(f"  Alarm at step {monitor.t}")
                print(f"  Diagnosis: {alarm.diagnosis}")
                print(f"  Note: Stable miscalibration detected (baseline was also miscalibrated)")
                break

        if not monitor.alarm_triggered:
            print(f"  No alarm - calibration is stable (even if poor)")


def demo_comparison():
    """Demonstrate changepoint localization methods."""
    print_section("5. CHANGEPOINT LOCALIZATION")

    print("Comparing changepoint localization methods:")
    print()

    np.random.seed(42)

    # Create monitor
    monitor = PITMonitor(false_alarm_rate=0.05)

    # Correct regime then regime change
    predicted_dist = norm(0, 1)
    data1 = norm(0, 1).rvs(size=100, random_state=42)
    data2 = norm(2, 1).rvs(size=100, random_state=43)

    for obs in data1:
        monitor.update(predicted_dist.cdf, obs)

    for obs in data2:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            break

    if monitor.alarm_triggered:
        print(f"  Alarm triggered at step {monitor.t}")
        print(f"  True changepoint: step 100")
        print()

        cp = monitor.localize_changepoint()
        print(f"  Changepoint estimate: step {cp}")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_section("6. VISUALIZATION")

    print("Creating diagnostic plots...")
    print()

    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)

    # Generate regime change scenario
    predicted_dist = norm(0, 1)

    # Baseline regime (well-calibrated)
    data_baseline = norm(0, 1).rvs(size=50, random_state=42)
    for obs in data_baseline:
        monitor.update(predicted_dist.cdf, obs)

    # Monitoring: same distribution for a while
    data1 = norm(0, 1).rvs(size=50, random_state=43)
    for obs in data1:
        monitor.update(predicted_dist.cdf, obs)

    # Changed regime
    data2 = norm(2, 1).rvs(size=100, random_state=44)
    for obs in data2:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            break

    # Create diagnostic plots
    fig = monitor.plot_diagnostics(figsize=(14, 10))
    plt.savefig("demo_diagnostics.png", dpi=150, bbox_inches="tight")
    print("  Saved: demo_diagnostics.png")
    print()
    print("  The diagnostic plots show:")
    print("    1. PIT histogram (should be flat if model is correct)")
    print("    2. Empirical CDF vs uniform (should match diagonal)")
    print("    3. KS distance over time (crosses threshold at alarm)")
    print("    4. PIT sequence over time (should be randomly scattered)")


def demo_state_inspection():
    """Demonstrate state inspection."""
    print_section("7. STATE INSPECTION")

    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=30)

    predicted_dist = norm(0, 1)
    data = norm(0, 1).rvs(size=70, random_state=42)

    for obs in data:
        monitor.update(predicted_dist.cdf, obs)

    state = monitor.get_state()
    baseline_diag = monitor.get_baseline_diagnostics()

    print("Current monitor state:")
    print(f"  • Total time steps: {state['t']}")
    print(f"  • Baseline locked: {state['baseline_locked']}")
    print(f"  • Baseline size: {state['baseline_size']}")
    print(f"  • Monitoring size: {state['monitoring_size']}")
    print(f"  • Two-sample KS distance: {state['ks_distance']:.4f}")
    print(f"  • Current threshold: {state['threshold']:.4f}")
    print(f"  • False alarm rate (α): {state['alpha']}")
    print(f"  • Alarm triggered: {state['alarm_triggered']}")
    print()
    print("Baseline diagnostics:")
    print(f"  • Baseline quality: {baseline_diag['quality']}")
    print(f"  • Baseline KS from uniform: {baseline_diag['ks_from_uniform']:.4f}")
    print()
    print("  This state can be:")
    print("    - Saved for later analysis")
    print("    - Used for custom monitoring logic")
    print("    - Serialized for persistent storage")


def main():
    """Run all demonstrations."""
    print_section("PIT MONITOR: COMPREHENSIVE DEMONSTRATION")

    print("This demo shows all major features of the PIT Monitor.")
    print("Each section demonstrates a different capability.")
    print()
    print("Press Enter to continue through each section...")
    input()

    demos = [
        demo_basic_usage,
        demo_alarm_triggering,
        demo_regime_change,
        demo_diagnostics,
        demo_comparison,
        demo_visualization,
        demo_state_inspection,
    ]

    for demo_func in demos:
        demo_func()
        print()
        input("Press Enter to continue...")

    print_section("SUMMARY")
    print("Key takeaways:")
    print()
    print("  1. PIT Monitor is simple to use:")
    print("     • One parameter: false_alarm_rate")
    print("     • One method: update(predicted_cdf, observation)")
    print("     • Clear outputs: AlarmInfo with diagnosis")
    print()
    print("  2. It's model-agnostic:")
    print("     • Works with any probabilistic model")
    print("     • Doesn't depend on domain or scale")
    print("     • Just needs a CDF function")
    print("     • Uses alpha-spending thresholds for simplicity")
    print()
    print("  3. It provides actionable insights:")
    print("     • When: alarm time")
    print("     • Where: changepoint localization")
    print("     • How: diagnostic interpretation")
    print()
    print("  4. It has theoretical guarantees:")
    print("     • False alarm rate control")
    print("     • Anytime-valid testing")
    print("     • No data peeking / optional stopping")
    print()
    print("  Use PIT Monitor when:")
    print("    ✓ You have probabilistic predictions")
    print("    ✓ Model validity matters more than immediate performance")
    print("    ✓ You want early warning of model degradation")
    print("    ✓ You need interpretable diagnostics")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

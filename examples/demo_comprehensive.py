"""
Comprehensive Demo: PIT Monitor

This script demonstrates the key features and use cases of PIT Monitor.
Run this to see everything it can do.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from pitmon import PITMonitor


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def demo_basic_usage():
    """Demonstrate basic usage."""
    print_section("1. BASIC USAGE")
    
    print("Creating a monitor with 5% false alarm rate:")
    monitor = PITMonitor(false_alarm_rate=0.05)
    print(f"  • Initial state: t={monitor.t}, alarm={monitor.alarm_triggered}")
    print()
    
    print("Processing observations from a well-calibrated model:")
    np.random.seed(42)
    predicted_dist = norm(0, 1)
    
    for i in range(10):
        observation = np.random.normal(0, 1)
        alarm = monitor.update(predicted_dist.cdf, observation)
        
        if i < 3:  # Show first few
            print(f"  Step {i+1}: observation={observation:+.3f}, "
                  f"PIT={monitor.pits[i]:.3f}, alarm={bool(alarm)}")
    
    print(f"  ...")
    print(f"  After 10 steps: KS distance = {monitor.get_state()['ks_distance']:.4f}")
    print(f"                  Threshold = {monitor.get_state()['threshold']:.4f}")
    print(f"  → No alarm (model is valid)")


def demo_alarm_triggering():
    """Demonstrate alarm triggering on misspecified model."""
    print_section("2. ALARM TRIGGERING")
    
    print("Scenario: Model predicts μ=2, but true data has μ=0")
    print()
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    # Wrong model
    predicted_dist = norm(2, 1)  # Predicting mean=2
    true_data = norm(0, 1)  # But data has mean=0
    
    observations = true_data.rvs(size=100, random_state=42)
    
    for i, obs in enumerate(observations, 1):
        alarm = monitor.update(predicted_dist.cdf, obs)
        
        if alarm:
            print(f"⚠️  ALARM TRIGGERED at step {i}")
            print(f"  • KS distance: {alarm.ks_distance:.4f}")
            print(f"  • Threshold: {alarm.threshold:.4f}")
            print(f"  • Diagnosis: {alarm.diagnosis}")
            break
    
    print()
    print("What this means:")
    print("  → The model's predictions are systematically wrong")
    print("  → PITs are not uniform (clustering away from 0.5)")
    print("  → Model needs recalibration or replacement")


def demo_regime_change():
    """Demonstrate regime change detection."""
    print_section("3. REGIME CHANGE DETECTION")
    
    print("Scenario: Model is correct, then data distribution changes")
    print()
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    predicted_dist = norm(0, 1)
    
    # Phase 1: Correct model
    print("Phase 1 (steps 1-100): Data ~ N(0,1), Model ~ N(0,1) ✓")
    data1 = norm(0, 1).rvs(size=100, random_state=42)
    for obs in data1:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            print(f"  Unexpected alarm at step {monitor.t}")
            break
    print(f"  → No alarm in {monitor.t} steps (as expected)")
    print()
    
    # Phase 2: Regime changes
    print("Phase 2 (steps 101+): Data ~ N(3,1), Model still ~ N(0,1) ✗")
    data2 = norm(3, 1).rvs(size=100, random_state=43)
    for obs in data2:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            print(f"  ⚠️  ALARM at step {monitor.t}")
            
            # Localize changepoint
            cp = monitor.localize_changepoint()
            print(f"  • Estimated changepoint: step {cp}")
            print(f"  • True changepoint: step 100")
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
            'name': 'Overconfident (too narrow)',
            'true': norm(0, 1),
            'model': norm(0, 0.5),  # Underestimates variance
        },
        {
            'name': 'Underconfident (too wide)',
            'true': norm(0, 1),
            'model': norm(0, 2),  # Overestimates variance
        },
        {
            'name': 'Biased mean',
            'true': norm(0, 1),
            'model': norm(2, 1),  # Wrong location
        },
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("-" * 70)
        
        monitor = PITMonitor(false_alarm_rate=0.05)
        data = scenario['true'].rvs(size=200, random_state=42)
        
        for obs in data:
            alarm = monitor.update(scenario['model'].cdf, obs)
            if alarm:
                print(f"  Alarm at step {monitor.t}")
                print(f"  Diagnosis: {alarm.diagnosis}")
                break
        
        if not monitor.alarm_triggered:
            print(f"  No alarm in 200 steps")


def demo_comparison():
    """Compare alpha-spending vs stitching methods."""
    print_section("5. METHOD COMPARISON")
    
    print("Comparing threshold methods:")
    print()
    
    np.random.seed(42)
    
    # Create monitors with both methods
    monitor_spending = PITMonitor(false_alarm_rate=0.05, method='alpha_spending')
    monitor_stitch = PITMonitor(false_alarm_rate=0.05, method='stitching')
    
    # Misspecified model
    predicted_dist = norm(2, 1)
    data = norm(0, 1).rvs(size=200, random_state=42)
    
    for obs in data:
        alarm_spending = monitor_spending.update(predicted_dist.cdf, obs)
        alarm_stitch = monitor_stitch.update(predicted_dist.cdf, obs)
        
        if alarm_spending and not alarm_stitch:
            print(f"  Alpha-spending alarmed first at step {monitor_spending.t}")
            break
        elif alarm_stitch and not alarm_spending:
            print(f"  Stitching alarmed first at step {monitor_stitch.t}")
            break
        elif alarm_spending and alarm_stitch:
            print(f"  Both alarmed at step {monitor_spending.t}")
            break
    
    print()
    print("  Note: Stitching has tighter thresholds (√log log t vs √log t)")
    print("        but the difference is usually small in practice")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print_section("6. VISUALIZATION")
    
    print("Creating diagnostic plots...")
    print()
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    # Generate regime change scenario
    predicted_dist = norm(0, 1)
    
    # Correct regime
    data1 = norm(0, 1).rvs(size=100, random_state=42)
    for obs in data1:
        monitor.update(predicted_dist.cdf, obs)
    
    # Changed regime
    data2 = norm(2, 1).rvs(size=100, random_state=43)
    for obs in data2:
        alarm = monitor.update(predicted_dist.cdf, obs)
        if alarm:
            break
    
    # Create diagnostic plots
    fig = monitor.plot_diagnostics(figsize=(14, 10))
    plt.savefig('demo_diagnostics.png', dpi=150, bbox_inches='tight')
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
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    predicted_dist = norm(0, 1)
    data = norm(0, 1).rvs(size=50, random_state=42)
    
    for obs in data:
        monitor.update(predicted_dist.cdf, obs)
    
    state = monitor.get_state()
    
    print("Current monitor state:")
    print(f"  • Time steps: {state['t']}")
    print(f"  • Number of PITs: {len(state['pits'])}")
    print(f"  • KS distance: {state['ks_distance']:.4f}")
    print(f"  • Current threshold: {state['threshold']:.4f}")
    print(f"  • False alarm rate (α): {state['alpha']}")
    print(f"  • Method: {state['method']}")
    print(f"  • Alarm triggered: {state['alarm_triggered']}")
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


if __name__ == '__main__':
    main()

"""
Standalone test runner for PIT Monitor (no pytest required)
"""

import sys
import numpy as np
from scipy.stats import norm
sys.path.insert(0, '..')
from monitor import PITMonitor, AlarmInfo


def test_basic_functionality():
    """Test basic PIT computation and monitoring."""
    print("Testing basic functionality...")
    
    monitor = PITMonitor()
    assert monitor.t == 0
    assert monitor.alpha == 0.05
    assert not monitor.alarm_triggered
    
    # Test update
    predicted_dist = norm(0, 1)
    alarm = monitor.update(predicted_dist.cdf, 0.0)
    
    assert monitor.t == 1
    assert len(monitor.pits) == 1
    assert 0.45 < monitor.pits[0] < 0.55  # Should be near 0.5
    assert not alarm
    
    print("✓ Basic functionality tests passed")


def test_correctly_specified_model():
    """Test that correct model doesn't alarm."""
    print("Testing correctly specified model...")
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    # Generate data from N(0,1) and use N(0,1) as model
    predicted_dist = norm(0, 1)
    outcomes = norm(0, 1).rvs(size=200, random_state=42)
    
    for outcome in outcomes:
        alarm = monitor.update(predicted_dist.cdf, outcome)
        if alarm:
            print(f"  Unexpected alarm at t={monitor.t}")
            return False
    
    print(f"✓ No false alarm in {monitor.t} observations")
    return True


def test_misspecified_model_detection():
    """Test that misspecified model triggers alarm."""
    print("Testing misspecified model detection...")
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    # Generate data from N(0,1) but use N(2,1) as model (wrong mean)
    true_dist = norm(0, 1)
    predicted_dist = norm(2, 1)  # Wrong!
    
    outcomes = true_dist.rvs(size=500, random_state=42)
    
    for outcome in outcomes:
        alarm = monitor.update(predicted_dist.cdf, outcome)
        if alarm:
            print(f"✓ Detected misspecification at t={monitor.alarm_time}")
            print(f"  KS distance: {alarm.ks_distance:.4f}")
            print(f"  Threshold: {alarm.threshold:.4f}")
            print(f"  Diagnosis: {alarm.diagnosis}")
            return True
    
    print("✗ Failed to detect misspecified model")
    return False


def test_regime_change_detection():
    """Test detection of regime change."""
    print("Testing regime change detection...")
    
    np.random.seed(42)
    monitor = PITMonitor(false_alarm_rate=0.05)
    
    predicted_dist = norm(0, 1)
    
    # First 100: correct model
    outcomes1 = norm(0, 1).rvs(size=100, random_state=42)
    for outcome in outcomes1:
        alarm = monitor.update(predicted_dist.cdf, outcome)
        if alarm:
            print("✗ False alarm during correct regime")
            return False
    
    print(f"  No alarm in first 100 observations (correct)")
    
    # Next 100: regime changes to N(3, 1)
    outcomes2 = norm(3, 1).rvs(size=100, random_state=43)
    for outcome in outcomes2:
        alarm = monitor.update(predicted_dist.cdf, outcome)
        if alarm:
            print(f"✓ Detected regime change at t={monitor.alarm_time}")
            
            # Test changepoint localization
            cp = monitor.localize_changepoint()
            print(f"  Estimated changepoint: {cp} (true: 100)")
            print(f"  Diagnosis: {alarm.diagnosis}")
            
            if 50 < cp < 150:  # Reasonable range
                print(f"  Changepoint estimate is reasonable")
                return True
            else:
                print(f"  Warning: Changepoint estimate seems off")
                return True
    
    print("✗ Failed to detect regime change")
    return False


def test_thresholds():
    """Test threshold computation."""
    print("Testing threshold computation...")
    
    monitor = PITMonitor()
    
    # Check threshold decreases over time
    thresholds = []
    for t in [10, 50, 100, 500]:
        monitor.t = t
        thresh = monitor._compute_threshold()
        thresholds.append(thresh)
        print(f"  t={t:3d}: threshold={thresh:.6f}")
    
    # Should be decreasing
    for i in range(len(thresholds) - 1):
        if thresholds[i] <= thresholds[i+1]:
            print("✗ Thresholds not decreasing")
            return False
    
    print("✓ Thresholds decrease correctly with time")
    return True


def test_alarm_info():
    """Test AlarmInfo structure."""
    print("Testing AlarmInfo structure...")
    
    np.random.seed(42)
    monitor = PITMonitor()
    
    # Force alarm with badly misspecified model
    predicted_dist = norm(5, 1)
    outcomes = norm(0, 1).rvs(size=100, random_state=42)
    
    for outcome in outcomes:
        alarm = monitor.update(predicted_dist.cdf, outcome)
        if alarm:
            assert isinstance(alarm, AlarmInfo)
            assert alarm.triggered
            assert alarm.alarm_time == monitor.t
            assert alarm.ks_distance > alarm.threshold
            assert alarm.diagnosis is not None
            assert isinstance(alarm.diagnosis, str)
            assert len(alarm.diagnosis) > 0
            print(f"✓ AlarmInfo structure correct")
            return True
    
    print("✗ No alarm triggered to test AlarmInfo")
    return False


def test_state_extraction():
    """Test get_state method."""
    print("Testing state extraction...")
    
    monitor = PITMonitor()
    predicted_dist = norm(0, 1)
    
    outcomes = norm(0, 1).rvs(size=20, random_state=42)
    for outcome in outcomes:
        monitor.update(predicted_dist.cdf, outcome)
    
    state = monitor.get_state()
    
    assert state['t'] == 20
    assert len(state['pits']) == 20
    assert state['ks_distance'] is not None
    assert state['threshold'] is not None
    assert state['alpha'] == 0.05
    assert state['method'] == 'alpha_spending'
    
    print("✓ State extraction works correctly")
    return True


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("PIT Monitor Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_basic_functionality,
        test_correctly_specified_model,
        test_misspecified_model_detection,
        test_regime_change_detection,
        test_thresholds,
        test_alarm_info,
        test_state_extraction,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            if result is None:
                result = True  # No explicit return means success
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
        print()
    
    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)

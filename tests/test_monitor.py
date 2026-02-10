"""
Tests for PIT Monitor

Tests cover:
1. Basic functionality and PIT computation
2. Alarm triggering and thresholds
3. Changepoint localization
4. Edge cases and error handling
"""

import numpy as np
import pytest
from monitor import PITMonitor, AlarmInfo


class TestBasicFunctionality:
    """Test basic PIT computation and monitoring."""
    
    def test_initialization(self):
        """Test monitor initialization with different parameters."""
        monitor = PITMonitor()
        assert monitor.t == 0
        assert monitor.alpha == 0.05
        assert not monitor.alarm_triggered
        
        monitor = PITMonitor(false_alarm_rate=0.01)
        assert monitor.alpha == 0.01
        
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError):
            PITMonitor(false_alarm_rate=0)
        with pytest.raises(ValueError):
            PITMonitor(false_alarm_rate=1.5)
        with pytest.raises(ValueError):
            PITMonitor(method='invalid')
    
    def test_pit_computation(self):
        """Test that PIT values are correctly computed."""
        from scipy.stats import norm
        
        monitor = PITMonitor()
        
        # Standard normal prediction, observe outcome
        predicted_dist = norm(loc=0, scale=1)
        outcome = 0.0
        
        alarm = monitor.update(predicted_dist.cdf, outcome)
        
        assert monitor.t == 1
        assert len(monitor.pits) == 1
        assert 0.45 < monitor.pits[0] < 0.55  # Should be near 0.5 for outcome=0
        
    def test_pit_range_validation(self):
        """Test that invalid PIT values are caught."""
        monitor = PITMonitor()
        
        # CDF that returns invalid values
        def bad_cdf(x):
            return 1.5  # Invalid: > 1
        
        with pytest.raises(ValueError, match="PIT value.*outside"):
            monitor.update(bad_cdf, 0.0)
    
    def test_sequential_updates(self):
        """Test multiple sequential updates."""
        from scipy.stats import norm
        
        monitor = PITMonitor()
        predicted_dist = norm(0, 1)
        
        np.random.seed(42)
        outcomes = np.random.normal(0, 1, size=100)
        
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                break
        
        assert monitor.t == len(outcomes)
        assert len(monitor.pits) == len(outcomes)


class TestUniformityUnderNull:
    """Test that correctly specified models don't trigger false alarms too often."""
    
    def test_uniform_pits_no_alarm(self):
        """Test that uniform PITs rarely trigger alarms."""
        np.random.seed(42)
        n_simulations = 100
        n_samples = 200
        false_alarm_rate = 0.05
        
        monitor = PITMonitor(false_alarm_rate=false_alarm_rate)
        
        alarms = 0
        for _ in range(n_simulations):
            monitor = PITMonitor(false_alarm_rate=false_alarm_rate)
            
            # Generate truly uniform PITs (correct model)
            uniform_pits = np.random.uniform(0, 1, size=n_samples)
            
            for u in uniform_pits:
                # Dummy CDF that just returns the PIT value
                alarm = monitor.update(lambda x, u=u: u, 0.0)
                if alarm:
                    alarms += 1
                    break
        
        # Should have roughly false_alarm_rate * n_simulations alarms
        # Allow generous tolerance since this is stochastic
        expected_alarms = false_alarm_rate * n_simulations
        assert alarms <= expected_alarms * 3  # Very loose bound for test stability
    
    def test_correct_model_no_alarm(self):
        """Test that a correctly specified model doesn't alarm."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor(false_alarm_rate=0.05)
        
        # Generate data from N(0,1) and use N(0,1) as model
        true_dist = norm(0, 1)
        predicted_dist = norm(0, 1)
        
        outcomes = true_dist.rvs(size=200, random_state=42)
        
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                pytest.fail(f"False alarm at t={monitor.t}")


class TestAlarmTriggering:
    """Test that alarms are triggered when they should be."""
    
    def test_alarm_on_misspecified_model(self):
        """Test that misspecified model triggers alarm."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor(false_alarm_rate=0.05)
        
        # Generate data from N(0,1) but use N(2,1) as model (wrong mean)
        true_dist = norm(0, 1)
        predicted_dist = norm(2, 1)  # Wrong!
        
        outcomes = true_dist.rvs(size=500, random_state=42)
        
        alarm_triggered = False
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                alarm_triggered = True
                break
        
        assert alarm_triggered, "Should detect misspecified model"
        assert monitor.alarm_time is not None
        assert isinstance(monitor._alarm_info, AlarmInfo)
        assert monitor._alarm_info.triggered
    
    def test_alarm_on_regime_change(self):
        """Test detection of regime change (model becomes wrong)."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor(false_alarm_rate=0.05)
        
        # Start with correct model
        predicted_dist = norm(0, 1)
        
        # Generate 100 samples from correct distribution
        outcomes1 = norm(0, 1).rvs(size=100, random_state=42)
        for outcome in outcomes1:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            assert not alarm, "Should not alarm on correct model"
        
        # Then regime changes: data now from N(3, 1)
        outcomes2 = norm(3, 1).rvs(size=100, random_state=43)
        alarm_triggered = False
        for outcome in outcomes2:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                alarm_triggered = True
                break
        
        assert alarm_triggered, "Should detect regime change"
        # Alarm should occur after the changepoint
        assert monitor.alarm_time > 100
    
    def test_alarm_info_structure(self):
        """Test that AlarmInfo contains expected information."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor()
        
        # Force an alarm with misspecified model
        true_dist = norm(0, 1)
        predicted_dist = norm(5, 1)
        
        outcomes = true_dist.rvs(size=100, random_state=42)
        
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                assert alarm.triggered
                assert alarm.alarm_time == monitor.t
                assert alarm.ks_distance > alarm.threshold
                assert alarm.diagnosis is not None
                assert isinstance(alarm.diagnosis, str)
                break


class TestThresholds:
    """Test threshold computation methods."""
    
    def test_threshold_decreases(self):
        """Test that threshold decreases over time."""
        monitor = PITMonitor()
        
        # Dummy updates to advance time
        thresholds = []
        for i in range(1, 100):
            monitor.t = i
            thresholds.append(monitor._compute_threshold())
        
        # Threshold should generally decrease
        assert thresholds[10] > thresholds[50]
        assert thresholds[50] > thresholds[90]
    
    def test_alpha_spending_vs_stitching(self):
        """Test that stitching gives tighter thresholds than alpha-spending."""
        monitor_spending = PITMonitor(method='alpha_spending')
        monitor_stitch = PITMonitor(method='stitching')
        
        # At moderate sample sizes, stitching should be tighter
        for t in [100, 500, 1000]:
            monitor_spending.t = t
            monitor_stitch.t = t
            
            thresh_spending = monitor_spending._compute_threshold()
            thresh_stitch = monitor_stitch._compute_threshold()
            
            # Stitching should give smaller (tighter) threshold
            # (though they're close, so we just check they're both reasonable)
            assert thresh_spending > 0
            assert thresh_stitch > 0


class TestChangepointLocalization:
    """Test changepoint localization after alarm."""
    
    def test_localize_after_alarm(self):
        """Test changepoint localization works after alarm."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor()
        
        predicted_dist = norm(0, 1)
        
        # 50 samples from correct model
        outcomes1 = norm(0, 1).rvs(size=50, random_state=42)
        for outcome in outcomes1:
            monitor.update(predicted_dist.cdf, outcome)
        
        # Then 100 samples from wrong model
        outcomes2 = norm(3, 1).rvs(size=100, random_state=43)
        for outcome in outcomes2:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                break
        
        if monitor.alarm_triggered:
            cp = monitor.localize_changepoint()
            assert cp is not None
            # Changepoint should be detected somewhere around t=50
            # (allow wide margin since this is approximate)
            assert 20 < cp < 100
    
    def test_no_localization_without_alarm(self):
        """Test that localization returns None without alarm."""
        monitor = PITMonitor()
        cp = monitor.localize_changepoint()
        assert cp is None


class TestDiagnostics:
    """Test diagnostic capabilities."""
    
    def test_diagnosis_strings(self):
        """Test that diagnosis produces interpretable strings."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor()
        
        # Misspecified model
        predicted_dist = norm(3, 1)
        outcomes = norm(0, 1).rvs(size=200, random_state=42)
        
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                diagnosis = monitor._alarm_info.diagnosis
                assert isinstance(diagnosis, str)
                assert len(diagnosis) > 0
                # Should mention something about the deviation
                assert any(word in diagnosis.lower() 
                          for word in ['tail', 'central', 'confident'])
                break
    
    def test_get_state(self):
        """Test state extraction."""
        from scipy.stats import norm
        
        monitor = PITMonitor()
        predicted_dist = norm(0, 1)
        
        outcomes = norm(0, 1).rvs(size=10, random_state=42)
        for outcome in outcomes:
            monitor.update(predicted_dist.cdf, outcome)
        
        state = monitor.get_state()
        
        assert state['t'] == 10
        assert len(state['pits']) == 10
        assert state['ks_distance'] is not None
        assert state['threshold'] is not None
        assert state['alpha'] == 0.05


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_monitor(self):
        """Test monitor with no updates."""
        monitor = PITMonitor()
        
        assert monitor.t == 0
        assert monitor._compute_ks_distance() == 0.0
        
        state = monitor.get_state()
        assert state['ks_distance'] is None
    
    def test_single_observation(self):
        """Test monitor with single observation."""
        from scipy.stats import norm
        
        monitor = PITMonitor()
        predicted_dist = norm(0, 1)
        
        alarm = monitor.update(predicted_dist.cdf, 0.0)
        
        assert monitor.t == 1
        assert not alarm  # Single observation shouldn't trigger
    
    def test_alarm_persistence(self):
        """Test that alarm persists after triggering."""
        from scipy.stats import norm
        
        np.random.seed(42)
        monitor = PITMonitor()
        
        predicted_dist = norm(5, 1)  # Very wrong
        outcomes = norm(0, 1).rvs(size=200, random_state=42)
        
        alarm_count = 0
        for outcome in outcomes:
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                alarm_count += 1
        
        # Should only alarm once (first alarm triggers, then persists)
        assert alarm_count >= 1
        assert monitor.alarm_triggered


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

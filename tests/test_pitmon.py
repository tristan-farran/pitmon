import pytest
import numpy as np
from scipy.stats import norm, uniform
from pitmon import PITMonitor, AlarmInfo


class TestPITMonitor:
    """Test suite for PITMonitor class."""

    def test_initialization(self):
        """Test monitor initialization with valid parameters."""
        monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=50)
        assert monitor.alpha == 0.05
        assert monitor.baseline_size == 50
        assert monitor.t == 0
        assert not monitor.baseline_locked
        assert not monitor.alarm_triggered

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with pytest.raises(ValueError):
            PITMonitor(false_alarm_rate=-0.1)
        with pytest.raises(ValueError):
            PITMonitor(false_alarm_rate=1.5)
        with pytest.raises(ValueError):
            PITMonitor(baseline_size=0)
        with pytest.raises(ValueError):
            PITMonitor(changepoint_budget=-0.1)

    def test_baseline_warning_small_size(self):
        """Test warning for small baseline sizes."""
        with pytest.warns(UserWarning):
            PITMonitor(baseline_size=20)

    def test_well_calibrated_no_alarm(self):
        """Test that well-calibrated forecasts don't trigger alarm."""
        np.random.seed(42)
        monitor = PITMonitor(false_alarm_rate=0.05, baseline_size=30)

        # Generate well-calibrated forecasts
        for _ in range(100):
            predicted_dist = norm(loc=0, scale=1)
            outcome = np.random.normal(0, 1)
            alarm = monitor.update(predicted_dist.cdf, outcome)

        assert not monitor.alarm_triggered

    def test_baseline_establishment(self):
        """Test baseline establishment process."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=30)

        baseline_complete = False
        for i in range(30):
            predicted_dist = norm(loc=0, scale=1)
            outcome = np.random.normal(0, 1)
            alarm = monitor.update(predicted_dist.cdf, outcome)

            if i < 29:
                assert not alarm.baseline_complete
            else:
                assert alarm.baseline_complete
                baseline_complete = True

        assert baseline_complete
        assert monitor.baseline_locked
        assert len(monitor.baseline_pits) == 30

    def test_changepoint_detection(self):
        """Test that changepoint is detected when calibration changes."""
        np.random.seed(42)
        monitor = PITMonitor(false_alarm_rate=0.20, baseline_size=30)

        # Baseline: well-calibrated using PIT values
        for _ in range(30):
            monitor.update_pit(np.random.uniform(0, 1))

        assert monitor.baseline_locked

        # Change: Push extreme PIT values (clear miscalibration)
        # All values near 0 indicate systematic underestimation
        alarm_triggered = False
        for _ in range(50):
            monitor.update_pit(0.01)  # Extreme values
            if monitor.alarm_triggered:
                alarm_triggered = True
                break

        assert alarm_triggered
        assert monitor.alarm_triggered

    def test_pit_value_validation(self):
        """Test that invalid PIT values raise errors."""
        monitor = PITMonitor()

        def bad_cdf(x):
            return 1.5  # Invalid CDF returning >1

        with pytest.raises(ValueError):
            monitor.update(bad_cdf, 0.5)

    def test_update_pit_method(self):
        """Test the update_pit method with pre-computed PIT values."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=30)

        # Use pre-computed PIT values (should be uniform if calibrated)
        for _ in range(50):
            pit = np.random.uniform(0, 1)
            alarm = monitor.update_pit(pit)

        assert monitor.baseline_locked
        assert not monitor.alarm_triggered

    def test_get_state(self):
        """Test get_state method returns correct information."""
        monitor = PITMonitor(baseline_size=30)

        # Initial state
        state = monitor.get_state()
        assert state["t"] == 0
        assert not state["baseline_locked"]

        # After some updates
        np.random.seed(42)
        for _ in range(40):
            monitor.update_pit(np.random.uniform(0, 1))

        state = monitor.get_state()
        assert state["t"] == 40
        assert state["baseline_locked"]
        assert state["baseline_size"] == 30
        assert state["monitoring_size"] == 10

    def test_get_baseline_diagnostics(self):
        """Test baseline diagnostics reporting."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=30)

        # Before baseline complete
        diag = monitor.get_baseline_diagnostics()
        assert not diag["complete"]

        # Complete baseline
        for _ in range(30):
            monitor.update_pit(np.random.uniform(0, 1))

        diag = monitor.get_baseline_diagnostics()
        assert diag["complete"]
        assert "ks_from_uniform" in diag
        assert "quality" in diag

    def test_get_summary(self):
        """Test comprehensive summary method."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=20)

        # Collect baseline
        for _ in range(20):
            monitor.update_pit(np.random.uniform(0, 1))

        summary = monitor.get_summary()
        assert summary["status"] == "monitoring"
        assert summary["observations_processed"] == 20
        assert "baseline" in summary
        assert "monitoring" in summary

    def test_export_data(self):
        """Test data export functionality."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=20)

        for _ in range(30):
            monitor.update_pit(np.random.uniform(0, 1))

        data = monitor.export_data()
        assert "metadata" in data
        assert "baseline_pits" in data
        assert "monitoring_pits" in data
        assert len(data["baseline_pits"]) == 20
        assert len(data["monitoring_pits"]) == 10

    def test_localize_changepoint(self):
        """Test changepoint localization."""
        np.random.seed(456)
        monitor = PITMonitor(false_alarm_rate=0.10, baseline_size=30)

        # Baseline
        for _ in range(30):
            predicted_dist = norm(loc=0, scale=1)
            outcome = np.random.normal(0, 1)
            monitor.update(predicted_dist.cdf, outcome)

        # Monitoring with changepoint at t=40
        for _ in range(10):
            predicted_dist = norm(loc=0, scale=1)
            outcome = np.random.normal(0, 1)
            monitor.update(predicted_dist.cdf, outcome)

        # Change happens here
        for _ in range(50):
            predicted_dist = norm(loc=0, scale=1)
            outcome = np.random.normal(0, 5)  # Big change
            alarm = monitor.update(predicted_dist.cdf, outcome)
            if alarm:
                break

        if monitor.alarm_triggered:
            cp = monitor.localize_changepoint()
            assert cp is not None
            assert cp >= monitor.baseline_size

    def test_alarm_info_boolean(self):
        """Test that AlarmInfo evaluates to boolean correctly."""
        alarm_false = AlarmInfo(triggered=False)
        alarm_true = AlarmInfo(triggered=True)

        assert not alarm_false
        assert alarm_true

    def test_ks_two_sample_static(self):
        """Test two-sample KS distance computation."""
        pits1 = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        pits2 = np.array([0.2, 0.4, 0.6, 0.8])

        ks_dist = PITMonitor._compute_ks_two_sample(pits1, pits2)
        assert 0 <= ks_dist <= 1

        # Identical distributions should have KS=0
        ks_same = PITMonitor._compute_ks_two_sample(pits1, pits1)
        assert ks_same == 0

    def test_geometric_sequence(self):
        """Test geometric sequence generation."""
        seq = PITMonitor._geometric_sequence(1, 100, base=2)
        assert seq[0] == 1
        assert seq[-1] == 100
        assert all(seq[i] < seq[i + 1] for i in range(len(seq) - 1))

    def test_pits_property(self):
        """Test that pits property returns all PITs."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=20)

        for _ in range(30):
            monitor.update_pit(np.random.uniform(0, 1))

        all_pits = monitor.pits
        assert len(all_pits) == 30
        assert len(monitor.baseline_pits) == 20
        assert len(monitor.monitoring_pits) == 10

    def test_no_alarm_after_triggered(self):
        """Test that updates after alarm don't change alarm state."""
        np.random.seed(789)
        monitor = PITMonitor(false_alarm_rate=0.20, baseline_size=20)

        # Baseline
        for _ in range(20):
            monitor.update_pit(np.random.uniform(0, 1))

        # Trigger alarm
        for _ in range(100):
            monitor.update_pit(0.01)  # Extreme values
            if monitor.alarm_triggered:
                break

        first_alarm_time = monitor.alarm_time
        first_alarm_info = monitor._alarm_info

        # More updates shouldn't change alarm
        for _ in range(10):
            alarm = monitor.update_pit(np.random.uniform(0, 1))

        assert monitor.alarm_time == first_alarm_time
        assert monitor._alarm_info == first_alarm_info


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_observation_baseline(self):
        """Test with baseline_size=1 (minimal)."""
        monitor = PITMonitor(baseline_size=1)
        monitor.update_pit(0.5)
        assert monitor.baseline_locked

    def test_uniform_pits(self):
        """Test with perfectly uniform PITs."""
        monitor = PITMonitor(baseline_size=30)
        pits = np.linspace(0, 1, 30)
        np.random.shuffle(pits)

        for pit in pits:
            monitor.update_pit(pit)

        assert monitor.baseline_locked
        diag = monitor.get_baseline_diagnostics()
        assert diag["quality"] == "good"

    def test_extreme_miscalibration_baseline(self):
        """Test baseline with very poor calibration."""
        monitor = PITMonitor(baseline_size=30)

        # All PITs near 0 (very miscalibrated)
        with pytest.warns(UserWarning):
            for _ in range(30):
                monitor.update_pit(0.01)

        assert monitor.baseline_locked
        diag = monitor.get_baseline_diagnostics()
        assert diag["quality"] == "poor"


class TestPlotting:
    """Test plotting functionality."""

    def test_plot_diagnostics_no_data(self):
        """Test plotting with no data."""
        monitor = PITMonitor()
        fig = monitor.plot_diagnostics()
        assert fig is not None

    def test_plot_diagnostics_baseline_only(self):
        """Test plotting with only baseline data."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=30)

        for _ in range(30):
            monitor.update_pit(np.random.uniform(0, 1))

        fig = monitor.plot_diagnostics()
        assert fig is not None

    def test_plot_diagnostics_with_monitoring(self):
        """Test plotting with baseline and monitoring data."""
        np.random.seed(42)
        monitor = PITMonitor(baseline_size=30)

        for _ in range(60):
            monitor.update_pit(np.random.uniform(0, 1))

        fig = monitor.plot_diagnostics()
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

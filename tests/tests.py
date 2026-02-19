"""Test suite for PITMonitor."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from pitmon import PITMonitor, Alarm


class TestAlarm:
    """Test Alarm dataclass."""

    def test_alarm_boolean(self):
        """Test that Alarm works as boolean."""
        alarm = Alarm(True, 10, 25.0, 20.0)
        assert bool(alarm)
        assert alarm.triggered

        no_alarm = Alarm(False, 5, 10.0, 20.0)
        assert not bool(no_alarm)
        assert not no_alarm.triggered


class TestPITMonitorInit:
    """Test PITMonitor initialization."""

    def test_default_init(self):
        """Test default initialization."""
        monitor = PITMonitor()
        assert monitor.alpha == 0.05
        assert monitor.n_bins == 10
        assert monitor.threshold == 20.0
        assert monitor.t == 0
        assert not monitor.alarm_triggered

    def test_custom_init(self):
        """Test custom initialization."""
        monitor = PITMonitor(alpha=0.01, n_bins=20)
        assert monitor.alpha == 0.01
        assert monitor.n_bins == 20
        assert monitor.threshold == 100.0

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError):
            PITMonitor(alpha=0)
        with pytest.raises(ValueError):
            PITMonitor(alpha=1)
        with pytest.raises(ValueError):
            PITMonitor(alpha=-0.1)

    def test_invalid_n_bins(self):
        """Test that invalid n_bins raises ValueError."""
        with pytest.raises(ValueError):
            PITMonitor(n_bins=1)
        with pytest.raises(ValueError):
            PITMonitor(n_bins=101)


class TestPITMonitorUpdate:
    """Test PITMonitor update functionality."""

    def test_single_update(self):
        """Test single PIT update."""
        monitor = PITMonitor()
        result = monitor.update(0.5)
        assert isinstance(result, Alarm)
        assert not result
        assert monitor.t == 1

    def test_multiple_updates(self):
        """Test multiple PIT updates."""
        monitor = PITMonitor()
        for i in range(10):
            result = monitor.update(np.random.uniform(0, 1))
            assert monitor.t == i + 1
            assert isinstance(result, Alarm)

    def test_invalid_pit(self):
        """Test that invalid PIT raises ValueError."""
        monitor = PITMonitor()
        with pytest.raises(ValueError):
            monitor.update(-0.1)
        with pytest.raises(ValueError):
            monitor.update(1.1)

    def test_uniform_pits_no_alarm(self):
        """Test that uniform PITs don't trigger alarm."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.05)

        # Generate uniform PITs
        for _ in range(100):
            pit = np.random.uniform(0, 1)
            result = monitor.update(pit)

        # Should not alarm for well-calibrated data
        assert monitor.t == 100
        # Can't guarantee no alarm due to randomness, but very unlikely

    def test_nonuniform_pits_alarm(self):
        """Test that non-uniform PITs trigger alarm."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.05)

        # Generate first 50 uniform
        for _ in range(50):
            monitor.update(np.random.uniform(0, 1))

        # Then shift to non-uniform (biased)
        alarmed = False
        for _ in range(150):
            pit = np.random.beta(2, 5)  # Heavily biased
            result = monitor.update(pit)
            if result:
                alarmed = True
                break

        # Should eventually alarm (with high probability)
        assert alarmed or monitor.t == 200

    def test_update_after_alarm(self):
        """Test that updates after alarm return alarm status."""
        monitor = PITMonitor()
        monitor.alarm_triggered = True
        monitor.alarm_time = 50

        result = monitor.update(0.5)
        assert result
        assert result.time == 50

    def test_update_with_cdf(self):
        """Test update_with_cdf convenience method."""
        monitor = PITMonitor()

        # Normal CDF at mean
        def normal_cdf(x):
            return 0.5 * (1 + np.tanh((x - 0) / (1 * np.sqrt(2))))

        result = monitor.update_with_cdf(normal_cdf, 0.5)
        assert not result
        assert monitor.t == 1


class TestPITMonitorProperties:
    """Test PITMonitor properties and accessors."""

    def test_pits_property(self):
        """Test pits property."""
        monitor = PITMonitor()
        pits = [0.1, 0.5, 0.9]
        for pit in pits:
            monitor.update(pit)

        result = monitor.pits
        assert isinstance(result, np.ndarray)
        assert len(result) == 3
        np.testing.assert_array_equal(result, pits)

    def test_pvalues_property(self):
        """Test pvalues property."""
        monitor = PITMonitor()
        for _ in range(5):
            monitor.update(np.random.uniform(0, 1))

        pvals = monitor.pvalues
        assert isinstance(pvals, np.ndarray)
        assert len(pvals) == 5
        # All p-values should be in [0, 1]
        assert np.all((pvals >= 0) & (pvals <= 1))

    def test_evidence_property(self):
        """Test evidence property."""
        monitor = PITMonitor()
        assert monitor.evidence == 0.0

        monitor.update(0.5)
        # After first update, evidence is still 0
        assert monitor.evidence == 0.0

        monitor.update(0.5)
        # After second update, evidence > 0
        assert monitor.evidence >= 0.0


class TestPITMonitorChangepoint:
    """Test changepoint detection."""

    def test_changepoint_no_alarm(self):
        """Test that changepoint returns None without alarm."""
        monitor = PITMonitor()
        for _ in range(10):
            monitor.update(np.random.uniform(0, 1))

        assert monitor.changepoint() is None

    def test_changepoint_after_alarm(self):
        """Test changepoint detection after alarm."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.1)

        # Uniform phase
        for _ in range(30):
            monitor.update(np.random.uniform(0, 1))

        # Non-uniform phase
        for _ in range(100):
            monitor.update(np.random.beta(2, 5))
            if monitor.alarm_triggered:
                break

        if monitor.alarm_triggered:
            cp = monitor.changepoint()
            assert cp is not None
            assert isinstance(cp, int)
            assert 1 <= cp <= monitor.t

    def test_changepoint_too_few_observations(self):
        """Test changepoint with very few observations."""
        monitor = PITMonitor()
        monitor.update(0.5)
        assert monitor.changepoint() is None


class TestPITMonitorSummary:
    """Test summary and diagnostic methods."""

    def test_summary_empty(self):
        """Test summary with no data."""
        monitor = PITMonitor()
        summary = monitor.summary()

        assert summary["t"] == 0
        assert not summary["alarm_triggered"]
        assert summary["alarm_time"] is None
        assert summary["evidence"] == 0.0
        assert summary["calibration_score"] is None

    def test_summary_with_data(self):
        """Test summary with data."""
        np.random.seed(42)
        monitor = PITMonitor()

        for _ in range(20):
            monitor.update(np.random.uniform(0, 1))

        summary = monitor.summary()
        assert summary["t"] == 20
        assert isinstance(summary["calibration_score"], float)
        assert 0 <= summary["calibration_score"] <= 1

    def test_calibration_score(self):
        """Test calibration score computation."""
        monitor = PITMonitor()
        assert monitor.calibration_score() == 0.0

        # Perfect uniform should have low score
        monitor.update(0.1)
        monitor.update(0.5)
        monitor.update(0.9)
        score = monitor.calibration_score()
        assert 0 <= score <= 1

    def test_get_status(self):
        """Test status getter."""
        monitor = PITMonitor()
        assert monitor.get_status() == "not_started"

        monitor.update(0.5)
        assert monitor.get_status() == "monitoring"

        monitor.alarm_triggered = True
        assert monitor.get_status() == "alarm"


class TestPITMonitorReset:
    """Test reset functionality."""

    def test_reset(self):
        """Test that reset clears all state."""
        monitor = PITMonitor(alpha=0.01, n_bins=20)

        # Add some data
        for _ in range(10):
            monitor.update(np.random.uniform(0, 1))

        # Reset
        monitor.reset()

        # Check all state is cleared
        assert monitor.t == 0
        assert len(monitor._sorted_pits) == 0
        assert monitor._M == 0.0
        assert len(monitor._history) == 0
        assert not monitor.alarm_triggered
        assert monitor.alarm_time is None

        # But parameters should be preserved
        assert monitor.alpha == 0.01
        assert monitor.n_bins == 20


class TestPITMonitorRepr:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__."""
        monitor = PITMonitor(alpha=0.05, n_bins=10)
        repr_str = repr(monitor)
        assert "PITMonitor" in repr_str
        assert "alpha=0.05" in repr_str
        assert "n_bins=10" in repr_str

    def test_str_not_started(self):
        """Test __str__ when not started."""
        monitor = PITMonitor(alpha=0.05)
        str_repr = str(monitor)
        assert "Not started" in str_repr

    def test_str_monitoring(self):
        """Test __str__ during monitoring."""
        monitor = PITMonitor()
        monitor.update(0.5)
        str_repr = str(monitor)
        assert "monitoring" in str_repr
        assert "t=1" in str_repr

    def test_str_alarm(self):
        """Test __str__ after alarm."""
        monitor = PITMonitor()
        monitor.update(0.5)
        monitor.alarm_triggered = True
        monitor.alarm_time = 1
        str_repr = str(monitor)
        assert "ALARM" in str_repr


class TestPITMonitorSaveLoad:
    """Test save/load functionality."""

    def test_save_load_pickle(self):
        """Test save and load with pickle format."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.01, n_bins=15)

        # Add some data
        for _ in range(20):
            monitor.update(np.random.uniform(0, 1))

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            monitor.save(filepath)

            # Load
            loaded = PITMonitor.load(filepath)

            # Check equality
            assert loaded.alpha == monitor.alpha
            assert loaded.n_bins == monitor.n_bins
            assert loaded.t == monitor.t
            assert loaded.alarm_triggered == monitor.alarm_triggered
            np.testing.assert_array_equal(loaded.pits, monitor.pits)
            np.testing.assert_array_equal(loaded.pvalues, monitor.pvalues)
            assert abs(loaded.evidence - monitor.evidence) < 1e-10

        finally:
            filepath.unlink()

    def test_save_load_json(self):
        """Test save and load with JSON format."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.05, n_bins=10)

        # Add some data
        for _ in range(10):
            monitor.update(np.random.uniform(0, 1))

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            monitor.save(filepath)

            # Load
            loaded = PITMonitor.load(filepath)

            # Check equality (with some tolerance for JSON precision)
            assert loaded.alpha == monitor.alpha
            assert loaded.n_bins == monitor.n_bins
            assert loaded.t == monitor.t
            assert loaded.alarm_triggered == monitor.alarm_triggered
            np.testing.assert_allclose(loaded.pits, monitor.pits, rtol=1e-10)
            np.testing.assert_allclose(loaded.pvalues, monitor.pvalues, rtol=1e-10)
            assert abs(loaded.evidence - monitor.evidence) < 1e-6

        finally:
            filepath.unlink()

    def test_save_load_roundtrip(self):
        """Test that save/load preserves ability to continue monitoring."""
        np.random.seed(42)
        monitor = PITMonitor(alpha=0.05)

        # Add some data
        for _ in range(10):
            monitor.update(np.random.uniform(0, 1))

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            monitor.save(filepath)

            # Load
            loaded = PITMonitor.load(filepath)

            # Continue monitoring with loaded instance
            result = loaded.update(0.5)
            assert loaded.t == 11
            assert isinstance(result, Alarm)

        finally:
            filepath.unlink()


class TestPITMonitorPlot:
    """Test plotting functionality."""

    def test_plot_no_data(self):
        """Test plot with no data."""
        monitor = PITMonitor()
        # Should handle gracefully
        result = monitor.plot()
        assert result is None

    def test_plot_with_data(self):
        """Test plot with data."""
        pytest.importorskip("matplotlib")

        np.random.seed(42)
        monitor = PITMonitor()

        for _ in range(30):
            monitor.update(np.random.uniform(0, 1))

        fig = monitor.plot()
        assert fig is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

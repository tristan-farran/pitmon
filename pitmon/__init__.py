"""
PIT Monitor: Model-agnostic sequential validation via Probability Integral Transform

This package provides tools for monitoring probabilistic models in real-time
using the Probability Integral Transform (PIT) methodology.
"""

from .monitor import PITMonitor, AlarmInfo

__version__ = "0.1.0"
__all__ = ["PITMonitor", "AlarmInfo"]

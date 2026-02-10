"""
PIT Monitor: Model-agnostic sequential validation via Probability Integral Transform

A simple, principled tool for monitoring whether probabilistic models remain valid.
"""

from .monitor import PITMonitor, AlarmInfo

__version__ = "0.1.0"
__all__ = ["PITMonitor", "AlarmInfo"]

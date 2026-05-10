"""
research/signals/__init__.py
Public API for alpha-signal prototypes.
"""
from .mean_reversion import ZScoreMeanReversion
from .momentum import TimeSeriesMomentum

__all__ = ["ZScoreMeanReversion", "TimeSeriesMomentum"]

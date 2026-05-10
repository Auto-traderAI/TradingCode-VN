"""
research/data/__init__.py
Public API for the data-loading layer.
"""
from .loader import load_ohlcv, load_ohlcv_from_csv

__all__ = ["load_ohlcv", "load_ohlcv_from_csv"]

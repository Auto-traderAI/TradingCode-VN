"""
tests/conftest.py

Shared pytest fixtures for the Hermes research workspace test suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Raw (un-normalised) OHLCV — as it would arrive from a CSV file.
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_ohlcv_df() -> pd.DataFrame:
    """Return a 200-row raw OHLCV DataFrame with string datetime column."""
    n = 200
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame(
        {
            "datetime": idx.strftime("%Y-%m-%d"),
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Normalised OHLCV — DatetimeIndex(Asia/Ho_Chi_Minh), lowercase columns.
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_df(raw_ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Normalised OHLCV DataFrame produced by load_ohlcv."""
    from research.data import load_ohlcv

    return load_ohlcv(raw_ohlcv_df)


# ---------------------------------------------------------------------------
# OHLCV with ret_1 already computed (needed by several feature functions).
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_with_returns(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Normalised OHLCV DataFrame with ``ret_1`` log-return column."""
    from research.features import add_returns

    return add_returns(ohlcv_df.copy(), periods=[1])

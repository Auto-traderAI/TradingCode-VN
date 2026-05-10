"""
research/features/technical.py

Standard technical features derived from OHLCV data.
All functions accept a DataFrame with columns (open, high, low, close, volume)
and return the same DataFrame with new feature columns appended.

This is *research prototype* code — not production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(
    df: pd.DataFrame,
    col: str = "close",
    periods: list[int] | None = None,
) -> pd.DataFrame:
    """Add log-return columns for one or more look-back periods.

    Columns added: ``ret_<period>`` (e.g. ret_1, ret_5, ret_20).
    """
    periods = periods or [1, 5, 20]
    for p in periods:
        df[f"ret_{p}"] = np.log(df[col] / df[col].shift(p))
    return df


def add_volatility(
    df: pd.DataFrame,
    ret_col: str = "ret_1",
    windows: list[int] | None = None,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Add rolling realised volatility (std of log-returns × √periods_per_year).

    Requires ``ret_1`` column — call ``add_returns`` first.
    Columns added: ``vol_<window>``.

    Parameters
    ----------
    periods_per_year:
        Annualisation factor. Use 252 for daily bars, or
        252 × bars_per_day for intraday data (e.g. 252×16 for 1-min bars
        on a ~16-bar trading session).
    """
    if ret_col not in df.columns:
        raise KeyError(f"[technical] Column '{ret_col}' not found. Run add_returns first.")
    windows = windows or [5, 20, 60]
    for w in windows:
        df[f"vol_{w}"] = df[ret_col].rolling(w).std() * np.sqrt(periods_per_year)
    return df


def add_momentum(
    df: pd.DataFrame,
    col: str = "close",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add price momentum (current price / price n bars ago − 1).

    Columns added: ``mom_<window>``.
    """
    windows = windows or [5, 20, 60]
    for w in windows:
        df[f"mom_{w}"] = df[col] / df[col].shift(w) - 1
    return df

"""
research/data/loader.py

Utilities for loading VN30F OHLCV data into a standardised pandas DataFrame.

Assumptions
-----------
* Raw CSV layout (minimum required columns):
    datetime, open, high, low, close, volume
  Additional columns (e.g. open_interest, bid, ask) are preserved as-is.
* Datetime column is parsed and set as a timezone-aware index (Asia/Ho_Chi_Minh).
* All prices are in VND points (VN30F contract units).

Note: This module is a *research prototype* — not production-grade data infra.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path


_REQUIRED_COLS = {"datetime", "open", "high", "low", "close", "volume"}
_TZ = "Asia/Ho_Chi_Minh"


def _validate_columns(df: pd.DataFrame, source: str) -> None:
    missing = _REQUIRED_COLS - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"[loader] {source} is missing required columns: {missing}")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower().str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize(_TZ)
    else:
        df.index = df.index.tz_convert(_TZ)
    df = df.sort_index()
    return df


def load_ohlcv_from_csv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Parameters
    ----------
    path:
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex (Asia/Ho_Chi_Minh), columns: open, high, low, close, volume
        plus any extra columns present in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[loader] File not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, str(path))
    return _normalise(df)


def load_ohlcv(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Unified loader — accepts a file path or an already-loaded DataFrame.

    Parameters
    ----------
    source:
        * str / Path → delegates to ``load_ohlcv_from_csv``
        * pd.DataFrame → validates and normalises in-place (copy returned)

    Returns
    -------
    pd.DataFrame
        Standardised OHLCV DataFrame.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        _validate_columns(df, "DataFrame")
        return _normalise(df)
    return load_ohlcv_from_csv(source)

"""
research/features/microstructure.py

Market-microstructure features for VN30F intraday data.

References
----------
* Amihud (2002) — "Illiquidity and stock returns: cross-section and
  time-series effects", Journal of Financial Markets.
* Roll (1984) — "A simple implicit measure of the effective bid-ask
  spread in an efficient market", Journal of Finance.

This is *research prototype* code — not production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_spread_proxy(
    df: pd.DataFrame,
    col: str = "close",
    window: int = 20,
) -> pd.DataFrame:
    """Roll (1984) spread estimator.

    Spread ≈ 2 × √(−Cov(ΔP_t, ΔP_{t-1}))

    Estimated over a rolling ``window``.  NaN when covariance is positive
    (estimator is only valid under negative serial covariance).

    Column added: ``roll_spread``.
    """
    price_chg = df[col].diff()
    # Rolling covariance between Δp_t and Δp_{t-1}
    cov = price_chg.rolling(window).cov(price_chg.shift(1))
    spread = 2 * np.sqrt(np.maximum(-cov, 0))
    df["roll_spread"] = spread
    return df


def add_amihud_illiquidity(
    df: pd.DataFrame,
    ret_col: str = "ret_1",
    volume_col: str = "volume",
    window: int = 20,
    price_col: str = "close",
) -> pd.DataFrame:
    """Amihud (2002) illiquidity ratio.

    ILLIQ_t = |r_t| / (P_t × Volume_t)    (daily measure)
    Rolling mean over ``window`` bars.

    Requires ``ret_1`` column — call ``add_returns`` first.
    Column added: ``amihud``.
    """
    if ret_col not in df.columns:
        raise KeyError(f"[microstructure] Column '{ret_col}' not found. Run add_returns first.")

    dollar_volume = df[price_col] * df[volume_col]
    illiq = np.abs(df[ret_col]) / dollar_volume.replace(0, np.nan)
    df["amihud"] = illiq.rolling(window).mean()
    return df

"""
research/signals/momentum.py

Time-series momentum (TSMOM) alpha signal for VN30F.

Hypothesis
----------
Assets with positive past returns tend to continue rising over intermediate
horizons (1–12 months).  For intraday VN30F, we test shorter windows
(5–60 bars).

This is a *research prototype* — edge validity depends on regime and
transaction costs. Always walk-forward validate before drawing conclusions.

Reference
---------
Moskowitz, T., Ooi, Y.H., & Pedersen, L.H. (2012). "Time Series Momentum."
Journal of Financial Economics, 104(2), 228-250.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TimeSeriesMomentum:
    """Time-series momentum signal.

    Parameters
    ----------
    lookback:
        Return look-back window (bars).
    vol_window:
        Window for volatility scaling (default = lookback).
    col:
        Price column (default ``"close"``).
    vol_scale:
        If True, scale signal by inverse volatility (vol-targeting).
    """

    def __init__(
        self,
        lookback: int = 20,
        vol_window: int | None = None,
        col: str = "close",
        vol_scale: bool = True,
    ) -> None:
        self.lookback = lookback
        self.vol_window = vol_window or lookback
        self.col = col
        self.vol_scale = vol_scale

    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum signal.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional columns:
            * ``mom_ret``     — cumulative log-return over lookback
            * ``signal_raw``  — sign of mom_ret (+1 / -1)
            * ``signal``      — vol-scaled signal (if vol_scale=True) else raw
        """
        price = df[self.col]
        log_ret = np.log(price / price.shift(1))
        mom_ret = log_ret.rolling(self.lookback).sum()

        signal_raw = np.sign(mom_ret).fillna(0).astype(int)

        if self.vol_scale:
            vol = log_ret.rolling(self.vol_window).std() * np.sqrt(252)
            target_vol = 0.15  # annualised target volatility
            scale = (target_vol / vol).clip(0, 3)  # cap leverage at 3×
            signal = (signal_raw * scale).fillna(0)
        else:
            signal = signal_raw.astype(float)

        out = df.copy()
        out["mom_ret"] = mom_ret
        out["signal_raw"] = signal_raw
        out["signal"] = signal
        return out

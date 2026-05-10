"""
research/regime/volatility_regime.py

Simple volatility-based regime classifier.

Logic
-----
Classify each bar into a volatility regime by comparing rolling realised
volatility to its own long-run percentile thresholds.

Regimes
-------
* 0 — Low volatility  (vol < p33)
* 1 — Medium volatility (p33 ≤ vol < p67)
* 2 — High volatility  (vol ≥ p67)

This is *research prototype* code — not production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class VolatilityRegime:
    """Classify bars into volatility regimes using rolling percentile thresholds.

    Parameters
    ----------
    vol_window:
        Rolling window (bars) used to compute realised volatility.
    percentile_window:
        Expanding or rolling window to estimate percentile thresholds.
        Use ``None`` for full-history expanding window.
    low_pct, high_pct:
        Percentile thresholds separating low / medium / high vol regimes.
    """

    def __init__(
        self,
        vol_window: int = 20,
        percentile_window: int | None = None,
        low_pct: float = 33.0,
        high_pct: float = 67.0,
    ) -> None:
        self.vol_window = vol_window
        self.percentile_window = percentile_window
        self.low_pct = low_pct
        self.high_pct = high_pct

    # ------------------------------------------------------------------
    def fit_predict(
        self,
        df: pd.DataFrame,
        ret_col: str = "ret_1",
    ) -> pd.Series:
        """Compute volatility and assign regimes.

        Parameters
        ----------
        df:
            DataFrame containing ``ret_col``.
        ret_col:
            Column of log-returns.

        Returns
        -------
        pd.Series
            Integer regime labels (0 = low, 1 = medium, 2 = high),
            indexed like ``df``.
        """
        if ret_col not in df.columns:
            raise KeyError(f"[VolatilityRegime] Column '{ret_col}' not found.")

        vol = df[ret_col].rolling(self.vol_window).std() * np.sqrt(252)

        if self.percentile_window is None:
            low_thresh = vol.expanding().quantile(self.low_pct / 100)
            high_thresh = vol.expanding().quantile(self.high_pct / 100)
        else:
            low_thresh = vol.rolling(self.percentile_window).quantile(self.low_pct / 100)
            high_thresh = vol.rolling(self.percentile_window).quantile(self.high_pct / 100)

        regime = pd.Series(1, index=df.index, name="vol_regime", dtype=int)
        regime[vol < low_thresh] = 0
        regime[vol >= high_thresh] = 2

        return regime

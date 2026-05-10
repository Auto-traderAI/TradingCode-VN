"""
research/signals/mean_reversion.py

Z-score mean-reversion alpha signal for VN30F.

Hypothesis
----------
When the standardised price deviation from its rolling mean exceeds a
threshold (z > +k), the price is likely to revert downward (short signal).
Conversely when z < −k → long signal.

This is a *research prototype* — edge validity depends on regime and
transaction costs. Always walk-forward validate before drawing conclusions.

Reference
---------
Avellaneda & Lee (2010). "Statistical Arbitrage in the US Equities Market."
Quantitative Finance, 10(7), 761-782.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class ZScoreMeanReversion:
    """Rolling Z-score mean-reversion signal generator.

    Parameters
    ----------
    window:
        Look-back period for rolling mean and std.
    entry_z:
        Z-score threshold to trigger entry (absolute value).
    exit_z:
        Z-score threshold to trigger exit (absolute value, must be < entry_z).
    col:
        Price column to use (default ``"close"``).
    """

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        col: str = "close",
    ) -> None:
        if exit_z >= entry_z:
            raise ValueError("[ZScoreMeanReversion] exit_z must be < entry_z.")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.col = col

    # ------------------------------------------------------------------
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute z-scores and generate entry/exit signals.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with additional columns:
            * ``zscore``  — rolling z-score of price
            * ``signal``  — position direction: +1 (long), -1 (short), 0 (flat)
        """
        price = df[self.col]
        roll_mean = price.rolling(self.window).mean()
        roll_std = price.rolling(self.window).std()
        zscore = (price - roll_mean) / roll_std.replace(0, np.nan)

        # Vectorised signal generation
        # Entry: |z| >= entry_z; Exit: |z| <= exit_z
        signal = pd.Series(np.nan, index=df.index)
        signal[zscore <= -self.entry_z] = 1.0   # oversold → long
        signal[zscore >= self.entry_z] = -1.0   # overbought → short
        signal[np.abs(zscore) <= self.exit_z] = 0.0  # exit zone

        signal = signal.ffill().fillna(0).astype(int)

        out = df.copy()
        out["zscore"] = zscore
        out["signal"] = signal
        return out

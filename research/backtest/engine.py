"""
research/backtest/engine.py

Lightweight vectorised backtest engine for research prototypes.

Assumptions
-----------
* ``signal`` column contains position weights (-1, 0, +1 or fractional).
* Execution at next bar's open (1-bar delay to avoid look-ahead bias).
* Transaction cost modelled as a flat cost per unit of signal change.
* No position sizing beyond the signal weight itself.

This is *research prototype* code — not production execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe_ratio, max_drawdown, calmar_ratio


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "signal",
    price_col: str = "close",
    cost_per_trade: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Run a vectorised backtest.

    Parameters
    ----------
    df:
        DataFrame containing ``signal_col`` and ``price_col``.
    signal_col:
        Column name of position weights.
    price_col:
        Column used to compute bar returns.
    cost_per_trade:
        Transaction cost charged per unit |Δsignal| per bar.
    periods_per_year:
        Used for annualisation.

    Returns
    -------
    dict with keys:
        ``returns``      — pd.Series of strategy per-bar returns
        ``equity``       — cumulative equity curve (starting at 1.0)
        ``sharpe``       — annualised Sharpe ratio
        ``max_drawdown`` — maximum drawdown (negative)
        ``calmar``       — Calmar ratio
        ``n_trades``     — number of signal changes (proxy for trade count)
    """
    if signal_col not in df.columns:
        raise KeyError(f"[engine] Column '{signal_col}' not found.")

    bar_ret = np.log(df[price_col] / df[price_col].shift(1))

    # Position is entered at next bar (1-bar execution lag)
    position = df[signal_col].shift(1).fillna(0)

    # Gross strategy return
    strat_ret = position * bar_ret

    # Transaction costs (charged on position change)
    trade_cost = cost_per_trade * np.abs(position.diff().fillna(0))
    strat_ret = strat_ret - trade_cost

    equity = (1 + strat_ret).cumprod()
    n_trades = int((position.diff().abs() > 1e-9).sum())

    return {
        "returns": strat_ret,
        "equity": equity,
        "sharpe": sharpe_ratio(strat_ret, periods_per_year),
        "max_drawdown": max_drawdown(strat_ret),
        "calmar": calmar_ratio(strat_ret, periods_per_year),
        "n_trades": n_trades,
    }

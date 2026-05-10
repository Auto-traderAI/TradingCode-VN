"""
research/backtest/metrics.py

Standard performance metrics for backtest evaluation.

All metrics operate on a pd.Series of strategy returns (period returns,
not price levels).

This is *research prototype* code — not production.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    Parameters
    ----------
    returns:
        Series of period (e.g. daily) strategy returns.
    periods_per_year:
        Trading periods per year (252 for daily, 252×n_bars for intraday).
    risk_free:
        Annualised risk-free rate.

    Returns
    -------
    float
    """
    excess = returns - risk_free / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (negative number, e.g. -0.15 = 15% drawdown).

    Parameters
    ----------
    returns:
        Series of period strategy returns.

    Returns
    -------
    float
    """
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Calmar ratio = annualised return / |max drawdown|.

    Returns 0.0 if max drawdown is zero.
    """
    ann_ret = float((1 + returns).prod() ** (periods_per_year / len(returns)) - 1)
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd

"""
research/backtest/walkforward.py

Walk-forward validation utilities.

Walk-forward analysis is the gold standard for avoiding in-sample
overfitting in quantitative strategy research.  Each out-of-sample (OOS)
period is predicted only using information available up to that point.

This is *research prototype* code — not production.
"""
from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd

from .engine import run_backtest
from .metrics import sharpe_ratio, max_drawdown, calmar_ratio


def walk_forward_split(
    df: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = False,
) -> Generator[tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """Generate (train, test) DataFrame pairs for walk-forward validation.

    Parameters
    ----------
    df:
        Full dataset (chronologically sorted).
    train_size:
        Number of bars in each training window.
    test_size:
        Number of bars in each test (out-of-sample) window.
    step:
        Advance step between consecutive windows (default = test_size).
    expanding:
        If True, use an expanding training window instead of rolling.

    Yields
    ------
    (train_df, test_df) tuples.
    """
    step = step or test_size
    n = len(df)
    start = 0

    while start + train_size + test_size <= n:
        if expanding:
            train = df.iloc[:start + train_size]
        else:
            train = df.iloc[start: start + train_size]
        test = df.iloc[start + train_size: start + train_size + test_size]
        yield train, test
        start += step


def walk_forward_evaluate(
    df: pd.DataFrame,
    signal_func,
    train_size: int,
    test_size: int,
    step: int | None = None,
    expanding: bool = False,
    cost_per_trade: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Run walk-forward backtest and collect OOS performance metrics.

    Parameters
    ----------
    df:
        Full dataset.
    signal_func:
        Callable ``(train_df, test_df) → test_df_with_signal``.
        It receives the training DataFrame (for fitting) and the test
        DataFrame (for signal generation), and must return the test
        DataFrame augmented with a ``"signal"`` column.
    train_size, test_size, step, expanding:
        Passed to ``walk_forward_split``.
    cost_per_trade:
        Transaction cost per unit |Δsignal|.
    periods_per_year:
        Annualisation factor.

    Returns
    -------
    pd.DataFrame
        One row per OOS fold with columns:
        start, end, sharpe, max_drawdown, calmar, n_trades.
    """
    records = []

    for fold_idx, (train, test) in enumerate(
        walk_forward_split(df, train_size, test_size, step, expanding)
    ):
        test_with_signal = signal_func(train, test)
        result = run_backtest(
            test_with_signal,
            cost_per_trade=cost_per_trade,
            periods_per_year=periods_per_year,
        )
        records.append(
            {
                "fold": fold_idx,
                "start": test.index[0],
                "end": test.index[-1],
                "sharpe": result["sharpe"],
                "max_drawdown": result["max_drawdown"],
                "calmar": result["calmar"],
                "n_trades": result["n_trades"],
            }
        )

    summary = pd.DataFrame(records)
    return summary

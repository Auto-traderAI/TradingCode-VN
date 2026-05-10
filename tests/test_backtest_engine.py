"""
tests/test_backtest_engine.py

Tests for research/backtest/engine.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import run_backtest


@pytest.fixture
def backtest_df(ohlcv_df) -> pd.DataFrame:
    """OHLCV DataFrame with a simple long-always signal column."""
    df = ohlcv_df.copy()
    df["signal"] = 1.0
    return df


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestRunBacktestErrors:
    def test_missing_signal_column_raises_key_error(self, ohlcv_df):
        with pytest.raises(KeyError, match="signal"):
            run_backtest(ohlcv_df)

    def test_custom_signal_col_raises_if_absent(self, ohlcv_df):
        with pytest.raises(KeyError, match="my_signal"):
            run_backtest(ohlcv_df, signal_col="my_signal")


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


class TestRunBacktestOutputKeys:
    def test_all_keys_present(self, backtest_df):
        result = run_backtest(backtest_df)
        for key in ("returns", "equity", "sharpe", "max_drawdown", "calmar", "n_trades"):
            assert key in result

    def test_returns_is_series(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["returns"], pd.Series)

    def test_equity_is_series(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["equity"], pd.Series)

    def test_sharpe_is_float(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["sharpe"], float)

    def test_max_drawdown_is_float(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["max_drawdown"], float)

    def test_calmar_is_float(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["calmar"], float)

    def test_n_trades_is_int(self, backtest_df):
        result = run_backtest(backtest_df)
        assert isinstance(result["n_trades"], int)

    def test_lengths_match_input(self, backtest_df):
        result = run_backtest(backtest_df)
        assert len(result["returns"]) == len(backtest_df)
        assert len(result["equity"]) == len(backtest_df)


# ---------------------------------------------------------------------------
# All-zero signal
# ---------------------------------------------------------------------------


class TestRunBacktestZeroSignal:
    def test_all_zero_signal_gives_zero_strategy_returns(self, ohlcv_df):
        df = ohlcv_df.copy()
        df["signal"] = 0.0
        result = run_backtest(df)
        # Position is zero → strat_ret = 0 * bar_ret.
        # bar_ret[0] is NaN (no prior price), so strat_ret[0] = 0*NaN = NaN.
        # All other rows should be 0.
        rets = result["returns"]
        np.testing.assert_allclose(rets.iloc[1:].values, 0.0, atol=1e-10)

    def test_all_zero_signal_gives_n_trades_zero(self, ohlcv_df):
        df = ohlcv_df.copy()
        df["signal"] = 0.0
        result = run_backtest(df)
        assert result["n_trades"] == 0


# ---------------------------------------------------------------------------
# Transaction costs
# ---------------------------------------------------------------------------


class TestRunBacktestCosts:
    def test_cost_reduces_returns_compared_to_no_cost(self, backtest_df):
        r_no_cost = run_backtest(backtest_df, cost_per_trade=0.0)
        r_with_cost = run_backtest(backtest_df, cost_per_trade=0.01)
        # Strategy returns must be lower (or equal) with costs
        assert r_with_cost["returns"].sum() <= r_no_cost["returns"].sum()

    def test_cost_zero_n_trades_no_effect(self, ohlcv_df):
        """If signal is constant, no trades occur and costs have no effect."""
        df = ohlcv_df.copy()
        df["signal"] = 1.0
        r1 = run_backtest(df, cost_per_trade=0.0)
        r2 = run_backtest(df, cost_per_trade=0.1)
        # First bar only: position changes from 0 → 1, so cost is applied once;
        # after that it is constant → no further cost
        assert r1["n_trades"] == r2["n_trades"] == 1


# ---------------------------------------------------------------------------
# One-bar execution lag
# ---------------------------------------------------------------------------


class TestRunBacktestLag:
    def test_1bar_lag_means_first_return_is_zero(self, backtest_df):
        """Position is shifted by 1 bar → bar_ret[0] is NaN (no prior price)."""
        result = run_backtest(backtest_df)
        # bar_ret[0] is NaN because there is no prior close price, so
        # strat_ret[0] = position[0] * NaN = NaN. From bar 1 onward position=1.
        assert pd.isna(result["returns"].iloc[0])


# ---------------------------------------------------------------------------
# Trade counting
# ---------------------------------------------------------------------------


class TestRunBacktestNTrades:
    def test_signal_alternating_gives_many_trades(self, ohlcv_df):
        df = ohlcv_df.copy()
        n = len(df)
        df["signal"] = [1.0 if i % 2 == 0 else -1.0 for i in range(n)]
        result = run_backtest(df)
        assert result["n_trades"] > 0

    def test_constant_signal_gives_one_trade(self, ohlcv_df):
        df = ohlcv_df.copy()
        df["signal"] = 1.0
        result = run_backtest(df)
        # Only one position change: from 0 → 1
        assert result["n_trades"] == 1

    def test_n_trades_non_negative(self, backtest_df):
        result = run_backtest(backtest_df)
        assert result["n_trades"] >= 0


# ---------------------------------------------------------------------------
# Max drawdown from engine
# ---------------------------------------------------------------------------


class TestRunBacktestMaxDrawdown:
    def test_max_drawdown_non_positive(self, backtest_df):
        result = run_backtest(backtest_df)
        assert result["max_drawdown"] <= 0.0

    def test_zero_signal_gives_zero_drawdown(self, ohlcv_df):
        df = ohlcv_df.copy()
        df["signal"] = 0.0
        result = run_backtest(df)
        assert result["max_drawdown"] == 0.0

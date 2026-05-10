"""
tests/test_backtest_walkforward.py

Tests for research/backtest/walkforward.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import walk_forward_evaluate, walk_forward_split


@pytest.fixture
def long_ohlcv_df(ohlcv_df) -> pd.DataFrame:
    """200-bar OHLCV DataFrame — sufficient for multi-fold walk-forward."""
    return ohlcv_df


# ---------------------------------------------------------------------------
# walk_forward_split
# ---------------------------------------------------------------------------


class TestWalkForwardSplit:
    def test_yields_tuples_of_dataframes(self, long_ohlcv_df):
        splits = list(walk_forward_split(long_ohlcv_df, train_size=50, test_size=20))
        assert len(splits) > 0
        train, test = splits[0]
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_train_and_test_sizes_correct(self, long_ohlcv_df):
        for train, test in walk_forward_split(long_ohlcv_df, train_size=60, test_size=30):
            assert len(train) == 60
            assert len(test) == 30

    def test_no_overlap_between_train_and_test(self, long_ohlcv_df):
        for train, test in walk_forward_split(long_ohlcv_df, train_size=60, test_size=20):
            assert not train.index.isin(test.index).any()

    def test_number_of_folds_with_default_step(self, long_ohlcv_df):
        n = len(long_ohlcv_df)
        train_size, test_size = 50, 20
        splits = list(walk_forward_split(long_ohlcv_df, train_size=train_size, test_size=test_size))
        expected = (n - train_size) // test_size
        assert len(splits) == expected

    def test_custom_step(self, long_ohlcv_df):
        splits = list(
            walk_forward_split(long_ohlcv_df, train_size=50, test_size=20, step=10)
        )
        # Smaller step → more folds
        splits_default = list(
            walk_forward_split(long_ohlcv_df, train_size=50, test_size=20)
        )
        assert len(splits) >= len(splits_default)

    def test_expanding_window_grows(self, long_ohlcv_df):
        splits = list(
            walk_forward_split(long_ohlcv_df, train_size=50, test_size=20, expanding=True)
        )
        lengths = [len(train) for train, _ in splits]
        # Each consecutive train window should be at least as large as the previous
        assert all(lengths[i] <= lengths[i + 1] for i in range(len(lengths) - 1))

    def test_expanding_first_fold_size_equals_train_size(self, long_ohlcv_df):
        splits = list(
            walk_forward_split(long_ohlcv_df, train_size=50, test_size=20, expanding=True)
        )
        train0, _ = splits[0]
        assert len(train0) == 50

    def test_too_small_dataset_yields_nothing(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        df = pd.DataFrame({"close": np.ones(n)}, index=idx)
        splits = list(walk_forward_split(df, train_size=50, test_size=20))
        assert splits == []

    def test_test_periods_are_non_overlapping(self, long_ohlcv_df):
        splits = list(walk_forward_split(long_ohlcv_df, train_size=50, test_size=20))
        all_test_indices = [idx for _, test in splits for idx in test.index]
        # No duplicate test index values (unique partitioning of test periods)
        assert len(all_test_indices) == len(set(all_test_indices))


# ---------------------------------------------------------------------------
# walk_forward_evaluate
# ---------------------------------------------------------------------------


def _constant_long_signal_func(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Simple signal function that always goes long."""
    df = test.copy()
    df["signal"] = 1.0
    return df


class TestWalkForwardEvaluate:
    def test_returns_dataframe(self, long_ohlcv_df):
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, long_ohlcv_df):
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        for col in ("fold", "start", "end", "sharpe", "max_drawdown", "calmar", "n_trades"):
            assert col in result.columns

    def test_row_count_matches_fold_count(self, long_ohlcv_df):
        splits = list(walk_forward_split(long_ohlcv_df, train_size=50, test_size=20))
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        assert len(result) == len(splits)

    def test_fold_column_is_sequential(self, long_ohlcv_df):
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        assert list(result["fold"]) == list(range(len(result)))

    def test_sharpe_is_numeric(self, long_ohlcv_df):
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        assert result["sharpe"].dtype in (np.float64, np.float32, float)

    def test_empty_result_for_insufficient_data(self):
        n = 40
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        df = pd.DataFrame(
            {
                "open": np.ones(n),
                "high": np.ones(n),
                "low": np.ones(n),
                "close": np.ones(n),
                "volume": np.ones(n),
                "signal": np.zeros(n),
            },
            index=idx,
        )
        result = walk_forward_evaluate(
            df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_expanding_mode_accepted(self, long_ohlcv_df):
        result = walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=_constant_long_signal_func,
            train_size=50,
            test_size=20,
            expanding=True,
        )
        assert len(result) > 0

    def test_signal_func_receives_train_and_test(self, long_ohlcv_df):
        """Verify that train is separate from test inside signal_func."""
        seen = []

        def record_func(train, test):
            seen.append((len(train), len(test)))
            df = test.copy()
            df["signal"] = 0.0
            return df

        walk_forward_evaluate(
            long_ohlcv_df,
            signal_func=record_func,
            train_size=50,
            test_size=20,
        )
        for train_len, test_len in seen:
            assert train_len == 50
            assert test_len == 20

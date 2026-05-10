"""
tests/test_signals_mean_reversion.py

Tests for research/signals/mean_reversion.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signals import ZScoreMeanReversion


class TestZScoreMeanReversionInit:
    def test_exit_z_ge_entry_z_raises_value_error(self):
        with pytest.raises(ValueError, match="exit_z must be < entry_z"):
            ZScoreMeanReversion(entry_z=1.0, exit_z=1.0)

    def test_exit_z_greater_than_entry_z_raises_value_error(self):
        with pytest.raises(ValueError, match="exit_z must be < entry_z"):
            ZScoreMeanReversion(entry_z=1.0, exit_z=2.0)

    def test_valid_params_accepted(self):
        z = ZScoreMeanReversion(window=30, entry_z=2.5, exit_z=0.5)
        assert z.window == 30
        assert z.entry_z == 2.5
        assert z.exit_z == 0.5

    def test_defaults(self):
        z = ZScoreMeanReversion()
        assert z.window == 20
        assert z.entry_z == 2.0
        assert z.exit_z == 0.5
        assert z.col == "close"


class TestZScoreMeanReversionGenerate:
    def test_output_columns_added(self, ohlcv_df):
        result = ZScoreMeanReversion(window=10).generate(ohlcv_df)
        assert "zscore" in result.columns
        assert "signal" in result.columns

    def test_signal_values_in_minus_one_zero_one(self, ohlcv_df):
        result = ZScoreMeanReversion(window=10).generate(ohlcv_df)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_output_is_copy_not_original(self, ohlcv_df):
        original_cols = list(ohlcv_df.columns)
        ZScoreMeanReversion(window=10).generate(ohlcv_df)
        assert list(ohlcv_df.columns) == original_cols

    def test_no_nan_in_signal(self, ohlcv_df):
        result = ZScoreMeanReversion(window=10).generate(ohlcv_df)
        assert result["signal"].isna().sum() == 0

    def test_zscore_near_zero_at_rolling_mean(self):
        """Z-score at the rolling mean should be approximately 0."""
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        # Constant price → zscore undefined (std=0) → NaN
        df = pd.DataFrame({"close": [1000.0] * n}, index=idx)
        result = ZScoreMeanReversion(window=10).generate(df)
        # Constant price gives std=0 → NaN zscore; signal fills to 0
        assert result["signal"].isna().sum() == 0

    def test_extreme_low_zscore_gives_long_signal(self):
        """When z << -entry_z, signal should be +1 (long)."""
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = np.ones(n) * 1000.0
        # Introduce a sharp dip far below the mean
        price[-1] = 100.0
        df = pd.DataFrame({"close": price}, index=idx)
        result = ZScoreMeanReversion(window=20, entry_z=2.0, exit_z=0.5).generate(df)
        assert result["signal"].iloc[-1] == 1

    def test_extreme_high_zscore_gives_short_signal(self):
        """When z >> +entry_z, signal should be -1 (short)."""
        n = 60
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = np.ones(n) * 1000.0
        price[-1] = 5000.0
        df = pd.DataFrame({"close": price}, index=idx)
        result = ZScoreMeanReversion(window=20, entry_z=2.0, exit_z=0.5).generate(df)
        assert result["signal"].iloc[-1] == -1

    def test_custom_col(self, ohlcv_df):
        result = ZScoreMeanReversion(window=10, col="open").generate(ohlcv_df)
        assert "zscore" in result.columns

    def test_row_count_preserved(self, ohlcv_df):
        result = ZScoreMeanReversion(window=10).generate(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_signal_starts_at_zero_with_no_history(self):
        """Before the rolling window, zscore is NaN → signal should be 0 after ffill."""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = 1000 + np.arange(n, dtype=float)
        df = pd.DataFrame({"close": price}, index=idx)
        result = ZScoreMeanReversion(window=20, entry_z=2.0, exit_z=0.5).generate(df)
        # No signal before window fills
        assert result["signal"].iloc[0] == 0

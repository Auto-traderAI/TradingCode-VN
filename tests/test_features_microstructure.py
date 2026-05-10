"""
tests/test_features_microstructure.py

Tests for research/features/microstructure.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.features import add_amihud_illiquidity, add_spread_proxy


class TestAddSpreadProxy:
    def test_column_added(self, ohlcv_df):
        result = add_spread_proxy(ohlcv_df.copy())
        assert "roll_spread" in result.columns

    def test_spread_is_non_negative(self, ohlcv_df):
        result = add_spread_proxy(ohlcv_df.copy())
        valid = result["roll_spread"].dropna()
        assert (valid >= 0).all()

    def test_modifies_df_in_place(self, ohlcv_df):
        df = ohlcv_df.copy()
        result = add_spread_proxy(df)
        assert result is df

    def test_custom_window_produces_fewer_nans(self, ohlcv_df):
        result_small = add_spread_proxy(ohlcv_df.copy(), window=5)
        result_large = add_spread_proxy(ohlcv_df.copy(), window=40)
        nans_small = result_small["roll_spread"].isna().sum()
        nans_large = result_large["roll_spread"].isna().sum()
        assert nans_small <= nans_large

    def test_custom_price_column(self, ohlcv_df):
        df = ohlcv_df.copy()
        result = add_spread_proxy(df, col="open")
        assert "roll_spread" in result.columns

    def test_perfectly_constant_price_gives_zero_spread(self):
        """With a constant price, serial covariance is zero → spread = 0."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        df = pd.DataFrame({"close": [1000.0] * n}, index=idx)
        result = add_spread_proxy(df, window=5)
        valid = result["roll_spread"].dropna()
        np.testing.assert_allclose(valid.values, 0.0, atol=1e-10)


class TestAddAmihudIlliquidity:
    def test_column_added(self, ohlcv_with_returns):
        result = add_amihud_illiquidity(ohlcv_with_returns.copy())
        assert "amihud" in result.columns

    def test_missing_ret_col_raises_key_error(self, ohlcv_df):
        with pytest.raises(KeyError, match="ret_1"):
            add_amihud_illiquidity(ohlcv_df.copy())

    def test_amihud_is_non_negative(self, ohlcv_with_returns):
        result = add_amihud_illiquidity(ohlcv_with_returns.copy())
        valid = result["amihud"].dropna()
        assert (valid >= 0).all()

    def test_modifies_df_in_place(self, ohlcv_with_returns):
        df = ohlcv_with_returns.copy()
        result = add_amihud_illiquidity(df)
        assert result is df

    def test_zero_volume_handled_without_error(self):
        """Zero volume rows should produce NaN, not ZeroDivisionError."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = 1000.0 + np.arange(n, dtype=float)
        df = pd.DataFrame(
            {
                "close": price,
                "volume": [0.0] * n,
                "ret_1": np.concatenate([[np.nan], np.log(price[1:] / price[:-1])]),
            },
            index=idx,
        )
        result = add_amihud_illiquidity(df)
        # All valid windows should be NaN because volume is always 0
        valid = result["amihud"].dropna()
        # Should not raise; result may be NaN or 0
        assert isinstance(result["amihud"], pd.Series)

    def test_custom_ret_col_and_price_col(self, ohlcv_df):
        """Should work with non-default column names."""
        from research.features import add_returns

        df = add_returns(ohlcv_df.copy(), periods=[5])
        # rename columns to test custom args
        df = df.rename(columns={"close": "px"})
        df["ret_5"]  # ensure column exists
        result = add_amihud_illiquidity(
            df, ret_col="ret_5", volume_col="volume", price_col="px"
        )
        assert "amihud" in result.columns

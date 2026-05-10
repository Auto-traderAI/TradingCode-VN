"""
tests/test_features_technical.py

Tests for research/features/technical.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.features import add_momentum, add_returns, add_volatility


class TestAddReturns:
    def test_default_periods_added(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy())
        for p in (1, 5, 20):
            assert f"ret_{p}" in result.columns

    def test_custom_periods_added(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy(), periods=[3, 10])
        assert "ret_3" in result.columns
        assert "ret_10" in result.columns
        assert "ret_1" not in result.columns

    def test_returns_modifies_df_in_place(self, ohlcv_df):
        df = ohlcv_df.copy()
        result = add_returns(df, periods=[1])
        assert result is df

    def test_ret_1_first_row_is_nan(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy(), periods=[1])
        assert pd.isna(result["ret_1"].iloc[0])

    def test_log_return_formula_correctness(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy(), periods=[1])
        close = ohlcv_df["close"]
        expected = float(np.log(close.iloc[1] / close.iloc[0]))
        assert abs(result["ret_1"].iloc[1] - expected) < 1e-10

    def test_period5_return_formula_correctness(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy(), periods=[5])
        close = ohlcv_df["close"]
        expected = float(np.log(close.iloc[5] / close.iloc[0]))
        assert abs(result["ret_5"].iloc[5] - expected) < 1e-10

    def test_zero_nan_count_after_first_period_rows(self, ohlcv_df):
        result = add_returns(ohlcv_df.copy(), periods=[1])
        # Row 1 onwards should be non-NaN
        assert result["ret_1"].iloc[1:].notna().all()


class TestAddVolatility:
    def test_default_vol_columns_added(self, ohlcv_with_returns):
        result = add_volatility(ohlcv_with_returns.copy())
        for w in (5, 20, 60):
            assert f"vol_{w}" in result.columns

    def test_custom_windows_added(self, ohlcv_with_returns):
        result = add_volatility(ohlcv_with_returns.copy(), windows=[10])
        assert "vol_10" in result.columns
        assert "vol_5" not in result.columns

    def test_missing_ret_col_raises_key_error(self, ohlcv_df):
        with pytest.raises(KeyError, match="ret_1"):
            add_volatility(ohlcv_df.copy())

    def test_volatility_is_non_negative(self, ohlcv_with_returns):
        result = add_volatility(ohlcv_with_returns.copy(), windows=[5])
        valid = result["vol_5"].dropna()
        assert (valid >= 0).all()

    def test_volatility_modifies_df_in_place(self, ohlcv_with_returns):
        df = ohlcv_with_returns.copy()
        result = add_volatility(df, windows=[5])
        assert result is df

    def test_annualisation_factor_scales_result(self, ohlcv_with_returns):
        """Doubling periods_per_year should increase vol by sqrt(2)."""
        v1 = add_volatility(ohlcv_with_returns.copy(), windows=[20], periods_per_year=252)
        v2 = add_volatility(ohlcv_with_returns.copy(), windows=[20], periods_per_year=504)
        ratio = (v2["vol_20"] / v1["vol_20"]).dropna()
        np.testing.assert_allclose(ratio, np.sqrt(2), rtol=1e-6)

    def test_custom_ret_col(self, ohlcv_df):
        """add_volatility should work with an alternative returns column."""
        df = add_returns(ohlcv_df.copy(), periods=[5])
        result = add_volatility(df, ret_col="ret_5", windows=[5])
        assert "vol_5" in result.columns


class TestAddMomentum:
    def test_default_mom_columns_added(self, ohlcv_df):
        result = add_momentum(ohlcv_df.copy())
        for w in (5, 20, 60):
            assert f"mom_{w}" in result.columns

    def test_custom_windows_added(self, ohlcv_df):
        result = add_momentum(ohlcv_df.copy(), windows=[10, 30])
        assert "mom_10" in result.columns
        assert "mom_30" in result.columns
        assert "mom_5" not in result.columns

    def test_momentum_formula_correctness(self, ohlcv_df):
        result = add_momentum(ohlcv_df.copy(), windows=[5])
        close = ohlcv_df["close"]
        expected = float(close.iloc[5] / close.iloc[0] - 1)
        assert abs(result["mom_5"].iloc[5] - expected) < 1e-10

    def test_momentum_modifies_df_in_place(self, ohlcv_df):
        df = ohlcv_df.copy()
        result = add_momentum(df)
        assert result is df

    def test_first_n_rows_are_nan(self, ohlcv_df):
        result = add_momentum(ohlcv_df.copy(), windows=[5])
        assert pd.isna(result["mom_5"].iloc[0])
        assert pd.isna(result["mom_5"].iloc[4])
        assert pd.notna(result["mom_5"].iloc[5])

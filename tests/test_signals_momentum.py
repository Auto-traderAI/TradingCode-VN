"""
tests/test_signals_momentum.py

Tests for research/signals/momentum.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.signals import TimeSeriesMomentum


class TestTimeSeriesMomentumInit:
    def test_default_vol_window_equals_lookback(self):
        sig = TimeSeriesMomentum(lookback=15)
        assert sig.vol_window == 15

    def test_explicit_vol_window_accepted(self):
        sig = TimeSeriesMomentum(lookback=10, vol_window=30)
        assert sig.vol_window == 30

    def test_defaults(self):
        sig = TimeSeriesMomentum()
        assert sig.lookback == 20
        assert sig.col == "close"
        assert sig.vol_scale is True
        assert sig.periods_per_year == 252


class TestTimeSeriesMomentumGenerate:
    def test_output_columns_vol_scale_on(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=10).generate(ohlcv_df)
        for col in ("mom_ret", "signal_raw", "signal"):
            assert col in result.columns

    def test_output_columns_vol_scale_off(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=10, vol_scale=False).generate(ohlcv_df)
        for col in ("mom_ret", "signal_raw", "signal"):
            assert col in result.columns

    def test_signal_raw_values_in_minus_one_zero_one(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=10, vol_scale=False).generate(ohlcv_df)
        valid = result["signal_raw"].dropna()
        assert set(valid.unique()).issubset({-1, 0, 1})

    def test_signal_vol_scale_off_equals_signal_raw(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=10, vol_scale=False).generate(ohlcv_df)
        pd.testing.assert_series_equal(
            result["signal"].astype(float),
            result["signal_raw"].astype(float),
            check_names=False,
        )

    def test_vol_scale_caps_leverage(self, ohlcv_df):
        """Vol-scaled signal should never exceed 3× (cap in momentum.py)."""
        result = TimeSeriesMomentum(lookback=5, vol_scale=True).generate(ohlcv_df)
        assert (result["signal"].abs() <= 3 + 1e-9).all()

    def test_output_is_copy_not_original(self, ohlcv_df):
        original_cols = list(ohlcv_df.columns)
        TimeSeriesMomentum(lookback=5).generate(ohlcv_df)
        assert list(ohlcv_df.columns) == original_cols

    def test_first_lookback_rows_signal_raw_is_zero(self, ohlcv_df):
        lookback = 15
        result = TimeSeriesMomentum(lookback=lookback, vol_scale=False).generate(ohlcv_df)
        # First (lookback-1) rows have no complete window → mom_ret NaN → signal_raw 0
        assert (result["signal_raw"].iloc[:lookback - 1] == 0).all()

    def test_custom_col_parameter(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=5, col="open", vol_scale=False).generate(ohlcv_df)
        assert "signal" in result.columns

    def test_no_nan_in_signal_after_generate(self, ohlcv_df):
        result = TimeSeriesMomentum(lookback=10).generate(ohlcv_df)
        assert result["signal"].isna().sum() == 0

    def test_trending_up_gives_positive_signal(self):
        """Strongly trending-up prices should yield positive raw signal after lookback."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = pd.Series(np.linspace(1000, 1200, n), index=idx, name="close")
        df = pd.DataFrame({"close": price})
        result = TimeSeriesMomentum(lookback=10, vol_scale=False).generate(df)
        # After the lookback warm-up, all signals should be +1
        tail = result["signal_raw"].iloc[10:]
        assert (tail == 1).all()

    def test_trending_down_gives_negative_signal(self):
        """Strongly trending-down prices should yield negative raw signal after lookback."""
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        price = pd.Series(np.linspace(1200, 1000, n), index=idx, name="close")
        df = pd.DataFrame({"close": price})
        result = TimeSeriesMomentum(lookback=10, vol_scale=False).generate(df)
        tail = result["signal_raw"].iloc[10:]
        assert (tail == -1).all()

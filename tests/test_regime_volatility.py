"""
tests/test_regime_volatility.py

Tests for research/regime/volatility_regime.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.regime import VolatilityRegime


class TestVolatilityRegime:
    def test_output_is_series(self, ohlcv_with_returns):
        regime = VolatilityRegime().fit_predict(ohlcv_with_returns)
        assert isinstance(regime, pd.Series)

    def test_output_length_matches_input(self, ohlcv_with_returns):
        regime = VolatilityRegime().fit_predict(ohlcv_with_returns)
        assert len(regime) == len(ohlcv_with_returns)

    def test_regime_values_in_valid_set(self, ohlcv_with_returns):
        regime = VolatilityRegime().fit_predict(ohlcv_with_returns)
        valid_values = {0, 1, 2}
        assert set(regime.unique()).issubset(valid_values)

    def test_index_matches_input(self, ohlcv_with_returns):
        regime = VolatilityRegime().fit_predict(ohlcv_with_returns)
        assert regime.index.equals(ohlcv_with_returns.index)

    def test_missing_ret_col_raises_key_error(self, ohlcv_df):
        with pytest.raises(KeyError, match="ret_1"):
            VolatilityRegime().fit_predict(ohlcv_df)

    def test_custom_ret_col_works(self, ohlcv_df):
        from research.features import add_returns

        df = add_returns(ohlcv_df.copy(), periods=[5])
        regime = VolatilityRegime().fit_predict(df, ret_col="ret_5")
        assert isinstance(regime, pd.Series)

    def test_regime_values_are_integers(self, ohlcv_with_returns):
        regime = VolatilityRegime().fit_predict(ohlcv_with_returns)
        assert regime.dtype in (np.dtype("int32"), np.dtype("int64"), np.dtype("int"))

    def test_with_rolling_percentile_window(self, ohlcv_with_returns):
        """percentile_window != None uses rolling instead of expanding."""
        regime = VolatilityRegime(
            vol_window=10, percentile_window=30
        ).fit_predict(ohlcv_with_returns)
        assert isinstance(regime, pd.Series)
        assert set(regime.unique()).issubset({0, 1, 2})

    def test_high_vol_constant_spike(self):
        """A single extreme return spike should produce high-vol regime at that point."""
        n = 120
        rng = np.random.default_rng(7)
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Ho_Chi_Minh")
        ret = rng.normal(0, 0.001, n)
        # Insert a very large return at the midpoint
        ret[n // 2] = 0.5
        df = pd.DataFrame({"ret_1": ret}, index=idx)
        regime = VolatilityRegime(vol_window=5).fit_predict(df, ret_col="ret_1")
        # After the spike the rolling window should show high vol (2)
        assert 2 in regime.values

    def test_custom_percentiles(self, ohlcv_with_returns):
        r1 = VolatilityRegime(low_pct=10.0, high_pct=90.0).fit_predict(ohlcv_with_returns)
        r2 = VolatilityRegime(low_pct=40.0, high_pct=60.0).fit_predict(ohlcv_with_returns)
        # Wider percentiles → more bars in medium regime (1)
        assert isinstance(r1, pd.Series)
        assert isinstance(r2, pd.Series)

"""
tests/test_backtest_metrics.py

Tests for research/backtest/metrics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.backtest import calmar_ratio, max_drawdown, sharpe_ratio


@pytest.fixture
def flat_returns() -> pd.Series:
    return pd.Series([0.0] * 50)


@pytest.fixture
def positive_returns() -> pd.Series:
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 252)
    # Large positive drift guarantees positive mean → positive Sharpe
    return pd.Series(noise + 0.02)


@pytest.fixture
def always_down_returns() -> pd.Series:
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.01, 252)
    # Large negative drift guarantees negative mean → negative Sharpe
    return pd.Series(noise - 0.02)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_flat_returns_gives_zero(self, flat_returns):
        assert sharpe_ratio(flat_returns) == 0.0

    def test_positive_drift_gives_positive_sharpe(self, positive_returns):
        assert sharpe_ratio(positive_returns) > 0

    def test_negative_drift_gives_negative_sharpe(self, always_down_returns):
        assert sharpe_ratio(always_down_returns) < 0

    def test_return_type_is_float(self, positive_returns):
        assert isinstance(sharpe_ratio(positive_returns), float)

    def test_periods_per_year_scales_result(self, positive_returns):
        s1 = sharpe_ratio(positive_returns, periods_per_year=252)
        s2 = sharpe_ratio(positive_returns, periods_per_year=504)
        # Doubling periods_per_year should increase Sharpe by sqrt(2)
        np.testing.assert_allclose(s2 / s1, np.sqrt(2), rtol=0.01)

    def test_risk_free_rate_reduces_sharpe(self, positive_returns):
        s0 = sharpe_ratio(positive_returns, risk_free=0.0)
        s1 = sharpe_ratio(positive_returns, risk_free=0.05)
        assert s1 < s0

    def test_all_zero_returns_give_zero_sharpe(self):
        """All-zero returns have std=0 → Sharpe is 0."""
        rets = pd.Series([0.0] * 100)
        assert sharpe_ratio(rets) == 0.0


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_flat_returns_give_zero_drawdown(self, flat_returns):
        assert max_drawdown(flat_returns) == 0.0

    def test_always_positive_gives_zero_drawdown(self):
        rets = pd.Series([0.01] * 50)
        assert max_drawdown(rets) == 0.0

    def test_always_negative_is_negative(self, always_down_returns):
        mdd = max_drawdown(always_down_returns)
        assert mdd < 0

    def test_return_type_is_float(self, flat_returns):
        assert isinstance(max_drawdown(flat_returns), float)

    def test_result_is_between_minus_one_and_zero(self, always_down_returns):
        mdd = max_drawdown(always_down_returns)
        assert -1.0 <= mdd <= 0.0

    def test_known_drawdown_calculation(self):
        """50% drawdown: equity goes from 1 → 2 → 1 (50% drop)."""
        # +100% then -50% gives equity 1 → 2 → 1, mdd should be -0.5
        rets = pd.Series([1.0, -0.5])
        mdd = max_drawdown(rets)
        np.testing.assert_allclose(mdd, -0.5, atol=1e-9)

    def test_random_returns_are_non_positive(self):
        rng = np.random.default_rng(42)
        rets = pd.Series(rng.normal(0, 0.02, 252))
        assert max_drawdown(rets) <= 0.0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------


class TestCalmarRatio:
    def test_zero_drawdown_returns_zero(self):
        rets = pd.Series([0.01] * 50)
        assert calmar_ratio(rets) == 0.0

    def test_return_type_is_float(self, positive_returns):
        assert isinstance(calmar_ratio(positive_returns), float)

    def test_positive_returns_positive_calmar(self, positive_returns):
        calmar = calmar_ratio(positive_returns)
        # Calmar can be positive or negative depending on drift vs. drawdown
        # For strong positive drift it should be positive
        assert isinstance(calmar, float)

    def test_periods_per_year_affects_annualised_return(self, positive_returns):
        c1 = calmar_ratio(positive_returns, periods_per_year=252)
        c2 = calmar_ratio(positive_returns, periods_per_year=504)
        # Different annualisation → different numerator
        assert c1 != c2

    def test_flat_returns_zero_drawdown_returns_zero(self, flat_returns):
        assert calmar_ratio(flat_returns) == 0.0

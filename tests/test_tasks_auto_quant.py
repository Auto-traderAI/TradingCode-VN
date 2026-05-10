"""
tests/test_tasks_auto_quant.py

Tests for research/tasks/auto_quant_research.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.tasks import AutoResearchConfig, run_auto_quant_research
from research.tasks.auto_quant_research import _prepare_features, _strategy_score


# ---------------------------------------------------------------------------
# Helper: minimal OHLCV DataFrame accepted by load_ohlcv
# ---------------------------------------------------------------------------


@pytest.fixture
def small_ohlcv_df() -> pd.DataFrame:
    """200-row raw OHLCV DataFrame for faster test runs.

    The ``datetime`` column is required by ``load_ohlcv``.
    """
    n = 200
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 1000.0 + np.cumsum(rng.normal(0, 5, n))
    return pd.DataFrame(
        {
            "datetime": idx.strftime("%Y-%m-%d"),
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": rng.integers(1000, 5000, n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# AutoResearchConfig defaults
# ---------------------------------------------------------------------------


class TestAutoResearchConfig:
    def test_default_momentum_lookbacks(self):
        cfg = AutoResearchConfig()
        assert cfg.momentum_lookbacks == [10, 20, 40]

    def test_default_meanrev_windows(self):
        cfg = AutoResearchConfig()
        assert cfg.meanrev_windows == [20, 40]

    def test_default_meanrev_entry_z(self):
        cfg = AutoResearchConfig()
        assert cfg.meanrev_entry_z == [1.5, 2.0, 2.5]

    def test_default_scoring_weights(self):
        cfg = AutoResearchConfig()
        assert cfg.weight_sharpe == 1.0
        assert cfg.weight_calmar == 0.5
        assert cfg.weight_drawdown == 1.0

    def test_default_regime_filter_is_true(self):
        assert AutoResearchConfig().regime_filter is True

    def test_custom_weights_accepted(self):
        cfg = AutoResearchConfig(weight_sharpe=2.0, weight_calmar=0.0)
        assert cfg.weight_sharpe == 2.0
        assert cfg.weight_calmar == 0.0


# ---------------------------------------------------------------------------
# _prepare_features
# ---------------------------------------------------------------------------


class TestPrepareFeatures:
    def test_adds_expected_columns(self, small_ohlcv_df):
        result = _prepare_features(small_ohlcv_df)
        for col in ("ret_1", "ret_5", "ret_20", "vol_20", "vol_60", "mom_5", "mom_20", "mom_60"):
            assert col in result.columns

    def test_returns_copy(self, small_ohlcv_df):
        result = _prepare_features(small_ohlcv_df)
        assert "ret_1" not in small_ohlcv_df.columns  # original untouched


# ---------------------------------------------------------------------------
# _strategy_score
# ---------------------------------------------------------------------------


class TestStrategyScore:
    def test_score_increases_with_sharpe(self):
        cfg = AutoResearchConfig(weight_sharpe=1.0, weight_calmar=0.0, weight_drawdown=0.0)
        m1 = {"sharpe": 1.0, "calmar": 0.0, "max_drawdown": 0.0}
        m2 = {"sharpe": 2.0, "calmar": 0.0, "max_drawdown": 0.0}
        assert _strategy_score(m2, cfg) > _strategy_score(m1, cfg)

    def test_larger_drawdown_reduces_score(self):
        cfg = AutoResearchConfig(weight_sharpe=0.0, weight_calmar=0.0, weight_drawdown=1.0)
        m1 = {"sharpe": 0.0, "calmar": 0.0, "max_drawdown": -0.1}
        m2 = {"sharpe": 0.0, "calmar": 0.0, "max_drawdown": -0.5}
        assert _strategy_score(m1, cfg) > _strategy_score(m2, cfg)


# ---------------------------------------------------------------------------
# run_auto_quant_research — output structure
# ---------------------------------------------------------------------------


class TestRunAutoQuantResearch:
    @pytest.fixture
    def result(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[10],
            meanrev_windows=[20],
            meanrev_entry_z=[2.0],
            regime_filter=False,
        )
        return run_auto_quant_research(small_ohlcv_df, config=cfg)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_leaderboard_key(self, result):
        assert "leaderboard" in result

    def test_has_best_strategy_key(self, result):
        assert "best_strategy" in result

    def test_has_best_params_key(self, result):
        assert "best_params" in result

    def test_has_best_signal_frame_key(self, result):
        assert "best_signal_frame" in result

    def test_leaderboard_is_dataframe(self, result):
        assert isinstance(result["leaderboard"], pd.DataFrame)

    def test_leaderboard_columns(self, result):
        expected = {"strategy", "score", "sharpe", "calmar", "max_drawdown", "n_trades"}
        assert expected.issubset(set(result["leaderboard"].columns))

    def test_best_strategy_is_string(self, result):
        assert isinstance(result["best_strategy"], str)

    def test_best_params_is_dict(self, result):
        assert isinstance(result["best_params"], dict)

    def test_best_signal_frame_has_signal_column(self, result):
        assert "signal" in result["best_signal_frame"].columns

    def test_leaderboard_sorted_by_score_descending(self, result):
        scores = result["leaderboard"]["score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_best_strategy_is_top_leaderboard_entry(self, result):
        top_strategy = result["leaderboard"]["strategy"].iloc[0]
        assert result["best_strategy"] == top_strategy


# ---------------------------------------------------------------------------
# run_auto_quant_research — regime filter
# ---------------------------------------------------------------------------


class TestRunAutoQuantResearchRegimeFilter:
    def test_regime_filter_true_runs_without_error(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[10],
            meanrev_windows=[],
            meanrev_entry_z=[],
            regime_filter=True,
        )
        result = run_auto_quant_research(small_ohlcv_df, config=cfg)
        assert "leaderboard" in result

    def test_regime_filter_false_runs_without_error(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[10],
            meanrev_windows=[],
            meanrev_entry_z=[],
            regime_filter=False,
        )
        result = run_auto_quant_research(small_ohlcv_df, config=cfg)
        assert "leaderboard" in result


# ---------------------------------------------------------------------------
# run_auto_quant_research — empty config raises ValueError
# ---------------------------------------------------------------------------


class TestRunAutoQuantResearchEmptyConfig:
    def test_all_empty_lists_raises_value_error(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[],
            meanrev_windows=[],
            meanrev_entry_z=[],
        )
        with pytest.raises(ValueError, match="No strategy candidate"):
            run_auto_quant_research(small_ohlcv_df, config=cfg)

    def test_empty_momentum_but_meanrev_runs(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[],
            meanrev_windows=[20],
            meanrev_entry_z=[2.0],
            regime_filter=False,
        )
        result = run_auto_quant_research(small_ohlcv_df, config=cfg)
        assert result["best_strategy"] == "zscore_mean_reversion"

    def test_empty_meanrev_but_momentum_runs(self, small_ohlcv_df):
        cfg = AutoResearchConfig(
            momentum_lookbacks=[10],
            meanrev_windows=[],
            meanrev_entry_z=[],
            regime_filter=False,
        )
        result = run_auto_quant_research(small_ohlcv_df, config=cfg)
        assert result["best_strategy"] == "ts_momentum"


# ---------------------------------------------------------------------------
# run_auto_quant_research — default config (smoke test)
# ---------------------------------------------------------------------------


class TestRunAutoQuantResearchDefault:
    def test_default_config_produces_nine_candidates(self, small_ohlcv_df):
        """Default config: 3 momentum + 2×3 meanrev = 9 candidates."""
        cfg = AutoResearchConfig(regime_filter=False)
        result = run_auto_quant_research(small_ohlcv_df, config=cfg)
        assert len(result["leaderboard"]) == 9

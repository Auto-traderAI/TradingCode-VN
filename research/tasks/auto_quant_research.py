"""
research/tasks/auto_quant_research.py

Automated quantitative research task:
- prepare common features
- evaluate multiple strategy configurations
- rank and return the strongest trading indicator candidate

This is research prototype code — not production execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from research.backtest import run_backtest
from research.data import load_ohlcv
from research.features import add_returns, add_volatility, add_momentum
from research.regime import VolatilityRegime
from research.signals import TimeSeriesMomentum, ZScoreMeanReversion


@dataclass
class AutoResearchConfig:
    """Configuration for automated strategy search."""

    momentum_lookbacks: list[int] = field(default_factory=lambda: [10, 20, 40])
    meanrev_windows: list[int] = field(default_factory=lambda: [20, 40])
    meanrev_entry_z: list[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    meanrev_exit_z: float = 0.5
    cost_per_trade: float = 0.0002
    periods_per_year: int = 252
    regime_filter: bool = True


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_returns(out, periods=[1, 5, 20])
    out = add_volatility(out, ret_col="ret_1", windows=[20, 60])
    out = add_momentum(out, windows=[5, 20, 60])
    return out


def _strategy_score(metrics: dict[str, Any]) -> float:
    sharpe = float(metrics["sharpe"])
    calmar = float(metrics["calmar"])
    max_drawdown = abs(float(metrics["max_drawdown"]))
    return sharpe + 0.5 * calmar - max_drawdown


def _build_regime_mask(df: pd.DataFrame) -> pd.Series:
    regime = VolatilityRegime(vol_window=20).fit_predict(df, ret_col="ret_1")
    # Trade only outside high-vol regime (label 2)
    return (regime != 2).astype(float)


def _evaluate_candidate(
    df: pd.DataFrame,
    signal: pd.Series,
    name: str,
    params: dict[str, Any],
    config: AutoResearchConfig,
) -> dict[str, Any]:
    bt_df = df.copy()
    bt_df["signal"] = signal
    result = run_backtest(
        bt_df,
        cost_per_trade=config.cost_per_trade,
        periods_per_year=config.periods_per_year,
    )
    score = _strategy_score(result)
    return {
        "strategy": name,
        "params": params,
        "score": score,
        "sharpe": float(result["sharpe"]),
        "calmar": float(result["calmar"]),
        "max_drawdown": float(result["max_drawdown"]),
        "n_trades": int(result["n_trades"]),
        "signal": signal,
    }


def run_auto_quant_research(
    source: str | pd.DataFrame,
    config: AutoResearchConfig | None = None,
) -> dict[str, Any]:
    """Run an automated quant-research task and return best indicator candidate.

    Parameters
    ----------
    source:
        CSV path or OHLCV DataFrame accepted by ``load_ohlcv``.
    config:
        Optional ``AutoResearchConfig``.

    Returns
    -------
    dict
        {
          "leaderboard": pd.DataFrame,
          "best_strategy": str,
          "best_params": dict,
          "best_signal_frame": pd.DataFrame
        }
    """
    cfg = config or AutoResearchConfig()

    df = load_ohlcv(source)
    df = _prepare_features(df)

    regime_mask = _build_regime_mask(df) if cfg.regime_filter else pd.Series(1.0, index=df.index)

    evaluations: list[dict[str, Any]] = []

    for lookback in cfg.momentum_lookbacks:
        signal_df = TimeSeriesMomentum(
            lookback=lookback,
            periods_per_year=cfg.periods_per_year,
        ).generate(df)
        signal = signal_df["signal"].fillna(0) * regime_mask
        evaluations.append(
            _evaluate_candidate(
                df=df,
                signal=signal,
                name="ts_momentum",
                params={"lookback": lookback, "regime_filter": cfg.regime_filter},
                config=cfg,
            )
        )

    for window in cfg.meanrev_windows:
        for entry_z in cfg.meanrev_entry_z:
            signal_df = ZScoreMeanReversion(
                window=window,
                entry_z=entry_z,
                exit_z=cfg.meanrev_exit_z,
            ).generate(df)
            signal = signal_df["signal"].fillna(0) * regime_mask
            evaluations.append(
                _evaluate_candidate(
                    df=df,
                    signal=signal,
                    name="zscore_mean_reversion",
                    params={
                        "window": window,
                        "entry_z": entry_z,
                        "exit_z": cfg.meanrev_exit_z,
                        "regime_filter": cfg.regime_filter,
                    },
                    config=cfg,
                )
            )

    if not evaluations:
        raise ValueError("[auto_quant_research] No strategy candidate was generated.")

    best = max(evaluations, key=lambda row: (row["score"], row["sharpe"], row["calmar"]))
    leaderboard = (
        pd.DataFrame(evaluations)
        .drop(columns=["signal"])
        .sort_values(["score", "sharpe", "calmar"], ascending=False)
        .reset_index(drop=True)
    )
    best_signal_frame = df.copy()
    best_signal_frame["signal"] = best["signal"]

    return {
        "leaderboard": leaderboard,
        "best_strategy": best["strategy"],
        "best_params": best["params"],
        "best_signal_frame": best_signal_frame,
    }

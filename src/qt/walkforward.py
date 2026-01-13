from __future__ import annotations

import json
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .backtest import ExecutionConfig, RiskConfig, run_backtest
from .strategy import EMATrend, BollingerMeanReversion


def _make_strategy(name: str):
    name = (name or "").lower()
    if name == "ema_trend":
        return EMATrend()
    if name in {"bollinger_mr", "bollinger_mean_reversion", "bollinger"}:
        return BollingerMeanReversion()
    raise ValueError(f"Unknown strategy '{name}'")


def walk_forward_eval(
    df: pd.DataFrame,
    strat_name: str,
    exec_cfg: ExecutionConfig,
    risk_cfg: RiskConfig,
    train: int = 252 * 2,
    test: int = 63,
    step: int = 63,
    param_grid: Dict[str, list] | None = None,
) -> pd.DataFrame:
    """Simple walk-forward evaluation.

    For each window:
    - Choose parameters on the TRAIN slice (maximizing Sharpe on net returns)
    - Freeze those parameters and evaluate on the TEST slice
    """
    if param_grid is None:
        if (strat_name or "").lower() == "ema_trend":
            param_grid = {
                "fast": [10, 20],
                "slow": [50, 80],
                "threshold": [0.0005, 0.001, 0.002],
                "rebalance_every": [5],
                "min_hold": [5],
            }
        else:
            param_grid = {
                "window": [20, 40],
                "k": [2.0],
                "entry_z": [2.0, 2.5],
                "exit_z": [0.5, 1.0],
                "rebalance_every": [5],
                "min_hold": [3],
            }

    n = len(df)
    rows = []
    i = 0
    win_id = 0
    strat = _make_strategy(strat_name)

    # Cartesian product of grid
    keys = list(param_grid.keys())
    grid_vals = [param_grid[k] for k in keys]

    def iter_params():
        import itertools

        for vals in itertools.product(*grid_vals):
            yield dict(zip(keys, vals))

    while i + train + test <= n:
        train_df = df.iloc[i : i + train].reset_index(drop=True)
        test_df = df.iloc[i + train : i + train + test].reset_index(drop=True)

        best_params = None
        best_score = -1e18
        best_train_metrics = None

        for params in iter_params():
            sig = strat.generate(train_df, **params).position
            res = run_backtest(train_df, sig, exec_cfg, risk_cfg)
            score = float(res.metrics.get("sharpe", 0.0))
            if score > best_score:
                best_score = score
                best_params = params
                best_train_metrics = res.metrics

        assert best_params is not None

        sig_test = strat.generate(test_df, **best_params).position
        res_test = run_backtest(test_df, sig_test, exec_cfg, risk_cfg)

        rows.append(
            {
                "window": win_id,
                "train_start": str(train_df["timestamp"].iloc[0]),
                "train_end": str(train_df["timestamp"].iloc[-1]),
                "test_start": str(test_df["timestamp"].iloc[0]),
                "test_end": str(test_df["timestamp"].iloc[-1]),
                "params": json.dumps(best_params),
                "train_sharpe": float(best_train_metrics.get("sharpe", 0.0)),
                "train_total_return": float(best_train_metrics.get("total_return", 0.0)),
                "train_max_drawdown": float(best_train_metrics.get("max_drawdown", 0.0)),
                "test_sharpe": float(res_test.metrics.get("sharpe", 0.0)),
                "test_gross_sharpe": float(res_test.metrics.get("gross_sharpe", 0.0)),
                "test_total_return": float(res_test.metrics.get("total_return", 0.0)),
                "test_max_drawdown": float(res_test.metrics.get("max_drawdown", 0.0)),
                "test_total_cost": float(res_test.metrics.get("total_cost", 0.0)),
                "test_avg_turnover": float(res_test.metrics.get("avg_turnover", 0.0)),
                "test_avg_abs_exposure": float(res_test.metrics.get("avg_abs_exposure", 0.0)),
            }
        )

        win_id += 1
        i += int(step)

    return pd.DataFrame(rows)

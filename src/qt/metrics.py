from __future__ import annotations
import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, ann: float = 252.0) -> float:
    r = returns.dropna()
    if r.std(ddof=0) < 1e-12:
        return 0.0
    return float(np.sqrt(ann) * r.mean() / r.std(ddof=0))

def sortino(returns: pd.Series, ann: float = 252.0) -> float:
    r = returns.dropna()
    dn = r[r < 0]
    if dn.std(ddof=0) < 1e-12:
        return 0.0
    return float(np.sqrt(ann) * r.mean() / dn.std(ddof=0))

def max_drawdown(equity: pd.Series) -> float:
    x = equity.values
    peak = np.maximum.accumulate(x)
    dd = x / (peak + 1e-12) - 1.0
    return float(dd.min())

def turnover(exposure: pd.Series) -> float:
    return float(exposure.diff().abs().mean())

def compute_metrics(returns, equity, cost, exposure) -> dict:
    returns = pd.Series(returns)
    equity = pd.Series(equity)
    cost = pd.Series(cost)
    exposure = pd.Series(exposure)
    return {
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "total_cost": float(cost.sum()),
        "avg_turnover": turnover(exposure),
        "avg_abs_exposure": float(exposure.abs().mean()),
        "n_days": int(len(equity)),
    }

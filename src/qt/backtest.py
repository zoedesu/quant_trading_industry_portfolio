from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .execution import ExecutionConfig, apply_costs
from .metrics import compute_metrics
from .indicators import atr as atr_fn, ewma_vol as ewma_vol_fn


@dataclass
class RiskConfig:
    """Risk and sizing controls.

    Notes:
    - Position signal should be in [-1, 1]. RiskConfig controls *sizing* and safety brakes.
    - Sizing uses only past information by construction.
    """

    # hard caps
    max_leverage: float = 1.0
    max_pos_frac: float = 1.0

    # vol targeting
    vol_target_annual: float = 0.20
    sizing_method: str = "ewma"  # "atr" or "ewma"
    atr_window: int = 14
    ewma_lambda: float = 0.94

    # tail-risk brakes
    max_drawdown_stop: float | None = 0.30  # stop trading after 30% peak-to-trough drawdown
    daily_loss_limit: float | None = None   # e.g. 0.03 -> stop for the day after -3% return

    # churn control at the execution layer (independent from strategy rules)
    rebalance_every: int = 1  # 1=daily, 5=weekly; applied to exposure changes


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _vol_estimate(df: pd.DataFrame, returns: pd.Series, risk_cfg: RiskConfig) -> pd.Series:
    """Vol estimate aligned to df rows.

    Important: result is shifted so that vol_est[t] uses data up to t-1,
    which avoids any lookahead when sizing exposure for day t.
    """
    if risk_cfg.sizing_method.lower() == "atr":
        a = atr_fn(df["high"], df["low"], df["close"], risk_cfg.atr_window)
        vol = (a / (df["close"] + 1e-12)).clip(lower=1e-12)
    elif risk_cfg.sizing_method.lower() == "ewma":
        vol = ewma_vol_fn(returns, lam=risk_cfg.ewma_lambda).clip(lower=1e-12)
    else:
        raise ValueError("sizing_method must be 'atr' or 'ewma'")
    return vol.shift(1).bfill().clip(lower=1e-6)


def run_backtest(
    df: pd.DataFrame,
    position: pd.Series,
    exec_cfg: ExecutionConfig,
    risk_cfg: RiskConfig,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Vectorized-ish single-asset backtest with explicit cash accounting.

    Key convention:
    - We execute trades at time t (based on information up to t),
      then we hold yesterday's exposure during today's return:
        pnl_gross[t] = exposure[t-1] * equity[t-1] * return[t]

    This is the standard no-lookahead convention for daily bars.
    """
    df = df.copy().reset_index(drop=True)
    position = position.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)

    # Choose how to map bar data to returns
    if exec_cfg.fill_price == "next_open":
        # enter at next open -> return from open_t to open_{t+1}
        ret = df["open"].shift(-1) / df["open"] - 1.0
    elif exec_cfg.fill_price == "close":
        # enter/exit at close -> close-to-close returns
        ret = df["close"].pct_change()
    else:
        raise ValueError("fill_price must be 'next_open' or 'close'")

    valid = ret.notna()
    df = df.loc[valid].reset_index(drop=True)
    position = position.loc[valid].reset_index(drop=True)
    ret = ret.loc[valid].reset_index(drop=True)

    # ---- sizing: vol targeting ----
    vol_est = _vol_estimate(df, ret, risk_cfg)
    target_daily = risk_cfg.vol_target_annual / np.sqrt(252.0)
    scale = (target_daily / vol_est).clip(lower=0.0, upper=risk_cfg.max_leverage)

    desired_exposure = (position * scale).clip(-risk_cfg.max_leverage, risk_cfg.max_leverage)
    desired_exposure = desired_exposure.clip(
        -risk_cfg.max_leverage * risk_cfg.max_pos_frac,
        risk_cfg.max_leverage * risk_cfg.max_pos_frac,
    )

    # Optional execution-layer rebalancing frequency (reduces churn/turnover)
    if int(risk_cfg.rebalance_every) > 1:
        k = int(risk_cfg.rebalance_every)
        mask = (np.arange(len(desired_exposure)) % k) == 0
        desired_exposure = desired_exposure.where(mask).ffill().fillna(0.0)

    # trades/exposure changes are computed against *previous* exposure
    exp_prev = desired_exposure.shift(1).fillna(0.0)
    trade = desired_exposure - exp_prev

    n = len(df)
    equity = np.zeros(n, dtype=float)
    pnl_gross = np.zeros(n, dtype=float)
    pnl_net = np.zeros(n, dtype=float)
    cost = np.zeros(n, dtype=float)
    trade_notional = np.zeros(n, dtype=float)

    equity[0] = float(initial_capital)

    peak = equity[0]
    stopped = False

    for t in range(1, n):
        # tail-risk: stop trading permanently after breaching max drawdown
        if stopped:
            # flatten at first opportunity
            exp_prev_t = 0.0
            trade_t = -exp_prev.iloc[t]  # exposure goes to 0
        else:
            exp_prev_t = float(exp_prev.iloc[t])
            trade_t = float(trade.iloc[t])

        # notional traded is proportional to equity (simple margin model)
        trade_notional[t] = trade_t * equity[t - 1]
        cost[t] = apply_costs(trade_notional[t], exec_cfg.fee_bps, exec_cfg.slippage_bps)

        pnl_gross[t] = exp_prev_t * equity[t - 1] * float(ret.iloc[t])
        pnl_net[t] = pnl_gross[t] - cost[t]
        equity[t] = equity[t - 1] + pnl_net[t]

        # update drawdown and apply stop logic
        peak = max(peak, equity[t])
        dd = equity[t] / (peak + 1e-12) - 1.0

        if (risk_cfg.max_drawdown_stop is not None) and (dd <= -float(risk_cfg.max_drawdown_stop)):
            stopped = True

        if risk_cfg.daily_loss_limit is not None:
            # daily loss is net pnl relative to equity_{t-1}
            day_ret = pnl_net[t] / (equity[t - 1] + 1e-12)
            if day_ret <= -float(risk_cfg.daily_loss_limit):
                # stop for the next day only by forcing desired exposure to 0 at t+1 via exp_prev mechanics
                exp_prev.iloc[t + 1 :] = 0.0
                trade.iloc[t + 1 :] = 0.0

    eq = pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "equity": equity,
            "pnl_gross": pnl_gross,
            "pnl_net": pnl_net,
            "cost": cost,
            "exposure_target": desired_exposure.values,
            "exposure_held": exp_prev.values,
            "returns_net": pd.Series(equity).pct_change().fillna(0.0).values,
        }
    )
    eq["returns_gross"] = (eq["pnl_gross"] / (pd.Series(equity).shift(1).fillna(equity[0]) + 1e-12)).fillna(0.0)
    eq["cum_cost"] = eq["cost"].cumsum()
    eq["cum_pnl_gross"] = eq["pnl_gross"].cumsum()
    eq["cum_pnl_net"] = eq["pnl_net"].cumsum()

    trades = pd.DataFrame(
        {
            "timestamp": df["timestamp"].values,
            "trade_exposure": trade.values,
            "trade_notional": trade_notional,
            "cost": cost,
        }
    )

    metrics_net = compute_metrics(eq["returns_net"], eq["equity"], eq["cost"], eq["exposure_held"])
    metrics_gross = compute_metrics(eq["returns_gross"], (initial_capital + eq["cum_pnl_gross"]), 0.0 * eq["cost"], eq["exposure_held"])

    metrics = {
        **metrics_net,
        "gross_sharpe": float(metrics_gross.get("sharpe", 0.0)),
        "gross_total_return": float((initial_capital + eq["cum_pnl_gross"].iloc[-1]) / initial_capital - 1.0),
        "gross_total_pnl": float(eq["cum_pnl_gross"].iloc[-1]),
        "net_total_pnl": float(eq["cum_pnl_net"].iloc[-1]),
    }

    return BacktestResult(eq, trades, metrics)

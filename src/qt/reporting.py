from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd


def save_equity_plot(eq: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(pd.to_datetime(eq["timestamp"], utc=True), eq["equity"].astype(float))
    plt.xlabel("time")
    plt.ylabel("equity")
    plt.title("Equity curve (net of costs)")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_drawdown_plot(eq: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    equity = eq["equity"].astype(float).values
    peak = (pd.Series(equity)).cummax().values
    dd = equity / (peak + 1e-12) - 1.0
    plt.figure()
    plt.plot(pd.to_datetime(eq["timestamp"], utc=True), dd)
    plt.xlabel("time")
    plt.ylabel("drawdown")
    plt.title("Drawdown")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_trade_hist(trades: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(trades["trade_notional"].abs().astype(float), bins=50)
    plt.xlabel("abs(trade notional)")
    plt.ylabel("count")
    plt.title("Trade size histogram")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_gross_vs_net(eq: pd.DataFrame, out: Path) -> None:
    """Cumulative gross PnL vs costs vs net PnL."""
    out.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.to_datetime(eq["timestamp"], utc=True)
    cum_gross = eq.get("cum_pnl_gross", eq["pnl_gross"].cumsum()).astype(float)
    cum_cost = eq.get("cum_cost", eq["cost"].cumsum()).astype(float)
    cum_net = eq.get("cum_pnl_net", eq["pnl_net"].cumsum()).astype(float)

    plt.figure()
    plt.plot(ts, cum_gross, label="cum gross pnl")
    plt.plot(ts, -cum_cost, label="-cum costs")
    plt.plot(ts, cum_net, label="cum net pnl")
    plt.xlabel("time")
    plt.ylabel("pnl")
    plt.title("Gross vs costs vs net")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_rolling_eval_plot(roll: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if roll is None or len(roll) == 0:
        return
    plt.figure()
    plt.plot(roll["window"], roll["test_sharpe"], marker="o")
    plt.axhline(0.0)
    plt.xlabel("window")
    plt.ylabel("test Sharpe")
    plt.title("Walk-forward: test Sharpe per window")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def save_metrics_pretty(metrics: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

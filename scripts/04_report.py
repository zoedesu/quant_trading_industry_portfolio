from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from qt.reporting import (
    save_equity_plot,
    save_drawdown_plot,
    save_trade_hist,
    save_gross_vs_net,
    save_rolling_eval_plot,
    save_metrics_pretty,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outdir = root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    eq_path = outdir / "equity_curve.csv"
    tr_path = outdir / "trades.csv"
    m_path = outdir / "metrics.json"

    if eq_path.exists():
        eq = pd.read_csv(eq_path)
        save_equity_plot(eq, outdir / "equity.png")
        save_drawdown_plot(eq, outdir / "drawdown.png")
        save_gross_vs_net(eq, outdir / "gross_vs_net.png")
    else:
        eq = None

    if tr_path.exists():
        trades = pd.read_csv(tr_path)
        save_trade_hist(trades, outdir / "trade_hist.png")

    if (outdir / "rolling_eval.csv").exists():
        roll = pd.read_csv(outdir / "rolling_eval.csv")
        save_rolling_eval_plot(roll, outdir / "rolling_eval.png")

    if m_path.exists():
        metrics = json.loads(m_path.read_text(encoding="utf-8"))
        save_metrics_pretty(metrics, outdir / "metrics_pretty.json")

    print("saved plots + metrics_pretty.json")


if __name__ == "__main__":
    main()

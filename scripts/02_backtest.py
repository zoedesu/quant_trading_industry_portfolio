from __future__ import annotations

from pathlib import Path
import argparse
import json
import yaml

from qt.data import load_ohlcv_from_config
from qt.strategy import EMATrend, BollingerMeanReversion
from qt.backtest import run_backtest, ExecutionConfig, RiskConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to YAML config (default: config/example.yaml)")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else root / "config" / "example.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    df = load_ohlcv_from_config(cfg, base_dir=root).df

    exec_cfg = ExecutionConfig(
        fill_price=cfg["execution"]["fill_price"],
        fee_bps=float(cfg["execution"]["fee_bps"]),
        slippage_bps=float(cfg["execution"]["slippage_bps"]),
    )
    risk_cfg = RiskConfig(
        max_leverage=float(cfg["execution"].get("max_leverage", 1.0)),
        max_pos_frac=float(cfg["risk"]["max_pos_frac"]),
        vol_target_annual=float(cfg["risk"]["vol_target_annual"]),
        sizing_method=str(cfg["risk"].get("sizing_method", "ewma")),
        atr_window=int(cfg["risk"]["atr_window"]),
        ewma_lambda=float(cfg["risk"].get("ewma_lambda", 0.94)),
        max_drawdown_stop=float(cfg["risk"].get("max_drawdown_stop", 0.30)) if cfg["risk"].get("max_drawdown_stop", None) is not None else None,
        daily_loss_limit=float(cfg["risk"].get("daily_loss_limit", 0.0)) if cfg["risk"].get("daily_loss_limit", None) else None,
        rebalance_every=int(cfg["execution"].get("rebalance_every", 1)),
    )

    strat_name = cfg["strategy"]["name"]
    params = cfg["strategy"]["params"]

    strat = EMATrend() if strat_name == "ema_trend" else BollingerMeanReversion()
    sig = strat.generate(df, **params).position

    res = run_backtest(df, sig, exec_cfg, risk_cfg)

    outdir = root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    res.equity.to_csv(outdir / "equity_curve.csv", index=False)
    res.trades.to_csv(outdir / "trades.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(res.metrics, indent=2), encoding="utf-8")
    print("saved outputs. metrics:", res.metrics)


if __name__ == "__main__":
    main()

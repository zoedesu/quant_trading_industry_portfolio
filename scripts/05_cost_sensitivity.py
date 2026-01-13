from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import yaml
import pandas as pd
import matplotlib.pyplot as plt

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

    base_exec = ExecutionConfig(
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

    fee_grid = cfg.get("cost_sweep", {}).get("fee_bps", [0, 2, 5, 10])
    slp_grid = cfg.get("cost_sweep", {}).get("slippage_bps", [0, 1, 2, 5])

    rows = []
    for fee, slp in itertools.product(fee_grid, slp_grid):
        exec_cfg = ExecutionConfig(fill_price=base_exec.fill_price, fee_bps=float(fee), slippage_bps=float(slp))
        res = run_backtest(df, sig, exec_cfg, risk_cfg)
        rows.append(
            {
                "fee_bps": float(fee),
                "slippage_bps": float(slp),
                "sharpe": float(res.metrics.get("sharpe", 0.0)),
                "gross_sharpe": float(res.metrics.get("gross_sharpe", 0.0)),
                "total_return": float(res.metrics.get("total_return", 0.0)),
                "gross_total_return": float(res.metrics.get("gross_total_return", 0.0)),
                "total_cost": float(res.metrics.get("total_cost", 0.0)),
                "avg_turnover": float(res.metrics.get("avg_turnover", 0.0)),
            }
        )

    outdir = root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows).sort_values(["fee_bps", "slippage_bps"]).reset_index(drop=True)
    out.to_csv(outdir / "cost_sweep.csv", index=False)

    piv = out.pivot(index="fee_bps", columns="slippage_bps", values="sharpe")
    plt.figure()
    plt.imshow(piv.values, aspect="auto")
    plt.xticks(range(len(piv.columns)), [str(c) for c in piv.columns])
    plt.yticks(range(len(piv.index)), [str(i) for i in piv.index])
    plt.xlabel("slippage_bps")
    plt.ylabel("fee_bps")
    plt.title("Cost sensitivity: net Sharpe")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(outdir / "cost_sweep.png", dpi=200)
    plt.close()

    print("saved:", outdir / "cost_sweep.csv")


if __name__ == "__main__":
    main()

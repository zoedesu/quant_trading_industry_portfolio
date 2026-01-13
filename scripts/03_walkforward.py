from __future__ import annotations

from pathlib import Path
import argparse
import json
import yaml

from qt.data import load_ohlcv_from_config
from qt.backtest import ExecutionConfig, RiskConfig
from qt.walkforward import walk_forward_eval


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

    roll = walk_forward_eval(
        df,
        cfg["strategy"]["name"],
        exec_cfg,
        risk_cfg,
        train=int(cfg.get("walkforward", {}).get("train", 252 * 2)),
        test=int(cfg.get("walkforward", {}).get("test", 63)),
        step=int(cfg.get("walkforward", {}).get("step", 63)),
    )

    outdir = root / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    roll.to_csv(outdir / "rolling_eval.csv", index=False)

    summary = {
        "n_windows": int(len(roll)),
        "avg_test_sharpe": float(roll["test_sharpe"].mean()) if len(roll) else 0.0,
        "median_test_sharpe": float(roll["test_sharpe"].median()) if len(roll) else 0.0,
        "avg_test_gross_sharpe": float(roll["test_gross_sharpe"].mean()) if len(roll) else 0.0,
    }
    (outdir / "rolling_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("saved rolling eval outputs")


if __name__ == "__main__":
    main()

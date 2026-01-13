from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import yaml

from qt.data import OHLCV


def generate_synthetic_ohlcv(n: int = 2600, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV series with a simple regime change.

    - First 40% of the sample: lower volatility (1% daily)
    - Remaining 60%: higher volatility (2% daily)

    This is intentionally simplistic; use real market data for any serious evaluation.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    # regime schedule
    vol = np.where(idx < n * 0.4, 0.01, 0.02)
    drift = np.where(idx < n * 0.5, 0.0002, 0.0004)

    rets = drift + vol * rng.standard_normal(n)
    price = 100.0 * np.exp(np.cumsum(rets))

    close = pd.Series(price)
    open_ = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001 * rng.standard_normal(n))
    high = np.maximum(open_, close) * (1 + np.abs(0.002 * rng.standard_normal(n)))
    low = np.minimum(open_, close) * (1 - np.abs(0.002 * rng.standard_normal(n)))
    volume = 1e6 * (1 + 0.1 * rng.standard_normal(n)).clip(0.1)

    ts = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


def _load_cfg(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to YAML config (default: config/example.yaml)")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config) if args.config else root / "config" / "example.yaml"
    cfg = _load_cfg(cfg_path)

    data_cfg = cfg.get("data", {})
    source = (data_cfg.get("source") or "csv").lower()

    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    if source == "yahoo":
        symbol = str(data_cfg.get("symbol", "SPY"))
        interval = str(data_cfg.get("interval", "1d"))
        start = str(data_cfg.get("start", "2015-01-01"))
        end = data_cfg.get("end", None)

        out = out_dir / f"{symbol}_{interval}.csv"
        df = OHLCV.from_yahoo(symbol=symbol, start=start, end=end, interval=interval).df
        df.to_csv(out, index=False)
        print("saved real data:", out, "rows:", len(df))
        return

    # csv: if file exists, do nothing (lets you bring your own dataset)
    if source == "csv" and "path" in data_cfg:
        path = root / str(data_cfg["path"])
        if path.exists():
            print("data already exists:", path)
            return

    # fallback: generate synthetic
    n = int(data_cfg.get("n", 2600))
    seed = int(data_cfg.get("seed", 42))
    out = out_dir / "sample_ohlcv.csv"
    df = generate_synthetic_ohlcv(n=n, seed=seed)
    df.to_csv(out, index=False)
    print("saved synthetic data:", out, "rows:", len(df))


if __name__ == "__main__":
    main()

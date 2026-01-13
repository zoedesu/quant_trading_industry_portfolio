from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd

REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class OHLCV:
    """OHLCV wrapper with light validation."""

    df: pd.DataFrame

    @staticmethod
    def from_csv(path: str | Path) -> "OHLCV":
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Required: {REQUIRED_COLS}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        return OHLCV(df)
    @staticmethod
    def from_yahoo(symbol: str, start: str = "2015-01-01", end: str | None = None, interval: str = "1d") -> "OHLCV":
        """Fetch OHLCV from Yahoo Finance via yfinance.

        Notes
        -----
        yfinance can return either:
        - single-level columns: ["Open","High","Low","Close","Adj Close","Volume"]
        - MultiIndex columns (field, ticker) depending on yfinance version / parameters.
        This method normalizes both cases into the REQUIRED_COLS schema.
        """
        try:
            import yfinance as yf
        except ImportError as e:
            raise ImportError("yfinance is required for real data. pip install yfinance") from e

        raw = yf.download(symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
        if raw is None or len(raw) == 0:
            raise ValueError(f"No data returned for symbol={symbol}. Check ticker / internet / date range.")

        # Normalize MultiIndex columns (field, ticker) -> field only
        if isinstance(raw.columns, pd.MultiIndex):
            # Prefer slicing by the ticker level if present
            lvl = raw.columns.nlevels - 1
            tickers = set(raw.columns.get_level_values(lvl))
            if symbol in tickers:
                raw = raw.xs(symbol, axis=1, level=lvl, drop_level=True)
            else:
                # Fall back: keep first level and drop duplicates
                raw.columns = raw.columns.get_level_values(0)
                raw = raw.loc[:, ~raw.columns.duplicated()]

        raw = raw.reset_index()
        # Date/Datetime column name differs across yfinance versions
        dt_col = "Date" if "Date" in raw.columns else ("Datetime" if "Datetime" in raw.columns else None)
        if dt_col is None:
            raise ValueError(f"Unexpected Yahoo data columns: {list(raw.columns)}")
        raw = raw.rename(columns={dt_col: "timestamp"})

        def _as_1d_series(v):
            """Convert v to a 1D pandas Series."""
            if isinstance(v, pd.Series):
                return v
            if isinstance(v, pd.DataFrame):
                if v.shape[1] == 1:
                    return v.iloc[:, 0]
                # if multiple columns remain, take first
                return v.iloc[:, 0]
            a = np.asarray(v)
            if a.ndim > 1:
                a = a.squeeze()
            return pd.Series(a)

        # Some versions return lower-case, others title-case
        colmap = {c.lower(): c for c in raw.columns}
        def _get(name: str):
            key = name.lower()
            if key in colmap:
                return _as_1d_series(raw[colmap[key]])
            raise KeyError(name)

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["timestamp"], utc=True, errors="coerce"),
                "open": pd.to_numeric(_get("Open"), errors="coerce"),
                "high": pd.to_numeric(_get("High"), errors="coerce"),
                "low": pd.to_numeric(_get("Low"), errors="coerce"),
                "close": pd.to_numeric(_get("Close"), errors="coerce"),
                "volume": pd.to_numeric(_get("Volume") if "volume" in colmap else 0.0, errors="coerce").fillna(0.0),
            }
        )
        df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").reset_index(drop=True)
        return OHLCV(df)

    def to_returns(self) -> pd.Series:
        return self.df["close"].pct_change()

    def copy(self) -> "OHLCV":
        return OHLCV(self.df.copy())


def load_ohlcv_from_config(cfg: dict, base_dir: Path) -> OHLCV:
    """Helper used by scripts. Supports csv/synthetic/yahoo sources."""
    data_cfg = cfg.get("data", {})
    source = (data_cfg.get("source") or "csv").lower()

    if source == "csv":
        path = base_dir / str(data_cfg["path"])
        return OHLCV.from_csv(path)

    if source == "yahoo":
        symbol = str(data_cfg.get("symbol", "SPY"))
        start = str(data_cfg.get("start", "2015-01-01"))
        end = data_cfg.get("end", None)
        interval = str(data_cfg.get("interval", "1d"))
        return OHLCV.from_yahoo(symbol=symbol, start=start, end=end, interval=interval)

    raise ValueError(f"Unknown data.source={source}. Supported: csv, yahoo")

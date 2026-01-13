from __future__ import annotations

import numpy as np
import pandas as pd


def ema(x: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return x.ewm(span=span, adjust=False).mean()


def sma(x: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return x.rolling(window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (RSI).

    Simple rolling-mean implementation; avoids lookahead by construction.
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    roll_up = up.rolling(window, min_periods=window).mean()
    roll_down = down.rolling(window, min_periods=window).mean()

    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger_bands(close: pd.Series, window: int = 20, k: float = 2.0):
    """Bollinger Bands: mid, upper, lower."""
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


def ewma_vol(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA volatility estimate.

    var_t = lam * var_{t-1} + (1-lam) * r_t^2
    Caller should shift if they want an estimate based on info up to t-1.
    """
    r = pd.Series(returns).fillna(0.0)
    alpha = 1.0 - float(lam)
    var = (r ** 2).ewm(alpha=alpha, adjust=False).mean()
    return np.sqrt(var.clip(lower=1e-12))


def realized_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling realized volatility of returns."""
    r = pd.Series(returns).fillna(0.0)
    return r.rolling(window, min_periods=window).std(ddof=0)

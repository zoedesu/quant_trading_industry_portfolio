from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .indicators import ema, bollinger_bands


@dataclass
class StrategyResult:
    """Container returned by strategies.

    - signal: continuous (unbounded) score (for debugging/plots)
    - position: desired position in [-1, 1] *before* risk sizing
    """
    signal: pd.Series
    position: pd.Series


class BaseStrategy:
    name: str = "base"

    @staticmethod
    def _apply_controls(
        position: pd.Series,
        rebalance_every: int = 1,
        min_hold: int = 0,
    ) -> pd.Series:
        """Reduce churn / turnover in a simple, explicit way.

        - rebalance_every: only allow position changes every k bars; otherwise hold previous.
        - min_hold: enforce minimum holding period before switching to a new non-zero position.
        """
        pos = pd.Series(position).fillna(0.0).clip(-1.0, 1.0).astype(float)

        # Rebalance frequency control
        k = int(rebalance_every) if rebalance_every else 1
        if k > 1:
            idx = np.arange(len(pos))
            can_trade = (idx % k) == 0
            pos = pos.where(can_trade).ffill().fillna(0.0)

        # Minimum holding period (prevents rapid flip-flops)
        h = int(min_hold) if min_hold else 0
        if h > 0 and len(pos) > 0:
            out = pos.copy()
            last_change = 0
            for i in range(1, len(out)):
                if out.iloc[i] != out.iloc[i - 1]:
                    if (i - last_change) < h:
                        out.iloc[i] = out.iloc[i - 1]
                    else:
                        last_change = i
            pos = out

        return pos

    def generate(self, df: pd.DataFrame, **params) -> StrategyResult:
        raise NotImplementedError


class EMATrend(BaseStrategy):
    name = "ema_trend"

    def generate(
        self,
        df: pd.DataFrame,
        fast: int = 20,
        slow: int = 60,
        threshold: float = 0.001,
        rebalance_every: int = 5,
        min_hold: int = 5,
    ) -> StrategyResult:
        """Simple EMA trend-following signal with a no-trade band.

        signal = fast_ema/slow_ema - 1
        position:
          +1 if signal > threshold
          -1 if signal < -threshold
           0 otherwise
        """
        close = df["close"].astype(float)
        f = ema(close, int(fast))
        s = ema(close, int(slow))
        signal = (f / (s + 1e-12)) - 1.0

        th = float(threshold)
        pos = pd.Series(0.0, index=df.index)
        pos = pos.mask(signal > th, 1.0)
        pos = pos.mask(signal < -th, -1.0)

        pos = self._apply_controls(pos, rebalance_every=rebalance_every, min_hold=min_hold)
        return StrategyResult(signal=signal, position=pos)


class BollingerMeanReversion(BaseStrategy):
    name = "bollinger_mr"

    def generate(
        self,
        df: pd.DataFrame,
        window: int = 20,
        k: float = 2.0,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        rebalance_every: int = 5,
        min_hold: int = 3,
    ) -> StrategyResult:
        """Mean-reversion using Bollinger Bands with entry/exit hysteresis.

        - Enter long when z < -entry_z (price far below mid)
        - Enter short when z > +entry_z
        - Exit to flat when |z| < exit_z
        """
        close = df["close"].astype(float)
        mid, upper, lower = bollinger_bands(close, window=int(window), k=float(k))
        std = (upper - mid) / (float(k) + 1e-12)
        z = (close - mid) / (std + 1e-12)

        pos = pd.Series(0.0, index=df.index)

        in_pos = 0.0
        for i in range(len(df)):
            zi = float(z.iloc[i]) if pd.notna(z.iloc[i]) else 0.0
            if in_pos == 0.0:
                if zi <= -float(entry_z):
                    in_pos = 1.0
                elif zi >= float(entry_z):
                    in_pos = -1.0
            else:
                if abs(zi) <= float(exit_z):
                    in_pos = 0.0
            pos.iloc[i] = in_pos

        pos = self._apply_controls(pos, rebalance_every=rebalance_every, min_hold=min_hold)
        return StrategyResult(signal=z, position=pos)

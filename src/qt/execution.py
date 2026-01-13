from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class ExecutionConfig:
    fill_price: str = "next_open"
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

def apply_costs(trade_notional: float, fee_bps: float, slippage_bps: float) -> float:
    bps = (fee_bps + slippage_bps) / 1e4
    return float(abs(trade_notional) * bps)

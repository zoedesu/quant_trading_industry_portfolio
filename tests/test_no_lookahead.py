from pathlib import Path
from qt.data import OHLCV
from qt.strategy import EMATrend

def test_shifted_signal_first_zero():
    df = OHLCV.from_csv(Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv").df
    sig = EMATrend().generate(df, fast=5, slow=10).position
    assert float(sig.iloc[0]) == 0.0

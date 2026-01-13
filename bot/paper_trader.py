from pathlib import Path
from qt.data import OHLCV
from qt.strategy import EMATrend
from qt.backtest import run_backtest, ExecutionConfig, RiskConfig

def main():
    data_path = Path(__file__).resolve().parents[1] / "data" / "sample_ohlcv.csv"
    df = OHLCV.from_csv(data_path).df
    strat = EMATrend()
    sig = strat.generate(df, fast=20, slow=80).position
    res = run_backtest(df, sig, ExecutionConfig(), RiskConfig())
    print(res.metrics)

if __name__ == "__main__":
    main()

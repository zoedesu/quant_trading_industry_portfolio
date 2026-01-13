.PHONY: run-all rolling-eval clean

run-all:
	python scripts/01_generate_data.py
	python scripts/02_backtest.py
	python scripts/04_report.py

rolling-eval:
	python scripts/03_walkforward.py

clean:
	rm -rf outputs/*

@echo off
setlocal

set CFG=%1

if "%CFG%"=="" (
  set CFG=
  echo Using default config: config\example.yaml
) else (
  echo Using config: %CFG%
)

echo [1/4] Prepare data (synthetic or real via config)...
if "%CFG%"=="" (py scripts\01_generate_data.py) else (py scripts\01_generate_data.py --config %CFG%)
if errorlevel 1 goto :error

echo [2/4] Run backtest(s)...
if "%CFG%"=="" (py scripts\02_backtest.py) else (py scripts\02_backtest.py --config %CFG%)
if errorlevel 1 goto :error

echo [3/4] Build report plots...
py scripts\04_report.py
if errorlevel 1 goto :error

echo [4/4] Cost sensitivity sweep (fees/slippage)...
if "%CFG%"=="" (py scripts\05_cost_sensitivity.py) else (py scripts\05_cost_sensitivity.py --config %CFG%)
if errorlevel 1 goto :error

echo.
echo Done. See outputs\ folder.
exit /b 0

:error
echo.
echo ERROR: command failed.
exit /b 1

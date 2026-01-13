@echo off
setlocal

set CFG=%1

if "%CFG%"=="" (
  set CFG=
  echo Using default config: config\example.yaml
) else (
  echo Using config: %CFG%
)

echo Running walk-forward (rolling) evaluation...
if "%CFG%"=="" (py scripts\03_walkforward.py) else (py scripts\03_walkforward.py --config %CFG%)
if errorlevel 1 goto :error

echo Building report plots...
py scripts\04_report.py
if errorlevel 1 goto :error

echo.
echo Done. See outputs\rolling_eval.csv and outputs\rolling_eval.png
exit /b 0

:error
echo.
echo ERROR: command failed.
exit /b 1

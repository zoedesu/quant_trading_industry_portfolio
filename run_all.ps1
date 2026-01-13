param(
  [string]$Config = ""
)
$ErrorActionPreference = 'Stop'

if ($Config -ne "") {
  Write-Host "Using config: $Config"
  py scripts/01_generate_data.py --config $Config
  py scripts/02_backtest.py --config $Config
  py scripts/04_report.py
  py scripts/05_cost_sensitivity.py --config $Config
} else {
  Write-Host "Using default config: config/example.yaml"
  py scripts/01_generate_data.py
  py scripts/02_backtest.py
  py scripts/04_report.py
  py scripts/05_cost_sensitivity.py
}

Write-Host ""
Write-Host "Done. See outputs/ folder."

param(
  [string]$Config = ""
)
$ErrorActionPreference = 'Stop'

if ($Config -ne "") {
  Write-Host "Using config: $Config"
  py scripts/03_walkforward.py --config $Config
} else {
  Write-Host "Using default config: config/example.yaml"
  py scripts/03_walkforward.py
}

py scripts/04_report.py

Write-Host ''
Write-Host 'Done. See outputs/rolling_eval.csv and outputs/rolling_eval.png'

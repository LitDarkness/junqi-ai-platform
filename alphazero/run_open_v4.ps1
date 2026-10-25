param([string]$Config = "config/open_junqi_v4_repaired.json", [int]$ThroughIteration = 1)
$ErrorActionPreference = "Stop"
$TaskRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $TaskRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$TaskRoot\build;$TaskRoot\python;$env:PYTHONPATH"
& python "$TaskRoot\tools\run_open_v4.py" --config $Config --through-iteration $ThroughIteration
exit $LASTEXITCODE

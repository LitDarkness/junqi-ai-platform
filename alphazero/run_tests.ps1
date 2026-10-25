param([string]$Python = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = "$Root\build;$Root\python;$env:PYTHONPATH"
if (-not $Python) {
    $Preferred = "python"
    $Python = if (Test-Path -LiteralPath $Preferred) {
        $Preferred
    } else {
        (Get-Command python -ErrorAction Stop).Source
    }
}
& $Python .\cpp\smoke_test.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python .\tests\native_conformance.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m unittest discover -s .\tests -p "test_*.py" -v
exit $LASTEXITCODE

$ErrorActionPreference = 'Stop'
$releaseRoot = $PSScriptRoot
$releasePython = Join-Path $releaseRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $releasePython)) {
    & python -m venv (Join-Path $releaseRoot '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $releasePython -m pip install -r (Join-Path $releaseRoot 'requirements.txt') -e $releaseRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $releasePython (Join-Path $releaseRoot 'reproduce.py') @args
exit $LASTEXITCODE

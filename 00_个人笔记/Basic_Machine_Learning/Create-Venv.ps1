<#
.SYNOPSIS
Create this learning directory's .venv with a Python chosen by the user.
.DESCRIPTION
Structure: resolve Python -> create or reuse .venv -> verify isolation -> show usage.
Input: -Python executable path or command name (default: python on PATH).
Output: .venv/ beside this script; no packages are downloaded by this entry.
Call flow: main -> Get-SelectedPython / Confirm-LearningVenv -> native Python.
#>
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$taskRoot = [IO.Path]::GetFullPath($PSScriptRoot)
$taskVenv = Join-Path $taskRoot '.venv'
$taskVenvPython = Join-Path $taskVenv 'Scripts/python.exe'

function Get-SelectedPython {
    <# Resolve the user-selected executable and require a working Python 3. #>
    param([string]$Selection)
    $command = Get-Command -Name $Selection -CommandType Application -ErrorAction Stop | Select-Object -First 1
    $probe = 'import sys; assert sys.version_info.major == 3, ''Python 3 is required''; print(sys.version.split()[0])'
    $version = & $command.Source -B -c $probe
    if ($LASTEXITCODE -ne 0) { throw 'Selected Python did not run. Specify a working executable with -Python.' }
    Write-Host ('Selected Python: ' + $command.Source + ' (' + ($version -join '').Trim() + ')')
    return $command.Source
}

function Confirm-LearningVenv {
    <# Verify that the interpreter is isolated and belongs to this exact directory. #>
    $probe = 'import pathlib,sys; expected=pathlib.Path(sys.argv[1]).resolve(); actual=pathlib.Path(sys.prefix).resolve(); assert actual==expected and sys.prefix!=sys.base_prefix, ''Unexpected virtual environment''; assert not (actual/''conda-meta'').exists(), ''Expected Python venv''; print(''Verified venv: '' + str(actual)); print(''Python: '' + sys.version.split()[0])'
    & $taskVenvPython -B -c $probe $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment verification failed. Existing files were preserved.' }
}

if (Test-Path -LiteralPath $taskVenv) {
    if (-not (Test-Path -LiteralPath $taskVenvPython -PathType Leaf)) {
        throw '.venv already exists but has no Windows Python interpreter. Inspect or move it before retrying.'
    }
    Write-Host 'Existing .venv found; checking it without replacing its Python or packages.'
} else {
    $taskSelectedPython = Get-SelectedPython $Python
    & $taskSelectedPython -B -m venv $taskVenv
    if ($LASTEXITCODE -ne 0) { throw 'Python could not create .venv. Inspect the error above before retrying.' }
}

Confirm-LearningVenv
Write-Host 'Ready. Use this interpreter from Basic_Machine_Learning:'
Write-Host '  .\.venv\Scripts\python.exe -m pip install <packages-for-your-chapter>'
Write-Host '  .\.venv\Scripts\python.exe 01_linear_regression\01_linear_regression_pytorch.py'

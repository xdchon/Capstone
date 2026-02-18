param(
    [switch]$PersistForCondaEnv
)

if (-not $env:CONDA_PREFIX) {
    Write-Error "CONDA_PREFIX is not set. Activate your conda env first (e.g. conda activate cellpose-local)."
    exit 1
}

$candidatePlatforms = @(
    (Join-Path $env:CONDA_PREFIX "Library\plugins\platforms"),
    (Join-Path $env:CONDA_PREFIX "Lib\site-packages\PyQt6\Qt6\plugins\platforms"),
    (Join-Path $env:CONDA_PREFIX "lib\site-packages\PyQt6\Qt6\plugins\platforms"),
    (Join-Path $env:CONDA_PREFIX "Lib\site-packages\PySide6\plugins\platforms"),
    (Join-Path $env:CONDA_PREFIX "lib\site-packages\PySide6\plugins\platforms")
)

$platforms = $null
foreach ($candidate in $candidatePlatforms) {
    if (Test-Path (Join-Path $candidate "qwindows.dll")) {
        $platforms = $candidate
        break
    }
}

if (-not $platforms) {
    Write-Error "qwindows.dll not found in this env. Searched: $($candidatePlatforms -join '; ')"
    Write-Host "Suggested repair (choose one stack):"
    Write-Host "  conda install -n $($env:CONDA_DEFAULT_ENV) -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y"
    Write-Host "  OR"
    Write-Host "  python -m pip install --force-reinstall PyQt6 PyQt6-Qt6 qtpy pyqtgraph superqt"
    exit 1
}

$plugins = Split-Path -Parent $platforms

$env:QT_API = "pyqt6"
$env:QT_PLUGIN_PATH = $plugins
$env:QT_QPA_PLATFORM_PLUGIN_PATH = $platforms

Write-Host "Set for current shell:"
Write-Host "  QT_API=$env:QT_API"
Write-Host "  QT_PLUGIN_PATH=$env:QT_PLUGIN_PATH"
Write-Host "  QT_QPA_PLATFORM_PLUGIN_PATH=$env:QT_QPA_PLATFORM_PLUGIN_PATH"

if ($PersistForCondaEnv) {
    $actDir = Join-Path $env:CONDA_PREFIX "etc\conda\activate.d"
    $deactDir = Join-Path $env:CONDA_PREFIX "etc\conda\deactivate.d"
    New-Item -ItemType Directory -Force $actDir | Out-Null
    New-Item -ItemType Directory -Force $deactDir | Out-Null

    $activateScript = @(
        '$env:QT_API="pyqt6"',
        '$env:QT_PLUGIN_PATH="' + $plugins + '"',
        '$env:QT_QPA_PLATFORM_PLUGIN_PATH="' + $platforms + '"'
    ) -join [Environment]::NewLine
    Set-Content (Join-Path $actDir "qt-vars.ps1") $activateScript

    @"
Remove-Item Env:QT_API -ErrorAction SilentlyContinue
Remove-Item Env:QT_PLUGIN_PATH -ErrorAction SilentlyContinue
Remove-Item Env:QT_QPA_PLATFORM_PLUGIN_PATH -ErrorAction SilentlyContinue
"@ | Set-Content (Join-Path $deactDir "qt-vars.ps1")

    Write-Host "Persisted env vars for this conda env."
}

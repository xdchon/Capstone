param(
    [switch]$PersistForCondaEnv
)

if (-not $env:CONDA_PREFIX) {
    Write-Error "CONDA_PREFIX is not set. Activate your conda env first (e.g. conda activate cellpose-local)."
    exit 1
}

$plugins = Join-Path $env:CONDA_PREFIX "Library\plugins"
$platforms = Join-Path $plugins "platforms"

if (-not (Test-Path (Join-Path $platforms "qwindows.dll"))) {
    Write-Error "qwindows.dll not found at $platforms. Install Qt packages in this env first."
    Write-Host "Suggested: conda install -n $($env:CONDA_DEFAULT_ENV) -c conda-forge pyqt=6 qtpy pyqtgraph superqt qtbase -y"
    exit 1
}

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

    @"
`$env:QT_API="pyqt6"
`$env:QT_PLUGIN_PATH="`$env:CONDA_PREFIX\Library\plugins"
`$env:QT_QPA_PLATFORM_PLUGIN_PATH="`$env:CONDA_PREFIX\Library\plugins\platforms"
"@ | Set-Content (Join-Path $actDir "qt-vars.ps1")

    @"
Remove-Item Env:QT_API -ErrorAction SilentlyContinue
Remove-Item Env:QT_PLUGIN_PATH -ErrorAction SilentlyContinue
Remove-Item Env:QT_QPA_PLATFORM_PLUGIN_PATH -ErrorAction SilentlyContinue
"@ | Set-Content (Join-Path $deactDir "qt-vars.ps1")

    Write-Host "Persisted env vars for this conda env."
}

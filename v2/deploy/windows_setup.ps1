# windows_setup.ps1 - TradingBotV2 one-click setup for Windows Server 2022
# Run as Administrator in PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\windows_setup.ps1
#
# What this does:
#   1. Creates C:\TradingBotV2\ project directory
#   2. Installs Python 3.11 (if not present)
#   3. Creates a Python virtual environment
#   4. Installs TA-Lib precompiled wheel (no C compiler needed)
#   5. Installs all pip dependencies from requirements.txt
#   6. Copies project files to C:\TradingBotV2\
#   7. Creates data directory and blank .env from template
#
# After this script, run install_service.ps1 to set up auto-start.

param(
    [string]$ProjectSource = ($PSScriptRoot + "\..\..\"),
    [string]$InstallDir    = "C:\TradingBotV2"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------- Helpers -----------------------------------------------------------

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

function Assert-Admin {
    $current = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $current.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        Write-Host "ERROR: This script must be run as Administrator." -ForegroundColor Red
        exit 1
    }
}

function Get-PythonExe {
    $candidates = @(
        "python",
        "python3",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "C:\Python311\python.exe",
        "C:\Python312\python.exe"
    )
    foreach ($c in $candidates) {
        try {
            $ver = & $c --version 2>&1
            if ($ver -match "Python 3\.(1[1-9]|[2-9]\d)") {
                return $c
            }
        } catch { }
    }
    return $null
}

# ---------- 0. Require admin --------------------------------------------------

Assert-Admin

# ---------- 1. Python 3.11 ----------------------------------------------------

Write-Step "Checking Python 3.11+"
$pythonExe = Get-PythonExe

if (-not $pythonExe) {
    Write-Host "Python 3.11+ not found. Downloading installer..." -ForegroundColor Yellow

    $pythonInstaller = "$env:TEMP\python311.exe"
    $pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

    Write-Host "Downloading from $pythonUrl ..."
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing

    Write-Host "Installing Python 3.11 (silent)..."
    Start-Process -FilePath $pythonInstaller `
        -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" `
        -Wait -NoNewWindow

    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("PATH", "User")

    $pythonExe = Get-PythonExe
    if (-not $pythonExe) {
        Write-Host "ERROR: Python installation failed or PATH not updated." -ForegroundColor Red
        Write-Host "Please install Python 3.11 manually from https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
}

$pyVersion = & $pythonExe --version
Write-Host "Using: $pythonExe ($pyVersion)" -ForegroundColor Green

# ---------- 2. Create install directory ---------------------------------------

Write-Step "Creating install directory: $InstallDir"
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
New-Item -ItemType Directory -Path "$InstallDir\data" -Force | Out-Null
New-Item -ItemType Directory -Path "$InstallDir\logs" -Force | Out-Null
Write-Host "Directories created." -ForegroundColor Green

# ---------- 3. Copy project files ---------------------------------------------

Write-Step "Copying project files to $InstallDir"

$sourceRoot = Resolve-Path $ProjectSource
Write-Host "Source: $sourceRoot"

$excludeDirs  = @(".git", "__pycache__", "node_modules", ".pytest_cache", "venv", ".venv")
$excludeFiles = @("*.pyc", "*.pyo", "*.log", "*.db", ".env")

function Copy-TreeExcluding {
    param($Source, $Destination)
    Get-ChildItem -Path $Source -Recurse | ForEach-Object {
        foreach ($d in $excludeDirs) {
            if ($_.FullName -like "*\$d\*" -or $_.FullName -like "*\$d") { return }
        }
        foreach ($p in $excludeFiles) {
            if ($_.Name -like $p) { return }
        }
        $target = $_.FullName.Replace($Source, $Destination)
        if ($_.PSIsContainer) {
            New-Item -ItemType Directory -Path $target -Force | Out-Null
        } else {
            Copy-Item -Path $_.FullName -Destination $target -Force
        }
    }
}

Copy-TreeExcluding -Source $sourceRoot -Destination $InstallDir
Write-Host "Files copied." -ForegroundColor Green

# ---------- 4. Virtual environment --------------------------------------------

Write-Step "Creating Python virtual environment"
$venvPath = "$InstallDir\venv"

if (Test-Path "$venvPath\Scripts\python.exe") {
    Write-Host "Virtual environment already exists - skipping creation."
} else {
    & $pythonExe -m venv $venvPath
    Write-Host "Virtual environment created at $venvPath" -ForegroundColor Green
}

$venvPython = "$venvPath\Scripts\python.exe"
$venvPip    = "$venvPath\Scripts\pip.exe"

& $venvPython -m pip install --upgrade pip --quiet

# ---------- 5. TA-Lib precompiled wheel ---------------------------------------

Write-Step "Installing TA-Lib (precompiled wheel - no C compiler needed)"

$taLibCheck = & $venvPython -c "import talib; print('ok')" 2>&1
if ($taLibCheck -eq "ok") {
    Write-Host "TA-Lib already installed - skipping." -ForegroundColor Green
} else {
    $pyVerShort = (& $venvPython -c "import sys; print(str(sys.version_info.major) + str(sys.version_info.minor))").Trim()
    $taLibWheel = "TA_Lib-0.4.28-cp${pyVerShort}-cp${pyVerShort}-win_amd64.whl"
    $taLibUrl   = "https://github.com/cgohlke/talib-build/releases/download/v0.4.28/$taLibWheel"
    $wheelPath  = "$env:TEMP\$taLibWheel"

    Write-Host "Downloading TA-Lib wheel for Python $pyVerShort..."
    try {
        Invoke-WebRequest -Uri $taLibUrl -OutFile $wheelPath -UseBasicParsing
        & $venvPip install $wheelPath --quiet
        Write-Host "TA-Lib installed from precompiled wheel." -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Could not download precompiled TA-Lib wheel." -ForegroundColor Yellow
        Write-Host "Trying pip install TA-Lib..."
        try {
            & $venvPip install TA-Lib --quiet
            Write-Host "TA-Lib installed via pip." -ForegroundColor Green
        } catch {
            Write-Host "WARNING: TA-Lib installation failed. Bot will run without candle patterns (fallback enabled)." -ForegroundColor Yellow
        }
    }
}

# ---------- 6. pip install requirements ---------------------------------------

Write-Step "Installing Python dependencies"

$reqFile = "$InstallDir\v2\requirements.txt"
if (-not (Test-Path $reqFile)) {
    Write-Host "ERROR: requirements.txt not found at $reqFile" -ForegroundColor Red
    exit 1
}

$tempReq = "$env:TEMP\requirements_filtered.txt"
Get-Content $reqFile | Where-Object {
    $_ -notmatch "MetaTrader5" -and $_ -notmatch "^#" -and $_.Trim() -ne ""
} | Set-Content $tempReq

Write-Host "Installing from requirements.txt (this may take 3-5 minutes)..."
& $venvPip install -r $tempReq --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Some packages failed. Check output above." -ForegroundColor Yellow
} else {
    Write-Host "Dependencies installed." -ForegroundColor Green
}

# ---------- 7. MetaTrader5 package --------------------------------------------

Write-Step "Installing MetaTrader5 Python package"
& $venvPip install MetaTrader5 --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "MetaTrader5 package installed." -ForegroundColor Green
} else {
    Write-Host "WARNING: MetaTrader5 package install failed." -ForegroundColor Yellow
    Write-Host "Make sure MetaTrader5 terminal is installed before running the bot." -ForegroundColor Yellow
}

# ---------- 8. Create .env from template --------------------------------------

Write-Step "Setting up .env file"
$envFile     = "$InstallDir\.env"
$envTemplate = "$InstallDir\v2\deploy\.env.template"

if (Test-Path $envFile) {
    Write-Host ".env already exists - not overwriting." -ForegroundColor Yellow
    Write-Host "Edit $envFile to update credentials." -ForegroundColor Yellow
} else {
    if (Test-Path $envTemplate) {
        Copy-Item $envTemplate $envFile
        Write-Host ".env created from template at $envFile" -ForegroundColor Green
        Write-Host ""
        Write-Host "*** ACTION REQUIRED ***" -ForegroundColor Red
        Write-Host "Edit $envFile and fill in:" -ForegroundColor Yellow
        Write-Host "  MT5_LOGIN          = your Pepperstone MT5 account number" -ForegroundColor Yellow
        Write-Host "  MT5_PASSWORD       = your MT5 password" -ForegroundColor Yellow
        Write-Host "  MT5_SERVER         = exact server name from MT5 login screen" -ForegroundColor Yellow
        Write-Host "  BINANCE_API_KEY    = your Binance testnet API key" -ForegroundColor Yellow
        Write-Host "  BINANCE_API_SECRET = your Binance testnet secret" -ForegroundColor Yellow
        Write-Host "  TELEGRAM_BOT_TOKEN = token from @BotFather" -ForegroundColor Yellow
        Write-Host "  TELEGRAM_CHAT_ID   = your personal chat ID" -ForegroundColor Yellow
    } else {
        Write-Host "WARNING: .env.template not found at $envTemplate" -ForegroundColor Yellow
    }
}

# ---------- 9. Create startup batch file --------------------------------------

Write-Step "Creating startup batch file"
$batchLines = @(
    "@echo off",
    "REM TradingBotV2 startup script",
    "cd /d C:\TradingBotV2",
    "call venv\Scripts\activate.bat",
    "python -m v2.main",
    "pause"
)
$batchLines | Set-Content "$InstallDir\start_bot.bat" -Encoding ASCII
Write-Host "Created: $InstallDir\start_bot.bat" -ForegroundColor Green

# ---------- 10. Verify installation -------------------------------------------

Write-Step "Verifying installation"

$allOk = $true

$checkItems = @(
    @{ Label = "Python venv";    Path = "$venvPath\Scripts\python.exe" },
    @{ Label = "settings.py";    Path = "$InstallDir\v2\settings.py" },
    @{ Label = "main.py";        Path = "$InstallDir\v2\main.py" },
    @{ Label = ".env template";  Path = "$InstallDir\v2\deploy\.env.template" },
    @{ Label = "data directory"; Path = "$InstallDir\data" },
    @{ Label = "logs directory"; Path = "$InstallDir\logs" }
)

foreach ($item in $checkItems) {
    $ok = Test-Path $item.Path
    $status = if ($ok) { "[OK]" } else { "[MISSING]" }
    $color  = if ($ok) { "Green" } else { "Red" }
    Write-Host "  $status $($item.Label)" -ForegroundColor $color
    if (-not $ok) { $allOk = $false }
}

Write-Host ""
if ($allOk) {
    Write-Host "Setup complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Fill in credentials:      notepad $InstallDir\.env"
    Write-Host "  2. Install as Windows service: cd $InstallDir\v2\deploy && .\install_service.ps1"
    Write-Host "  3. Or run manually:            $InstallDir\start_bot.bat"
    Write-Host ""
    Write-Host "See v2\deploy\README_DEPLOY.md for full instructions."
} else {
    Write-Host "Setup completed with warnings. Check missing items above." -ForegroundColor Yellow
}

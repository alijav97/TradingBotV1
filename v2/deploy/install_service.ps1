# install_service.ps1 — Install TradingBotV2 as a Windows Service using NSSM
# Run as Administrator AFTER windows_setup.ps1 has completed.
#
# Usage:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\install_service.ps1
#
# To uninstall the service later:
#   .\install_service.ps1 -Uninstall
#
# What this does:
#   1. Downloads NSSM (Non-Sucking Service Manager) if not present
#   2. Installs TradingBotV2 as a Windows Service named "TradingBotV2"
#   3. Configures: auto-start, restart on failure, log rotation
#   4. Starts the service immediately

param(
    [string]$InstallDir   = "C:\TradingBotV2",
    [string]$ServiceName  = "TradingBotV2",
    [switch]$Uninstall    = $false
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ────────────────────────────────────────────────────────────────────

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

function Get-NssmPath {
    # Check common locations first
    $candidates = @(
        "$InstallDir\tools\nssm.exe",
        "C:\nssm\nssm.exe",
        "C:\tools\nssm\nssm.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    # Check PATH
    $inPath = Get-Command "nssm.exe" -ErrorAction SilentlyContinue
    if ($inPath) { return $inPath.Source }
    return $null
}

# ── 0. Admin check ─────────────────────────────────────────────────────────────

Assert-Admin

# ── Uninstall path ─────────────────────────────────────────────────────────────

if ($Uninstall) {
    Write-Step "Uninstalling service: $ServiceName"
    $nssmExe = Get-NssmPath
    if (-not $nssmExe) {
        Write-Host "NSSM not found. Trying sc.exe..." -ForegroundColor Yellow
        & sc.exe stop $ServiceName 2>$null
        & sc.exe delete $ServiceName 2>$null
    } else {
        & $nssmExe stop $ServiceName 2>$null
        & $nssmExe remove $ServiceName confirm
    }
    Write-Host "Service removed." -ForegroundColor Green
    exit 0
}

# ── 1. Validate prerequisites ──────────────────────────────────────────────────

Write-Step "Checking prerequisites"

if (-not (Test-Path "$InstallDir\venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found at $InstallDir\venv" -ForegroundColor Red
    Write-Host "Run windows_setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "$InstallDir\.env")) {
    Write-Host "ERROR: .env file not found at $InstallDir\.env" -ForegroundColor Red
    Write-Host "Copy v2\deploy\.env.template to $InstallDir\.env and fill in credentials." -ForegroundColor Yellow
    exit 1
}

# Quick sanity check — make sure MT5 credentials look non-default
$envContent = Get-Content "$InstallDir\.env" -Raw
if ($envContent -match "MT5_LOGIN=12345678" -or $envContent -match "YourMT5PasswordHere") {
    Write-Host "WARNING: .env still has placeholder values." -ForegroundColor Yellow
    Write-Host "Edit $InstallDir\.env with real credentials before continuing." -ForegroundColor Yellow
    $confirm = Read-Host "Continue anyway? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "Aborted." -ForegroundColor Red
        exit 1
    }
}

Write-Host "Prerequisites OK." -ForegroundColor Green

# ── 2. Download NSSM ──────────────────────────────────────────────────────────

Write-Step "Setting up NSSM (service manager)"
$nssmExe = Get-NssmPath

if (-not $nssmExe) {
    Write-Host "NSSM not found — downloading..."
    $nssmDir  = "$InstallDir\tools"
    $nssmZip  = "$env:TEMP\nssm.zip"
    $nssmUrl  = "https://nssm.cc/release/nssm-2.24.zip"

    New-Item -ItemType Directory -Path $nssmDir -Force | Out-Null

    try {
        Invoke-WebRequest -Uri $nssmUrl -OutFile $nssmZip -UseBasicParsing
        Expand-Archive -Path $nssmZip -DestinationPath "$env:TEMP\nssm_extract" -Force
        # NSSM zip contains nssm-2.24\win64\nssm.exe
        $extracted = Get-ChildItem "$env:TEMP\nssm_extract" -Recurse -Filter "nssm.exe" |
                     Where-Object { $_.FullName -match "win64" } |
                     Select-Object -First 1
        if (-not $extracted) {
            # Fallback: take any nssm.exe
            $extracted = Get-ChildItem "$env:TEMP\nssm_extract" -Recurse -Filter "nssm.exe" |
                         Select-Object -First 1
        }
        Copy-Item $extracted.FullName "$nssmDir\nssm.exe" -Force
        $nssmExe = "$nssmDir\nssm.exe"
        Write-Host "NSSM downloaded to $nssmExe" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Could not download NSSM." -ForegroundColor Red
        Write-Host "Download manually from https://nssm.cc/download and place nssm.exe in $nssmDir" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "NSSM found: $nssmExe" -ForegroundColor Green
}

# ── 3. Remove existing service if present ─────────────────────────────────────

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Step "Removing existing service: $ServiceName"
    & $nssmExe stop $ServiceName 2>$null
    Start-Sleep -Seconds 2
    & $nssmExe remove $ServiceName confirm
    Start-Sleep -Seconds 1
}

# ── 4. Install service ────────────────────────────────────────────────────────

Write-Step "Installing Windows Service: $ServiceName"

$pythonExe = "$InstallDir\venv\Scripts\python.exe"
$mainModule = "v2.main"
$logDir     = "$InstallDir\logs"

New-Item -ItemType Directory -Path $logDir -Force | Out-Null

# Install the service
& $nssmExe install $ServiceName $pythonExe "-m" $mainModule

# Working directory — must be set so relative imports resolve
& $nssmExe set $ServiceName AppDirectory $InstallDir

# Load .env via environment (NSSM sets service env vars from a file)
# We set each key individually from the .env file
Write-Host "Loading environment from .env..."
$envLines = Get-Content "$InstallDir\.env" | Where-Object {
    $_ -notmatch "^\s*#" -and $_ -match "="
}
foreach ($line in $envLines) {
    $kv = $line -split "=", 2
    if ($kv.Count -eq 2) {
        $key = $kv[0].Trim()
        $val = $kv[1].Trim()
        & $nssmExe set $ServiceName AppEnvironmentExtra "+$key=$val" 2>$null
    }
}

# Stdout / Stderr log files
& $nssmExe set $ServiceName AppStdout "$logDir\bot_stdout.log"
& $nssmExe set $ServiceName AppStderr "$logDir\bot_stderr.log"
& $nssmExe set $ServiceName AppStdoutCreationDisposition 4    # rotate
& $nssmExe set $ServiceName AppStderrCreationDisposition 4
& $nssmExe set $ServiceName AppRotateFiles 1
& $nssmExe set $ServiceName AppRotateSeconds 86400            # daily rotation
& $nssmExe set $ServiceName AppRotateBytes 20971520           # 20 MB max

# Restart on failure
& $nssmExe set $ServiceName AppRestartDelay 10000             # 10s delay before restart
& $nssmExe set $ServiceName AppThrottle 60000                 # throttle rapid restarts

# Service description
& $nssmExe set $ServiceName Description "TradingBotV2 — autonomous paper trading bot"
& $nssmExe set $ServiceName DisplayName "TradingBotV2"

# Start type: Automatic (delayed)
& $nssmExe set $ServiceName Start SERVICE_DELAYED_AUTO_START

Write-Host "Service installed." -ForegroundColor Green

# ── 5. Start the service ──────────────────────────────────────────────────────

Write-Step "Starting service"
& $nssmExe start $ServiceName
Start-Sleep -Seconds 3

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -eq "Running") {
    Write-Host "Service is RUNNING." -ForegroundColor Green
} else {
    Write-Host "WARNING: Service may not have started. Check logs:" -ForegroundColor Yellow
    Write-Host "  $logDir\bot_stderr.log"
    Write-Host ""
    Write-Host "To start manually: Start-Service $ServiceName"
    Write-Host "To view status:    Get-Service $ServiceName"
}

# ── 6. Summary ────────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host " TradingBotV2 Service Setup Complete" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Service name : $ServiceName"
Write-Host "Install dir  : $InstallDir"
Write-Host "Python       : $pythonExe"
Write-Host "Logs         : $logDir\"
Write-Host "API server   : http://localhost:8000"
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor Yellow
Write-Host "  Start   : Start-Service $ServiceName"
Write-Host "  Stop    : Stop-Service $ServiceName"
Write-Host "  Status  : Get-Service $ServiceName"
Write-Host "  Logs    : Get-Content $logDir\bot_stdout.log -Tail 50 -Wait"
Write-Host "  Remove  : .\install_service.ps1 -Uninstall"
Write-Host ""
Write-Host "The bot will also auto-start after every Windows reboot." -ForegroundColor Green

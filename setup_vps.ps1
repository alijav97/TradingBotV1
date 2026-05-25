# ============================================================
#  TradingBotV1 — VPS Auto-Setup Script
#  Target OS : Windows Server 2022 (fresh)
#  Run as    : Administrator  (right-click → Run with PowerShell)
# ============================================================

#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

# ── Config ────────────────────────────────────────────────────────────────────
$GITHUB_REPO  = "https://github.com/YOUR_GITHUB_USERNAME/TradingBotV1.git"
$INSTALL_DIR  = "C:\TradingBotV1"
$BOT_SCRIPT   = "C:\start_bot.ps1"
$TASK_NAME    = "TradingBotV1"
$BOT_PORT     = 8501

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  TradingBotV1 VPS Setup" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Install Chocolatey ────────────────────────────────────────────────
Write-Host "[1/13] Installing Chocolatey..." -ForegroundColor Yellow
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = `
        [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString(
        'https://community.chocolatey.org/install.ps1'))
    Write-Host "  Chocolatey installed." -ForegroundColor Green
} else {
    Write-Host "  Chocolatey already present — skipping." -ForegroundColor Green
}

# ── Step 2: Install Python 3.11 ──────────────────────────────────────────────
Write-Host "[2/13] Installing Python 3.11..." -ForegroundColor Yellow
choco install python311 -y --no-progress
Write-Host "  Python 3.11 installed." -ForegroundColor Green

# ── Step 3: Install Git ───────────────────────────────────────────────────────
Write-Host "[3/13] Installing Git..." -ForegroundColor Yellow
choco install git -y --no-progress
Write-Host "  Git installed." -ForegroundColor Green

# ── Step 4: Reload PATH (refreshenv) ─────────────────────────────────────────
Write-Host "[4/13] Refreshing environment PATH..." -ForegroundColor Yellow
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path", "User")
# Also run choco's refreshenv if available
$refreshEnv = "$env:ChocolateyInstall\bin\refreshenv.cmd"
if (Test-Path $refreshEnv) { & cmd /c $refreshEnv }
Write-Host "  PATH refreshed." -ForegroundColor Green

# ── Step 5: Clone the GitHub repo ────────────────────────────────────────────
Write-Host "[5/13] Cloning repository..." -ForegroundColor Yellow
if (Test-Path $INSTALL_DIR) {
    Write-Host "  $INSTALL_DIR already exists — pulling latest instead." -ForegroundColor Green
    Push-Location $INSTALL_DIR
    git pull origin main
    Pop-Location
} else {
    git clone $GITHUB_REPO $INSTALL_DIR
    Write-Host "  Repository cloned to $INSTALL_DIR" -ForegroundColor Green
}

# ── Step 6: cd into project ───────────────────────────────────────────────────
Write-Host "[6/13] Entering project directory..." -ForegroundColor Yellow
Set-Location $INSTALL_DIR

# ── Step 7: Create virtual environment ───────────────────────────────────────
Write-Host "[7/13] Creating Python virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "$INSTALL_DIR\venv")) {
    python -m venv venv
    Write-Host "  venv created." -ForegroundColor Green
} else {
    Write-Host "  venv already exists — skipping." -ForegroundColor Green
}

# ── Step 8: Install requirements.txt ─────────────────────────────────────────
Write-Host "[8/13] Installing requirements.txt..." -ForegroundColor Yellow
& "$INSTALL_DIR\venv\Scripts\pip.exe" install --upgrade pip --quiet
& "$INSTALL_DIR\venv\Scripts\pip.exe" install -r "$INSTALL_DIR\requirements.txt" --quiet
Write-Host "  requirements.txt installed." -ForegroundColor Green

# ── Step 9: Ensure critical packages present ─────────────────────────────────
Write-Host "[9/13] Ensuring yfinance, MetaTrader5, streamlit..." -ForegroundColor Yellow
& "$INSTALL_DIR\venv\Scripts\pip.exe" install yfinance MetaTrader5 streamlit --quiet
Write-Host "  Core packages verified." -ForegroundColor Green

# ── Step 10: Open firewall port 8501 ─────────────────────────────────────────
Write-Host "[10/13] Opening firewall port $BOT_PORT..." -ForegroundColor Yellow
$fwRuleName = "TradingBot_Streamlit_$BOT_PORT"
$existingRule = netsh advfirewall firewall show rule name=$fwRuleName 2>&1
if ($existingRule -notmatch "Rule Name") {
    netsh advfirewall firewall add rule `
        name=$fwRuleName `
        dir=in `
        action=allow `
        protocol=TCP `
        localport=$BOT_PORT | Out-Null
    Write-Host "  Firewall rule added for port $BOT_PORT." -ForegroundColor Green
} else {
    Write-Host "  Firewall rule already exists — skipping." -ForegroundColor Green
}

# ── Step 11: Create C:\start_bot.ps1 ─────────────────────────────────────────
Write-Host "[11/13] Creating $BOT_SCRIPT..." -ForegroundColor Yellow
$startBotContent = @"
# Auto-generated by setup_vps.ps1
Set-Location "$INSTALL_DIR"
& "$INSTALL_DIR\venv\Scripts\Activate.ps1"
`$env:PYTHONUTF8 = "1"

while (`$true) {
    Write-Host "Starting TradingBotV1..." -ForegroundColor Cyan
    & "$INSTALL_DIR\venv\Scripts\streamlit.exe" run bot_chat.py ``
        --server.port $BOT_PORT ``
        --server.address 0.0.0.0 ``
        --server.headless true ``
        --browser.gatherUsageStats false
    Write-Host "Bot exited — restarting in 5 seconds..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}
"@
Set-Content -Path $BOT_SCRIPT -Value $startBotContent -Encoding UTF8
Write-Host "  $BOT_SCRIPT created." -ForegroundColor Green

# ── Step 12: Register Windows Task Scheduler job ─────────────────────────────
Write-Host "[12/13] Registering Task Scheduler job '$TASK_NAME'..." -ForegroundColor Yellow

# Remove existing task if present
Unregister-ScheduledTask -TaskName $TASK_NAME -Confirm:$false -ErrorAction SilentlyContinue

$action  = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$BOT_SCRIPT`""

$trigger = New-ScheduledTaskTrigger -AtStartup

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -RunLevel Highest `
    -LogonType ServiceAccount

Register-ScheduledTask `
    -TaskName  $TASK_NAME `
    -Action    $action `
    -Trigger   $trigger `
    -Settings  $settings `
    -Principal $principal `
    -Description "TradingBotV1 Streamlit server — auto-restart on crash" | Out-Null

Write-Host "  Task '$TASK_NAME' registered (runs at startup, restarts on crash)." -ForegroundColor Green

# ── Step 13: Start the task immediately ──────────────────────────────────────
Write-Host "[13/13] Starting bot now (no reboot needed)..." -ForegroundColor Yellow
Start-ScheduledTask -TaskName $TASK_NAME
Start-Sleep -Seconds 4
$taskState = (Get-ScheduledTask -TaskName $TASK_NAME).State
Write-Host "  Task state: $taskState" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
$vpsIp = (
    Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notmatch "^127\." -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -First 1
).IPAddress

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE" -ForegroundColor Green
Write-Host "  Bot running at http://${vpsIp}:${BOT_PORT}" -ForegroundColor Green
Write-Host "  External: http://YOUR_VPS_IP:${BOT_PORT}" -ForegroundColor Green
Write-Host ""
Write-Host "  Useful commands:" -ForegroundColor Cyan
Write-Host "    Check status : Get-ScheduledTask -TaskName '$TASK_NAME'" -ForegroundColor White
Write-Host "    Stop bot     : Stop-ScheduledTask -TaskName '$TASK_NAME'" -ForegroundColor White
Write-Host "    Start bot    : Start-ScheduledTask -TaskName '$TASK_NAME'" -ForegroundColor White
Write-Host "    View logs    : Get-Content C:\TradingBotV1\streamlit_out.txt -Tail 50" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

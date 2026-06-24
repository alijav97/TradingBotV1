# which_bots_running.ps1 - list every running trading bot and identify it.
# Run in the VS Code PowerShell terminal:  .\which_bots_running.ps1
# Read-only: it only inspects processes, it does not start or stop anything.

$procs = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -match '^python' }   # python.exe and pythonw.exe

if (-not $procs) {
    Write-Host "No Python processes running - no bots are up." -ForegroundColor Yellow
    return
}

$rows = foreach ($p in $procs) {
    $cmd = [string]$p.CommandLine
    if     ($cmd -match 'eth_bot\.main')       { $bot = 'ETH bot' }
    elseif ($cmd -match 'btc_bot_1\.main')     { $bot = 'BTC bot 1' }
    elseif ($cmd -match 'btc_bot_2\.main')     { $bot = 'BTC bot 2' }
    elseif ($cmd -match 'TradingBotV2|\bv2\.') { $bot = 'WTI bot' }
    else                                       { $bot = 'other python' }

    $short = if ($cmd.Length -gt 70) { $cmd.Substring(0,70) + '...' } else { $cmd }
    [pscustomobject]@{
        Bot     = $bot
        PID     = $p.ProcessId
        Started = $p.CreationDate
        Command = $short
    }
}

$rows | Sort-Object Bot | Format-Table Bot, PID, Started, Command -AutoSize -Wrap
$up = @($rows | Where-Object { $_.Bot -ne 'other python' }).Count
Write-Host ("Bots up: {0}" -f $up) -ForegroundColor Cyan

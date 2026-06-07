@echo off
REM ============================================================================
REM  run_eth_bot.bat - launch the ETH bot in its own window (VPS).
REM
REM  Double-click this file, or run it from PowerShell:
REM      Start-Process "C:\Temp\TradingBotV1\run_eth_bot.bat"
REM
REM  It opens a dedicated "ETH BOT" console window running:
REM      C:\TradingBotV2\venv\Scripts\python.exe -m btc_research.eth_bot.main
REM
REM  This does NOT touch your WTI or BTC bots - it only starts a new ETH process.
REM
REM  WARNING: do NOT run this while an ETH bot is already running, or you will
REM  have two ETH instances scanning the same eth_trades.db. Stop the existing
REM  one first (PowerShell):
REM      Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
REM        Where-Object { $_.CommandLine -like '*eth_bot.main*' } |
REM        ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
REM ============================================================================

title ETH BOT
cd /d C:\Temp\TradingBotV1

REM Pull the latest code before launching (comment out the next line to skip).
git pull

"C:\TradingBotV2\venv\Scripts\python.exe" -m btc_research.eth_bot.main

echo.
echo ============================================================================
echo  ETH bot has exited. Review any error above. Press any key to close.
echo ============================================================================
pause >nul

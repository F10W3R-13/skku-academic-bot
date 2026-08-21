@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo [1/3] Starting regulations API...
start "SKKU Bot - API" cmd /k "python -m uvicorn api:app --port 8765"

echo [2/3] Waiting for API...
timeout /t 6 /nobreak >nul

echo [3/3] Starting WhatsApp bot...
start "SKKU Bot - WhatsApp" cmd /k "node bot.js"

echo.
echo Both windows are running. Keep this PC awake
echo (Settings > System > Power > Sleep: Never).
echo Close both windows to stop the bot.
pause

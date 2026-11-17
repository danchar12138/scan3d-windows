@echo off
rem Double-click: start Scan3D and open its page in the browser.
rem Drop a folder of photos or a video file onto this file: scan it directly.
setlocal
cd /d "%~dp0"
set "PY=tools\python\python.exe"

if not exist "%PY%" goto setup
"%PY%" app\scan3d.py doctor >nul 2>&1
if not errorlevel 1 goto run

:setup
echo Installing components (first run only)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Setup did not finish. Check the internet connection and run this file again;
  echo the download continues where it stopped.
  pause
  exit /b 1
)
"%PY%" app\scan3d.py doctor
if errorlevel 1 (
  pause
  exit /b 1
)

:run
if "%~1"=="" (
  "%PY%" app\scan3d.py serve
) else (
  "%PY%" app\scan3d.py run "%~1"
  echo.
  pause
)

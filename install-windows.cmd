@echo off
setlocal
rem Double-click entry. Bypasses the execution policy for this run only.
cd /d "%~dp0"
set "PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%PS%" set "PS=powershell.exe"
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-windows.ps1" %*
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%

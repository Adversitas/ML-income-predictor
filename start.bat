@echo off
rem Double-click to start the web app (Docker, MLflow, Prefect, first model) and open it.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\start.ps1"
if errorlevel 1 (pause) else (timeout /t 8)
